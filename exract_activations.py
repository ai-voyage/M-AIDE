'''extracts activation after experts'''
import tkinter as tk
from tkinter import scrolledtext
import json
import numpy as np
import torch
from datasets import Dataset
import helper.model_functions as model_functions
import re
import os


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# load the model
model, tokenizer = model_functions.load_model()
print("model loaded")


print("loading dataset")


with open("Path/to/Dataset.json", "r") as f:
    json_data = json.load(f)


print("formating prompt and dataset")


formatted_samples = [model_functions.create_prompt(sample) for sample in json_data]
dataset = Dataset.from_dict({"text": formatted_samples})

print("tokenizing dataset")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="longest",
        padding_side="right",
        max_length=1024,  # Adjust to your model's max sequence length
        return_tensors="pt",
        add_special_tokens=False
    )

print("Tokenizing dataset...")
tokenized_data = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Separate only on assistant
print(" Determine where assistant tokens start")

inst_token_ids =  tokenizer("[/INST]", add_special_tokens=False)["input_ids"]

def find_subsequence(full_list, sub_list):
    for i in range(len(full_list) - len(sub_list) + 1):
        if full_list[i:i + len(sub_list)] == sub_list:
            return i
    return -1

assistant_starts = []
for ids in tokenized_data["input_ids"]:
    idx = find_subsequence(ids, inst_token_ids)
    if idx != -1:
        assistant_starts.append(idx + len(inst_token_ids))
    else:
        print("error on assistant starts")
        assistant_starts.append(0)


for i, ids in enumerate(tokenized_data["input_ids"][:5]):
    decoded = tokenizer.decode(ids)
    print(f"\nDecoded [{i}]:", decoded)
    start = assistant_starts[i]
    assistant_tokens = ids[start:]
    assistant_text = tokenizer.decode(assistant_tokens, skip_special_tokens=True)

    print("Assistant starts at:", start)
    print("Token at assistant start:", assistant_text)
    print("=======================")

# --- Filter out [PAD] and special tokens ---
special_token_ids = set(tokenizer.convert_tokens_to_ids(t) for t in ["<s>", "</s>", "[PAD]", "[INST]", "[/INST]"])
clean_input_ids = []
clean_attention_mask = []

for ids, mask in zip(tokenized_data["input_ids"], tokenized_data["attention_mask"]):
    clean_ids = []
    clean_mask = []
    for token, m in zip(ids, mask):
        if token not in special_token_ids and m == 1:
            clean_ids.append(token)
            clean_mask.append(1)
    clean_input_ids.append(clean_ids)
    clean_attention_mask.append(clean_mask)

print(f"Tokenized and filtered. Kept {sum(len(seq) for seq in clean_input_ids)} tokens")


# Save flattened attention mask used for activation filtering later
token_data_filename = "/Path/to/activation_token_data_assistant.pt"
token_data_root = "/Path/to/root/"
if os.path.exists(token_data_filename):
    os.remove(token_data_filename)
    print("removed old attention mask file")
else:
    os.makedirs(token_data_root, exist_ok=True)




torch.save( {
        "input_ids": clean_input_ids,
        "attention_mask": clean_attention_mask,
        "assistant_start_indices": assistant_starts,
        "original_input_ids": tokenized_data["input_ids"],
        "original_attention_mask": tokenized_data["attention_mask"]
    }, token_data_filename)
print("Saved token data to", token_data_filename)



from torch.utils.data import DataLoader
# dataloader = DataLoader(tokenized_data, batch_size=4)
from transformers import default_data_collator
dataloader = DataLoader(
    tokenized_data,
    batch_size=4,
    collate_fn=default_data_collator  # ✅ fixes the list-of-dict issue
)

print("hook setup")

NUM_LAYERS = len(model.model.layers)
residual_batch_buffers = {f"layer_{i}": [] for i in range(NUM_LAYERS)}

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        out = output[0].detach().cpu()  # [B, T, 4096]
        residual_batch_buffers[f"layer_{layer_idx}"].append(out)
    return hook_fn

# Register hooks
for idx, block in enumerate(model.model.layers):
    block.register_forward_hook(make_hook(idx))


print("extract residuals.....")
from tqdm import tqdm
model.eval()

print("Starting residual extraction...")
output_dir = "/Path/to/mixtral_residuals_all_layers"
os.makedirs(output_dir, exist_ok=True)

activation_token_indices_path = os.path.join(output_dir, "activation_to_token_indices")
os.makedirs(activation_token_indices_path, exist_ok=True)
token_data = torch.load(token_data_filename)
clean_mask_seqs = token_data["attention_mask"]
# # Manual layer selection
# layer_token_indices = {
#     "layer_7": [],
#     "layer_15": [],
#     # ....
#     "layer_31": []
# }
layer_token_indices = {f"layer_{i}": [] for i in range(32)}
pointers = {layer: 0 for layer in layer_token_indices}
with torch.no_grad():
    for step, batch in enumerate(tqdm(dataloader)):
        # Clear old buffers
        for k in residual_batch_buffers:
            residual_batch_buffers[k].clear()

        # Run forward
        _ = model(input_ids=batch["input_ids"].to(device),
                  attention_mask=batch["attention_mask"].to(device))

        # For each layer, save this batch’s residuals to .npy
        for layer_name, batch_list in residual_batch_buffers.items():

            batch_tensor = torch.cat(batch_list, dim=0)  # [B, T, 4096]
            flat = batch_tensor.reshape(-1, batch_tensor.shape[-1])  # [B*T, 4096]
            # Dynamically compute per-step mask
            special_token_ids = set(
                tokenizer.convert_tokens_to_ids(t) for t in ["<s>", "</s>", "[PAD]", "[INST]", "[/INST]"])
            flat_mask = []

            for ids in batch["input_ids"]:
                for token in ids:
                    flat_mask.append(0 if token in special_token_ids else 1)
            batch_clean_mask = torch.tensor(flat_mask, dtype=torch.bool)
            filtered = flat[batch_clean_mask]
            global_indices = torch.arange(flat.shape[0]) + pointers[layer_name]
            kept_indices = global_indices[batch_clean_mask]
            layer_token_indices[layer_name].extend(kept_indices.tolist())
            pointers[layer_name] += flat.shape[0]

            out_path = os.path.join(output_dir, f"{layer_name}_batch_{step:04d}.npy")

            print("saving batch")
            np.save(out_path, filtered.numpy())

# Save after all steps
for layer_name, indices in layer_token_indices.items():
    out_path = os.path.join(activation_token_indices_path, f"{layer_name}_token_indices.npy")
    np.save(out_path, np.array(indices))
    print(f"Saved {len(indices)} indices for {layer_name} to {out_path}")
