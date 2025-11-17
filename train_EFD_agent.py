
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
# from sparse_autoencoder.model import AutoEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os
from tqdm import tqdm
from sparse_autoencoder.autoencoder.model import SparseAutoencoderConfig
import random
import helper.model_functions as model_functions


model, tokenizer = model_functions.load_model()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_name = input("Experiment name: ")


# Load YAML config
with open("mixtral_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set random seed
torch.manual_seed(config.get("seed"))
input_dir = config["input_dir"].strip()
output_root = config["output_dir"].strip() + f"/{experiment_name}"
token_data_path = config["tokenized_prompt"].strip()


token_data = torch.load(token_data_path)
assistant_starts = token_data["assistant_start_indices"]
original_input_ids = token_data["original_input_ids"]

# Flatten assistant token indices
assistant_positions = []
pointer = 0
for seq, start in zip(original_input_ids, assistant_starts):
    for i in range(len(seq)):
        if i >= start:
            assistant_positions.append(pointer + i)
    pointer += len(seq)
assistant_positions = set(assistant_positions)







model_config = config["model"]
training_config = config["training"]
seed = config["seed"]

print(f"Raw input_dir from config: '{config['input_dir']}'")
print(os.path.abspath(config["input_dir"]))


try:
    if os.path.exists(input_dir):
        print("input from config file found continuing the program")
        pass
    else:
        raise FileNotFoundError("input from config file not found")
except FileNotFoundError as e:
    print(f"error: input from config file not found at '{input_dir} error {e}")
    print("continuing the program")


def train_sae_acitvation_single_layer(activations, output_dir, model_cfg, training_cfg, seed, device):


    torch.manual_seed(seed)
    # Load activations

    data = torch.tensor(activations, dtype=torch.float32)

    print(f" Loaded {data.shape[0]} activation rows (filtered and averaged)")

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    # Model config
    print("configuring SAE model")


    # ✅ Convert dict to SparseAutoencoderConfig
    sae_config = SparseAutoencoderConfig(
        n_input_features=model_cfg["d_in"],

        n_learned_features= (model_cfg["d_in"]*model_cfg['expansion_factor']),
        n_components=None  # or set this if doing multi-component
    )

    print("sae config:", sae_config)



    print("instantiating SAE model")

    # Instantiate model
    sae_model = SparseAutoencoder(config=sae_config).to(device)
    print(sae_model)

    # Optimizer
    optimizer_name = training_config["optimizer"].lower()
    if optimizer_name == "adam":

        optimizer = torch.optim.Adam(sae_model.parameters(), lr=float(training_cfg["lr"]))
        l1_lambda = float(training_cfg["l1_lambda"])
    else:
        raise ValueError("Unknown optimizer")

    # Training loop
    os.makedirs(output_dir, exist_ok=True)
    loss_log = []

    print("assigning early stopping values")

    best_loss = float("inf")
    patience = 10
    patience_counter = 0
    print("Starting training...")
    for epoch in range(training_cfg["epochs"]):
        sae_model.train()
        epoch_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{training_cfg['epochs']}"):
            x = batch[0].to(device)
            optimizer.zero_grad()
            hidden, x_hat = sae_model(x)

            # MSE loss + L1 sparsity
            mse_loss = F.mse_loss(x_hat, x)
            l1_loss = l1_lambda * hidden.abs().mean()
            loss = mse_loss + l1_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_log.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(sae_model.state_dict(), os.path.join(output_dir, f"best_model_sae.pth"))
        else:
            patience_counter += 1
            print(f"no improvement. Pateience counter: {patience_counter}/{patience} ")
            if patience_counter >= patience:
                print("early stopping triggered at epoch", epoch+1, " for layer name", layer_name)
                print("log to text file.txt")
                with open(os.path.join(output_dir, "early_stop_log.txt"), "w") as f:
                    f.write(f"Stopped at epoch {epoch + 1} with best loss: {best_loss:.6f}\n")
                break

    np.save(os.path.join(output_dir, "loss_log.npy"), np.array(loss_log))

    print("Training complete. Model saved to:", output_dir)

    # Plot loss curve
    plt.plot(loss_log)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss (MSE + L1)")
    plt.title(f"Training Loss Curve {layer_name}")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"loss_curve.png"))
    plt.show()

    print(" Run model on all data to capture top-K activations")
    all_hidden = []

    sae_model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            hidden, _ = sae_model(x)
            all_hidden.append(hidden.cpu())

    all_hidden = torch.cat(all_hidden, dim=0)  # [N, d_hidden]
    k = min(10, all_hidden.shape[0])
    print(f"top {k} activation are being selected")
    non_zero_mask = (all_hidden.abs().sum(dim=1) != 0)
    if non_zero_mask.sum() < all_hidden.shape[0]:
        print(f"removing {all_hidden.shape[0] - non_zero_mask.sum()} rows with all zero features or padding activations from topk selection")

    all_hidden = all_hidden[non_zero_mask]
    topk_values, topk_indices = torch.topk(all_hidden, k=k, dim=0)  # top-10 examples per neuron

    np.save(os.path.join(output_dir, "topk_indices.npy"), topk_indices.numpy())
    np.save(os.path.join(output_dir, "topk_values.npy"), topk_values.numpy())
    print(f"finished training on {activation_file}. saved to {output_dir}")
import re

print(os.listdir(input_dir))

special_token_ids = set(tokenizer.convert_tokens_to_ids(t) for t in ["<s>", "</s>", "[PAD]", "[INST]", "[/INST]"])
for fnmae in sorted(os.listdir(input_dir), key=lambda x: max(map(int, re.findall(r"\d+", x))), reverse=True):
    print("processing", fnmae)
    if fnmae.endswith(".npy"):
        layer_name = fnmae.split(".npy")[0]
        activation_file = os.path.join(input_dir, fnmae)
        print("loading activations...", activation_file)
        activations = np.load(activation_file)

        activation_to_token_index = np.load(
            os.path.join("/media/nima/T7 Shield/empathy_layers_assistant_training/mixtral_residuals_all_layers/activation_to_token_indices",
                                    f"{layer_name.split('_filtered')[0]}_token_indices.npy"))
        activation_to_token_index = activation_to_token_index.reshape(-1, 2)[:, 0]
        # Mask activations to assistant only
        assistant_mask = [i for i, tok_idx in enumerate(activation_to_token_index) if tok_idx in assistant_positions]

        # Just checking assistant only before passing to SAE
        flattened_input_ids = [token for seq in token_data["original_input_ids"] for token in seq]

        # Remove [PAD] and other special tokens from assistant
        clean_mask = [i for i in assistant_mask if flattened_input_ids[activation_to_token_index[i]] not in special_token_ids]
        activations = activations[clean_mask]


        #Saniyu check
        context_window = 10
        for i in random.sample(clean_mask, 150):
            pointer = 0
            token_pos = activation_to_token_index[i]
            print(tokenizer.decode([token_pos]))

            for seq in original_input_ids:
                if token_pos < pointer + len(seq):
                    rel_idx = token_pos - pointer
                    context_ids = seq[max(0, rel_idx - context_window): rel_idx + context_window + 1]
                    token_id = seq[rel_idx]
                    token_str = tokenizer.decode([token_id])
                    # if token_str == "":
                    #     continue
                    context_str = tokenizer.decode(context_ids,skip_special_tokens=True)
                    print(f"Row {i}: token = '{token_str}' | context: ...{context_str}...")
                    break
                pointer += len(seq)
        # END sanity check


        output_dir = os.path.join(output_root, layer_name)
        if os.path.exists(output_dir):
            print(f"skipping {layer_name} - already trained")
            continue
        print("training sae for layer " + layer_name)
        train_sae_acitvation_single_layer(activations = activations, output_dir=output_dir, model_cfg=model_config, training_cfg=training_config, seed=seed, device=device)