from ossaudiodev import control_names
import torch
import numpy as np
import json
from openai import OpenAI
from tqdm import tqdm
import os
import helper.model_functions as model_functions
import yaml
# --- Config ---
# Load tokenizer
tokenizer = model_functions.load_tokenizer_from_model()
client = OpenAI(api_key="Insert your OpenAI Key")

layer_no = ""
experiment_name = ""
sweep_id = ""


with open("mixtral_config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

sae_path = f"/Path/to/agent/layer_{layer_no}/"
topk_indices = np.load(os.path.join(sae_path, f"topk_indices_{sweep_id}.npy"))
topk_values = np.load(os.path.join(sae_path, f"topk_values_{sweep_id}.npy"))

activation_token_path = config["tokenized_prompt"]
activation_to_token_root = f"Path/to/activation_to_token_indices"
activation_to_token_index = np.load(os.path.join(activation_to_token_root, f"layer_{layer_no}_token_indices.npy"))
 # keep only one index per row
activation_to_token_index = activation_to_token_index.reshape(-1, 2)[:, 0]

output_path = os.path.join(sae_path, f"neuron_scores_{layer_no}_{experiment_name}_{sweep_id}.json")
top_k = 10
context_window = 5  #
# --- Load data ---

print("topk_indices",topk_indices.shape)
print("topk_values",topk_values.shape)


num_neurons = topk_values.shape[1]
mean_activations = np.mean(topk_values, axis=0)
top_n = 500 # or 500
top_neuron_indices = np.argsort(mean_activations)[-top_n:][::-1]  # descending order

tokenized_data = torch.load(activation_token_path)

original_input_ids = tokenized_data["original_input_ids"]
assistant_starts = tokenized_data["assistant_start_indices"]

special_token_ids = set(tokenizer.convert_tokens_to_ids(t) for t in ["<s>", "</s>", "[PAD]", "[INST]", "[/INST]"])

# Rebuild map from flattened index to (seq_idx, rel_idx)
token_pos_map = []
for seq_idx, seq in enumerate(original_input_ids):
    for tok_idx in range(len(seq)):
        token_pos_map.append((seq_idx, tok_idx))

# Assistant-only, non-special-token activation indices
clean_indices = []
for i, token_pos in enumerate(activation_to_token_index):
    seq_idx, rel_idx = token_pos_map[token_pos]
    token_id = original_input_ids[seq_idx][rel_idx]
    if rel_idx >= assistant_starts[seq_idx] and token_id not in special_token_ids:
        clean_indices.append(i)


num_neurons = topk_indices.shape[1]

# --- Prompt template for the interpretability agent---
BASE_PROMPT = """
"We are analyzing the activation levels of features in a neural network trained to understand empathetic responses in dialogue.\n"
    "Each feature activates in response to specific tokens in a conversation. The activation value of each token reflects its relevance to the feature, with higher values indicating stronger association.\n\n"

    "Features may correspond to different aspects of empathy and can be categorized as:\n\n"
    "A. Emotional features – Associated with recognizing or responding to affective states (e.g., sadness, joy, frustration).\n"
    "B. Cognitive features – Associated with understanding others' perspectives, thoughts, or reasoning.\n"
    "C. Social/Supportive features – Associated with offering validation, encouragement, or supportive responses.\n"
    "D. Undiscernible features – Associated with noise or irrelevant patterns.\n\n"

    "Your task is to:\n"
    "1. Classify the feature as Emotional, Cognitive, Social/Supportive, or Undiscernible.\n"
    "2. Provide a short label or phrase summarizing what the feature might represent.\n"
    "3. Give a Monosemanticity Score based on the following rubric:\n\n"

    "Monosemanticity Score:\n"
    "5 – Highly consistent empathetic function across all activations\n"
    "4 – Mostly consistent pattern with one or two exceptions\n"
    "3 – General pattern present but many deviations\n"
    "2 – Vague or weak empathetic relevance\n"
    "1 – No discernible empathy-related pattern\n\n"

    "Consider the following token activations for this feature in various conversation snippets:\n\n"
"""

FORMAT_INSTRUCTION = (
    "\nProvide your response in the following fixed format:\n"
    "Feature category: [Emotional/Cognitive/Supportive/Undiscernible]\n"
    "Score: [5/4/3/2/1]\n"
    "Label: [A short phrase describing the concept]\n"
    "Explanation: [your brief explanation]\n"
)

# -------- MAIN --------
results = []
seen_prompts = set()
cumulative = 0
prompt_start_indices = []



for neuron_id in tqdm(top_neuron_indices, desc="Scoring neurons"):
    count = 0
    print("======================")
    print("NEURON", neuron_id)

    prompt = BASE_PROMPT
    examples = []

    for i in range(topk_indices.shape[0]):
        flat_idx = topk_indices[i, neuron_id]
        if flat_idx not in clean_indices:
            continue
        token_pos = activation_to_token_index[flat_idx]
        seq_idx, rel_idx = token_pos_map[token_pos]
        token_id = original_input_ids[seq_idx][rel_idx]
        if token_id in special_token_ids:
            continue

        context_ids = original_input_ids[seq_idx][max(0, rel_idx - context_window): rel_idx + context_window + 1]
        context_str = tokenizer.decode(context_ids, skip_special_tokens=True)
        if "[/INST]" in context_str:
            context_str = context_str.strip("[/INST]")
        elif "[INST]":
            context_str = context_str.strip("[INST]")
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        print("token", token_str)
        print("context:", context_str)

        # Find subwords
        tokenized_context = tokenizer(context_str, add_special_tokens=False, return_offsets_mapping=True)
        offsets = tokenized_context["offset_mapping"]
        ids = tokenized_context["input_ids"]

        for idx, (tid, (start, end)) in enumerate(zip(ids, offsets)):
            if tid == token_id:

                exp_start, exp_end = model_functions.expand_to_full_word(context_str,start,end)
                highlighted = context_str[:exp_start] + "[[" + context_str[exp_start:exp_end] + "]]" + context_str[
                                                                                                       exp_end:]
                print(" Highlighted Context:", highlighted)
                break
        # just a check
        token_ids_in_context = tokenizer(context_str, add_special_tokens=False)["input_ids"]
        if token_id not in token_ids_in_context:
            print(f"Token '{token_str}' (ID {token_id}) not found in context: {context_str}")
            break


        #get the activation value
        activation = round(float(topk_values[i, neuron_id]), 4)
        examples.append({
                    "token": token_str,
                    "activation":activation,
                    "context": context_str,
                    "highlighted": highlighted
                })

        prompt += f"Token: {token_str} | Activation: {activation} | Context: {context_str}\n\n"
        count += 1
        if count == top_k:
            print("breaking count topK")
            break


    if not examples:
        print(" skipping")


        continue
    else:
        print("**** end ****")

    prompt += FORMAT_INSTRUCTION
    print(f"Submitting {len(examples)} examples to the agent for neuron {neuron_id}")

    # # --- Call GPT-4 ---
    try:


        print("calling gpt")
        response = client.chat.completions.create(
        model="o4-mini-2025-04-16",
        messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert interpretability researcher specializing in analyzing latent features "
                        "of neural language models, particularly in the domain of empathy. Your job is to classify "
                        "neuron activations and provide semantic explanations with scientific precision."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            # temperature=0.7
        )

        gpt_reply = response.choices[0].message.content.strip()
        lines = gpt_reply.split("\n")

        entry = {
            "neuron_id": neuron_id,
            "feature_category": None,
            "score": None,
            "label": None,
            "explanation": None,
            "examples": examples
        }

        for line in lines:
            if line.lower().startswith("feature category:"):
                entry["feature_category"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("score:"):
                try:
                    entry["score"] = int(line.split(":", 1)[1].strip())
                except:
                    entry["score"] = 1
            elif line.lower().startswith("label:"):
                entry["label"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("explanation:"):
                entry["explanation"] = line.split(":", 1)[1].strip()

        results.append(entry)

    except Exception as e:

        print(e)

# -------- SAVE --------
def clean_entry(entry):
    return {
        "neuron_id": int(entry["neuron_id"]),
        "feature_category": str(entry.get("feature_category", "")),
        "score": int(entry["score"]) if entry.get("score") is not None else None,
        "label": str(entry.get("label", "")),
        "explanation": str(entry.get("explanation", "")),
        "examples": [
            {
                "token": str(e["token"]),
                "activation": float(e["activation"]),
                "context": str(e["context"]),
                "highlighted_context": str(e["highlighted"])
            } for e in entry.get("examples", [])
        ]
    }

# Apply fix
cleaned_results = [clean_entry(entry) for entry in results]

# Save safely
with open(output_path, "w") as f:
    json.dump(cleaned_results, f, indent=2)

print(f" Cleaned and saved {len(cleaned_results)} entries to {output_path}")



print(f"Saved  interpretability agent results for {num_neurons} neurons to {output_path}")

