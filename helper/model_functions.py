from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch import nn
import torch
from sparse_autoencoder import SparseAutoencoder, SparseAutoencoderConfig
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score
print("loading model functions from helper")
# Load the fine-tuned model and tokenizer
def load_model():
    model_path = "PATH/TO/MODEL"

    # Optimized BitsAndBytesConfig for reduced memory usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # float16 to save GPU memory
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distribute across GPUs if available
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Ensure `pad_token` is distinct from `eos_token`
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def create_prompt(sample):
    bos_token = "<s>"
    eos_token = "</s>"
    input_text = sample["user"].strip()
    output_text = sample["assistant"].strip()
    return f"{bos_token} [INST] {input_text} [/INST] {output_text}{eos_token}"


def clean_response(response, max_sentences=4):
    words = response.split()
    deduplicated = []
    for word in words:
        if not deduplicated or word != deduplicated[-1]:
            deduplicated.append(word)
    cleaned_response = " ".join(deduplicated)

    sentences = re.split(r'(?<=[.!?]) +', cleaned_response)
    return " ".join(sentences[:max_sentences])


def load_tokenizer_from_model():
    model_path = "PATH/TO/MODEL"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Ensure `pad_token` is distinct from `eos_token`
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer

# Expand to the full alphabetic word span
def expand_to_full_word(s, start, end):
    while start > 0 and s[start - 1].isalpha():
        start -= 1
    while end < len(s) and s[end].isalpha():
        end += 1
    return start, end

def load_sae_from_state_dict(path, device, return_feature_size=False):
    print("loading sae model on device", device)
    try:
        state_dict = torch.load(path, map_location=device)
        # Infer dimensions
        W_enc = state_dict["encoder.weight"]
        n_learned_features, n_input_features = W_enc.shape

        # Construct config
        config = SparseAutoencoderConfig(
            n_input_features=n_input_features,
            n_learned_features=n_learned_features,
            n_components=None)

        model = SparseAutoencoder(config).to(device)
        model.eval()
        if return_feature_size:
            print("returning feature size")
            expansion_factor = int(n_learned_features / n_input_features)
            return model, n_input_features, n_learned_features,expansion_factor
        else:
            # print("NOT returning feature size")
            return model
    except RuntimeError as e:
        print(f"[ERROR] Failed to load model at {path}")
        print(f"↪ {str(e)}")
        return None

def compute_metrics(user, assistant, ablated):

    # Load the Sentence-BERT model for cosine similarity
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    pairs = [
        (ablated, assistant),
        (user, assistant),
        (user, ablated)
    ]
    results = []
    for s1, s2 in pairs:
        emb1 = embedding_model.encode([s1])
        emb2 = embedding_model.encode([s2])
        cos_sim = cosine_similarity(emb1, emb2)[0][0]
        P, R, F1 = score([s1], [s2], lang="en", verbose=False)
        results.append({
            "pair": (s1, s2),
            "cosine": float(cos_sim),
            "bert_precision": float(P.mean()),
            "bert_recall": float(R.mean()),
            "bert_f1": float(F1.mean())
        })
    return results

