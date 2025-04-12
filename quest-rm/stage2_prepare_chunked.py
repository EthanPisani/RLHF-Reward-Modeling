import os
import torch
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig
from tqdm.auto import tqdm
from safetensors.torch import save_file
from argparse import ArgumentParser

# Set up CUDA optimizations for faster computation
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Define token patterns for gating different model families
token_patterns = {
    "llama3": [128009, 128006, 78191, 128007, 271],
    "gemma2": [107, 108, 106, 2516, 108],
}

def find_token_for_gating(lst, model_family):
    """Find the last occurrence of a token pattern in a list."""
    token_pattern = token_patterns[model_family]
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j: j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")

# Initialize argument parser
parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
parser.add_argument("--model_family", type=str, default="llama3")
parser.add_argument("--dataset_path", type=str, default="RLHFlow/UltraFeedback-preference-standard")
parser.add_argument("--source", default=None, type=str)
parser.add_argument("--dataset_split", type=str, default="train")
parser.add_argument("--n_shards", type=int, default=1)
parser.add_argument("--shard_idx", type=int, default=1)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--seq_len", type=int, default=8192)
args = parser.parse_args()

config = AutoConfig.from_pretrained(args.model_path)


config.output_hidden_states = True
# config.output_attentions = True
if args.model_family == "llama3":
    assert config.model_type == "llama"
elif args.model_family == "gemma2":
    assert config.model_type == "gemma2"
else:
    raise ValueError(f"Model family {args.model_family} is not supported")

HOME = os.path.expanduser("~")
model_name = args.model_path.split("/")[-1]
dataset_name = args.dataset_path.split("/")[-1]
save_path = f"{HOME}/data/ArmoRM/embeddings/{model_name}/{dataset_name}"
if args.source is not None:
    save_path += f"-{args.source}"
save_path += f"-{args.dataset_split}"

ds = datasets.load_dataset(args.dataset_path, split=args.dataset_split)
if args.source is not None:
    ds = ds.filter(lambda x: x["source"] == args.source)
if args.n_shards > 1:
    ds = ds.shuffle(seed=0)
    ds = ds.shard(num_shards=args.n_shards, index=args.shard_idx - 1)

device = f"cuda:{args.device}"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit quantization
    llm_int8_threshold=6.0,  # Default value; adjust if needed for lower precision
    llm_int8_has_fp16_weight=False,  # Ensure compatibility with your model
)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # Load the model in 4-bit precision
#     bnb_4bit_quant_type="nf4",
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_compute_dtype=torch.float16,
# )
bnb_config.model_type = config.model_type
# Load the model
# model = AutoModel.from_pretrained(args.model_path, quantization_config=bnb_config, device_map="auto")
# model = AutoModel.from_pretrained(args.model_path, device_map=device)
# from optimum.onnxruntime import ORTModelForSequenceClassification
# # export model
# config = AutoConfig.from_pretrained(args.model_path)

# config.output_attentions = True
# config.output_hidden_states = True
# model = ORTModelForSequenceClassification.from_pretrained(
#     args.model_path,

#     provider="CUDAExecutionProvider",
#     config=config,
    
#                                     output_hidden_states=True,
#                                     output_attentions=True
# )

# model = ORTModelForSequenceClassification.from_pretrained(
#     args.model_path,
#     # torch_dtype=torch.bfloat16,
#     # device=device,
#   export=True,
#   provider="CUDAExecutionProvider",
# )
model = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,  # Use bfloat16 precision for model weights to save memory
    #  quantization_config=bnb_config,
    device_map=device,
    attn_implementation="flash_attention_2",  # Specify the attention implementation for efficiency
)
# model = model.to_bettertransformer()
from torch.nn.attention import SDPBackend
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
embeddings = []
prompt_embeddings = []
file_counter = 0
chunk_size = 50000

def save_and_reset():
    """Save current embeddings to a safetensors file and reset lists."""
    global embeddings, prompt_embeddings, file_counter
    if embeddings:
        file_counter += 1
        os.makedirs(save_path, exist_ok=True)
        save_file(
            {
                "embeddings": torch.stack(embeddings),
                "prompt_embeddings": torch.stack(prompt_embeddings),
            },
            os.path.join(save_path, f"{file_counter:04d}.safetensors"),
        )
        print(f"Saved {len(embeddings)} items to {file_counter:04d}.safetensors")
        embeddings.clear()
        prompt_embeddings.clear()

for idx, example in enumerate(tqdm(ds, desc="Processing examples")):
    chosen = example["chosen"]
    rejected = example["rejected"]

    if "prompt" in example:
        chosen = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": chosen},
        ]
        rejected = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": rejected},
        ]

    pair_embeddings = []
    pair_prompt_embeddings = []

    for iter_example in [chosen, rejected]:
        if args.model_path.endswith("FsfairX-LLaMA3-RM-v0.1"):
            conv_formatted = tokenizer.apply_chat_template(
                iter_example, tokenize=False, add_generation_prompt=False
            ).replace(tokenizer.bos_token, "")
        else:
            conv_formatted = tokenizer.apply_chat_template(iter_example, tokenize=False)

        conv_tokenized = tokenizer(conv_formatted, return_tensors="pt").to(device)
        input_ids = conv_tokenized["input_ids"]

        if input_ids.shape[1] > args.seq_len:
            continue
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            with torch.no_grad():
                # print(model)
                output = model(**conv_tokenized)
                # get attributes from output
                last_hidden_state = output.last_hidden_state[0]

                # Find the position of the gating token and extract embeddings
                gating_token_position = find_token_for_gating(
                    input_ids[0].tolist(), args.model_family
                )
                prompt_embedding = last_hidden_state[gating_token_position].cpu()
                last_token_embedding = last_hidden_state[-1].cpu()

                pair_embeddings.append(last_token_embedding)
                pair_prompt_embeddings.append(prompt_embedding)

    if len(pair_embeddings) == 2:
        embeddings.append(torch.stack(pair_embeddings))
        prompt_embeddings.append(torch.stack(pair_prompt_embeddings))

    if len(embeddings) >= chunk_size:
        save_and_reset()

# Save remaining items
save_and_reset()

print(f"Processing completed. All files saved to {save_path}.")
