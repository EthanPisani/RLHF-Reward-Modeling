import os
import torch
import numpy as np
from safetensors.torch import load_file
from torch import nn
import torch.nn.functional as F
import torch
from glob import glob
from transformers import AutoTokenizer, AutoModel

# Define token patterns for gating different model families
token_patterns = {
    # Llama3 token IDs of "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "llama3": [128009, 128006, 78191, 128007, 271],
    # Gemma2 token IDs of "<end_of_turn>\n<start_of_turn>model\n"
    "gemma2": [107, 108, 106, 2516, 108],
}

def find_token_for_gating(input_ids, model_family):
    """
    Find the position of the gating token in the input sequence
    based on the model family.
    """
    pattern = token_patterns.get(model_family)
    if pattern is None:
        return -1  # default to last token if no pattern matches
    
    # Search for the pattern in the input_ids
    for i in range(len(input_ids) - len(pattern) + 1):
        if input_ids[i:i+len(pattern)] == pattern:
            return i + len(pattern) - 1  # return last position of the pattern
    return -1  # if pattern not found, return last token

# Define the attributes for multi-objective reward modeling
attributes = [
    "Contextual_Alignment",
    "Character_Consistency",
    "Descriptive_Depth",
    "Role_Specific_Knowledge",
    "Engagement_and_Collaboration",
    "Creativity_and_Emotional_Nuance",
]

class GatingNetwork(nn.Module):
    """
    Gating Network: A simple MLP with softmax output and temperature scaling
    This network learns to combine multiple reward objectives based on the input context
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        temperature: float = 10,
        logit_scale: float = 1.0,
        hidden_dim: int = 1024,
        n_hidden: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale)
        self.dropout_prob = dropout
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Apply the linear layers with ReLU and dropout
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout_prob > 0:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
        # Apply softmax with temperature scaling
        x = F.softmax(x / self.temperature, dim=1)
        return x * self.logit_scale[0]

def load_embeddings(embedding_path_pattern, device):
    """
    Load embeddings from safetensors files
    """
    file_paths = glob(embedding_path_pattern)
    if len(file_paths) == 0:
        raise ValueError(f"Embeddings not found at {embedding_path_pattern}")
    embeddings, prompt_embeddings = [], []
    for embedding_path in file_paths:
        embeddings_data = load_file(embedding_path)
        embeddings.append(embeddings_data["embeddings"].to(device))
        prompt_embeddings.append(embeddings_data["prompt_embeddings"].to(device))
    embeddings = torch.cat(embeddings, dim=0).float()
    prompt_embeddings = torch.cat(prompt_embeddings, dim=0).float()
    return embeddings, prompt_embeddings

def infer_attributes(text_input, model_path, gating_network_path, device="cuda:0"):
    # Load embeddings and regression layer
    embedding_path = f"./embeddings/{model_path.split('/')[-1]}/pair_data_v2_80K_wsafety-train/0001.safetensors"
    regression_layer_path = f"./stage1_train_model3/{model_path.split('/')[-1]}_model3.pt"
    
    embeddings, prompt_embeddings = load_embeddings(embedding_path, device=device)
    regression_layer = torch.load(regression_layer_path, map_location=device)["weight"]

    # Load gating network
    gating_network = GatingNetwork(
        prompt_embeddings.shape[-1],
        regression_layer.shape[0],
        n_hidden=3,
        hidden_dim=1024,
        logit_scale=1.0,
        temperature=10,
        dropout=0.2,
    ).to(device)
    gating_network.load_state_dict(torch.load(gating_network_path, map_location='cpu'))
    gating_network.eval()

    # Set up CUDA optimizations for faster computation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rm = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
    ).to(device)

    rm_tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Format the input as a conversation
    conversation = [
        {"role": "user", "content": text_input.split("Assistant:")[0].replace("User:", "").strip()},
        {"role": "assistant", "content": text_input.split("Assistant:")[1].strip()}
    ]

    # Format the conversation using the tokenizer's chat template
    if model_path.endswith("FsfairX-LLaMA3-RM-v0.1"):
        conv_formatted = rm_tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        ).replace(rm_tokenizer.bos_token, "")
    else:
        conv_formatted = rm_tokenizer.apply_chat_template(conversation, tokenize=False)

    # Tokenize the formatted conversation
    conv_tokenized = rm_tokenizer(conv_formatted, return_tensors="pt").to(device)
    input_ids = conv_tokenized["input_ids"]

    # Perform inference
    with torch.no_grad():
        output = rm(**conv_tokenized)
        last_hidden_state = output.last_hidden_state[0]

        # Find the gating token position based on model family
        model_family = "llama3" if "llama3" in model_path.lower() else "gemma2" if "gemma2" in model_path.lower() else None
        gating_token_position = find_token_for_gating(input_ids[0].tolist(), model_family)
        
        # Use the gating token position if found, otherwise use last token
        if gating_token_position == -1:
            text_embedding = last_hidden_state[-1]
        else:
            text_embedding = last_hidden_state[gating_token_position]
        
        # Add batch dimension (convert from 1D to 2D tensor)
        text_embedding = text_embedding.unsqueeze(0).float()  # Shape: [1, hidden_dim]

        gating_weights = gating_network(text_embedding)
        pred = text_embedding @ regression_layer.T * gating_weights
        pred *= 10  # Scale the predictions

    # Map predictions to attributes
    attribute_scores = {attr: score.item() for attr, score in zip(attributes, pred[0])}
    return attribute_scores

# Example usage
if __name__ == "__main__":
    model_path = "/mnt/m2kingston/dev/ai/models/FsfairX-LLaMA3-RM-v0.1"
    gating_network_path = "/home/redzuzu/data/ArmoRM/gating_network_FsfairX-LLaMA3-RM-v0.1.pt"

    text_input = """User: You are a medieval blacksmith, and I want to buy a sword.
Assistant: Welcome, traveler! I have short swords, longswords, and greatswords. Each is forged with the finest steel in the land. What do you desire?"""

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    attribute_scores = infer_attributes(text_input, model_path, gating_network_path, device)
    print("Attribute Scores:")
    for attr, score in attribute_scores.items():
        print(f"{attr}: {score:.4f}")