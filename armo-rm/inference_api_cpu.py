from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn
import torch.nn.functional as F
from glob import glob
from safetensors.torch import load_file
import os
import warnings
import time

# Suppress some warnings to keep the output clean
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*")

app = Flask(__name__)

# Initialize model and tokenizer (will be loaded on first request)
model = None
tokenizer = None
gating_network = None
regression_layer = None
device = "cpu"  # Force CPU inference

# Define attributes
attributes = [
    "Contextual_Alignment",
    "Character_Consistency",
    "Descriptive_Depth",
    "Role_Specific_Knowledge",
    "Engagement_and_Collaboration",
    "Creativity_and_Emotional_Nuance",
]

# Configuration - Update these paths as needed
MODEL_PATH = "/mnt/m2kingston/dev/ai/models/FsfairX-LLaMA3-RM-v0.1"
GATING_NETWORK_PATH = "/home/redzuzu/data/ArmoRM/gating_network_FsfairX-LLaMA3-RM-v0.1.pt"
EMBEDDING_PATH = f"./embeddings/FsfairX-LLaMA3-RM-v0.1/pair_data_v2_80K_wsafety-train/0001.safetensors"
REGRESSION_LAYER_PATH = f"./stage1_train_model3/FsfairX-LLaMA3-RM-v0.1_model3.pt"

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

def load_components():
    global model, tokenizer, gating_network, regression_layer
    
    # Load regression layer
    regression_layer = torch.load(REGRESSION_LAYER_PATH, map_location=device)["weight"]
    
    # Initialize and load gating network
    gating_network = GatingNetwork(
        4096,
        regression_layer.shape[0],
        n_hidden=3,
        hidden_dim=1024,
        logit_scale=1.0,
        temperature=10,
        dropout=0.2,
    ).to(device)
    gating_network.load_state_dict(torch.load(GATING_NETWORK_PATH, map_location=device))
    gating_network.eval()
    
    # Load tokenizer and model with CPU optimizations
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Load model with appropriate settings for CPU
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,  # Use float32 on CPU for stability
        device_map=None,  # No device map for CPU
        low_cpu_mem_usage=True,  # Optimize memory usage
    ).to(device)
    model.eval()

# Load components at startup
load_components()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():

    try:
        data = request.get_json()
        text_input = data.get('text', '')
        
        if not text_input:
            return jsonify({"error": "No text provided"}), 400
        
        # Format as conversation
        if "Assistant:" not in text_input:
            conversation = [{"role": "user", "content": text_input}]
        else:
            conversation = [
                {"role": "user", "content": text_input.split("Assistant:")[0].replace("User:", "").strip()},
                {"role": "assistant", "content": text_input.split("Assistant:")[1].strip()}
            ]
        start = time.perf_counter()
        # Format and tokenize
        conv_formatted = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        ).replace(tokenizer.bos_token, "")
        conv_tokenized = tokenizer(conv_formatted, return_tensors="pt").to(device)
        
        # Inference
        with torch.no_grad():
            # Use inference mode for better CPU performance
            with torch.inference_mode():
                output = model(**conv_tokenized)
                last_hidden_state = output.last_hidden_state[0]
                text_embedding = last_hidden_state[-1].unsqueeze(0).float()
                
                gating_weights = gating_network(text_embedding)
                pred = text_embedding @ regression_layer.T * gating_weights
                pred *= 10
        print(pred)
        print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        # Prepare results
        results = {attr: round(score.item(), 3) for attr, score in zip(attributes, pred[0])}
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Disable Flask debug mode for production
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)