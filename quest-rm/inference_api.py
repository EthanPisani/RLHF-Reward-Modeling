from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from torch import nn
import torch.nn.functional as F
from glob import glob
from safetensors.torch import load_file
import os
import warnings
import time
import argparse

app = Flask(__name__)
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*")

# Initialize model and tokenizer (will be loaded on first request)
model = None
tokenizer = None
gating_network = None
regression_layer = None
device = None  # Will be set based on CLI arguments

# Define attributes
attributes = [
    "Contextual_Alignment",
    "Character_Consistency",
    "Descriptive_Depth",
    "Role_Specific_Knowledge",
    "Engagement_and_Collaboration",
    "Creativity_and_Emotional_Nuance",
]

# Configuration
MODEL_PATH = "/mnt/m2kingston/dev/ai/models/FsfairX-LLaMA3-RM-v0.1"
GATING_NETWORK_PATH = "/home/redzuzu/data/ArmoRM/gating_network_FsfairX-LLaMA3-RM-v0.1.pt"
EMBEDDING_PATH = f"./embeddings/FsfairX-LLaMA3-RM-v0.1/pair_data_v2_80K_wsafety-train/0001.safetensors"
REGRESSION_LAYER_PATH = f"./stage1_train_model2/FsfairX-LLaMA3-RM-v0.1_model2.pt"

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

def load_model(precision_mode='bf16', cpu_only=False, disable_gate=False, disable_regression=False):
    global model, tokenizer, gating_network, regression_layer, device
    
    # Set device based on arguments and availability
    if cpu_only:
        device = "cpu"
        torch_dtype = torch.float32  # Use float32 for CPU
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    
    print(f"Using device: {device}")
    
    # Load gating network and regression layer
    if disable_regression:
        regression_layer = None
    else:
        regression_layer = torch.load(REGRESSION_LAYER_PATH, map_location=device)["weight"]
    
    if disable_gate:
        gating_network = None
    else:
        gating_network = GatingNetwork(
            4096,
            regression_layer.shape[0],
            n_hidden=3,
            hidden_dim=1024,
            logit_scale=1.0,
            temperature=10,
            dropout=0.2,
        ).to(device)


        gating_network.load_state_dict(torch.load(GATING_NETWORK_PATH, map_location='cpu'))
        gating_network.eval()
    
    print(f"Initial Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB" if device.startswith('cuda') else "Running on CPU")
    
    # Configure model loading based on precision mode and device
    if cpu_only:
        # CPU-only mode - use float32 and no quantization
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
    elif precision_mode == '8bit':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map=device if not cpu_only else "cpu",
            attn_implementation="flash_attention_2" if not cpu_only else None,
        )
    elif precision_mode == '4bit':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map=device if not cpu_only else "cpu",
            attn_implementation="flash_attention_2" if not cpu_only else None,
        )
    else:  # default to bf16 (or float32 on CPU)
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_dtype,
            device_map=device if not cpu_only else "cpu",
            attn_implementation="flash_attention_2" if not cpu_only else None,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if device.startswith('cuda'):
        print(f"CUDA Memory Allocated after loading: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

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
            output = model(**conv_tokenized)
            last_hidden_state = output.last_hidden_state[0]
            text_embedding = last_hidden_state[-1].unsqueeze(0).float()
            
            if gating_network is not None:
                gating_weights = gating_network(text_embedding)

                pred = text_embedding @ regression_layer.T * gating_weights
            else:
                pred = text_embedding @ regression_layer.T 
            pred *= 10
        print(pred)
        # Prepare results
        print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        results = {attr: round(score.item(), 3) for attr, score in zip(attributes, pred[0])}
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_in', type=str, default='bf16',
                       choices=['bf16', '8bit', '4bit'],
                       help='Precision mode for model loading (bf16, 8bit, or 4bit)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU-only inference')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address to run the server on')
    parser.add_argument('--port', type=int, default=5002,
                       help='Port to run the server on')
    parser.add_argument('--disable_gate', action='store_true',
                          help='Disable gating network')
    parser.add_argument('--disable_regression', action='store_true',
                          help='Disable regression layer')
    args = parser.parse_args()
    
    print(f"Loading model in {args.load_in} precision mode...")
    print(f"CPU-only mode: {args.cpu}")
    load_model(args.load_in, args.cpu, args.disable_gate, args.disable_regression)

    
    app.run(host=args.host, port=args.port, debug=False)