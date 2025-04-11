from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from torch import nn
import torch.nn.functional as F
import os
import warnings
import time
import argparse
from safetensors.torch import load_file

app = Flask(__name__)
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*")

# Initialize components (will be loaded on first request)
model = None
tokenizer = None
gating_network = None
regression_model = None  # Can be either linear weights or neural network
device = None

# Define attributes
attributes = [
    "Contextual_Alignment",
    "Character_Consistency",
    "Descriptive_Depth",
    "Role_Specific_Knowledge",
    "Engagement_and_Collaboration",
    "Creativity_and_Emotional_Nuance",
]

# ---------------------------
# Neural Network Architecture (Same as training)
# ---------------------------
class MultiObjectiveRegressor(nn.Module):
    def __init__(self, input_dim, num_attributes, hidden_dims=[512, 256], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.shared_backbone = nn.Sequential(*layers)
        self.attribute_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 1)
            ) for _ in range(num_attributes)
        ])
        
    def forward(self, x):
        features = self.shared_backbone(x)
        return torch.cat([head(features) for head in self.attribute_heads], dim=1)

class GatingNetwork(nn.Module):
    """Same as original gating network"""
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
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout_prob > 0:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = F.softmax(x / self.temperature, dim=1)
        return x * self.logit_scale[0]
# ---------------------------
# Model Loading
# ---------------------------
def load_model(args):
    global model, tokenizer, gating_network, regression_model, device
    
    # Set device
    device = "cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load base model
    
    # Load regression model based on method
    if not args.disable_regression:
        if args.regression_method == "neural":
            print("Loading neural regression model...")
            regression_model = MultiObjectiveRegressor(
                input_dim=4096,
                num_attributes=len(attributes),
                hidden_dims=[1024, 512, 256],
                dropout=0.3
            ).to(device)
            
            # Try loading as full state dict first, fall back to linear weights
            # save code is torch.save({"weight": torch.from_numpy(weights)}, save_path) so got to unpack
            # weights = 
            # unpack weights
            # state_dict = torch.load(args.regression_path, map_location=device)
            regression_model.load_state_dict(torch.load(args.regression_path, map_location=device))
            regression_model.eval()

        else:  # linear
            print("Loading linear regression weights...")
            regression_model = torch.load(args.regression_path, map_location=device)["weight"]
    # Load gating network if enabled
    if not args.disable_gate and args.gating_path:
        print("Loading gating network...")
        gating_network = GatingNetwork(
            4096,
            len(attributes),
            n_hidden=3,
            hidden_dim=1024,
            logit_scale=1.0,
            temperature=10,
            dropout=0.2,
        ).to(device)
        gating_network.load_state_dict(torch.load(args.gating_path, map_location=device))
        gating_network.eval()

    load_base_model(args)

def load_base_model(args):
    global model, tokenizer
    
    # Configure model loading based on precision mode and device
    if args.cpu:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16 if args.load_in == "bf16" else torch.float32
    
    if args.load_in == '8bit':
        if args.cpu:
            print("Loading 8-bit model on CPU...")

            # 8-bit on CPU
            model = AutoModel.from_pretrained(
                args.model_path,
                load_in_8bit=True,
                device_map="cpu",
                torch_dtype=torch.float16
            )
        else:
            # 8-bit on GPU
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            model = AutoModel.from_pretrained(
                args.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
    elif args.load_in == '4bit' and not args.cpu:
        # 4-bit is GPU only
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModel.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
    else:  # default to bf16 (or float32 on CPU)
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if not args.cpu else "cpu",
            attn_implementation="flash_attention_2" if not args.cpu else None,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
# ---------------------------
# Inference Logic
# ---------------------------
def get_rewards(text_embedding):
    """Calculate rewards using either linear or neural regression"""
    try:
        if isinstance(regression_model, nn.Module):  # Neural network
            with torch.no_grad():
                pred = regression_model(text_embedding)
        else:  # Linear weights
            pred = text_embedding @ regression_model.T
        
        # Apply gating if enabled
        if gating_network is not None:
            with torch.no_grad():
                gating_weights = gating_network(text_embedding)
            pred = pred * gating_weights
        
        return pred * 10  # Scale to 0-10 range
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise e

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
        start_time = time.perf_counter()
        
        if "Assistant:" not in text_input:
            conversation = [{"role": "user", "content": text_input}]
        else:
            conversation = [
                {"role": "user", "content": text_input.split("Assistant:")[0].replace("User:", "").strip()},
                {"role": "assistant", "content": text_input.split("Assistant:")[1].strip()}
            ]
        
        # Tokenize and get embeddings
        conv_formatted = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        ).replace(tokenizer.bos_token, "")
        
        with torch.no_grad():
            conv_tokenized = tokenizer(conv_formatted, return_tensors="pt").to(device)
            output = model(**conv_tokenized)
            last_hidden_state = output.last_hidden_state[0]
            text_embedding = last_hidden_state[-1].unsqueeze(0).float()
            
            # Get rewards
            pred = get_rewards(text_embedding)
        print(f"Predictions: {pred}")
        # Prepare results
        inference_time = (time.perf_counter() - start_time) * 1000
        print(f"Inference time: {inference_time:.2f} ms")
        
        results = {attr: round(score.item(), 3) for attr, score in zip(attributes, pred[0])}
        return jsonify(results)
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/mnt/m2kingston/dev/ai/models/FsfairX-LLaMA3-RM-v0.1",
                       help='Path to the base reward model')
    parser.add_argument('--regression_path', type=str, required=True,
                       help='Path to regression weights or neural model checkpoint')
    parser.add_argument('--gating_path', type=str, default="",
                       help='Path to gating network weights (optional)')
    parser.add_argument('--regression_method', type=str, default='linear',
                       choices=['linear', 'neural'],
                       help='Regression method to use')
    parser.add_argument('--load_in', type=str, default='bf16',
                       choices=['bf16', '8bit', '4bit', 'float32'],
                       help='Precision mode for model loading')
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
    
    print(f"Initializing with configuration:")
    print(f"- Regression method: {args.regression_method}")
    print(f"- Precision: {args.load_in}")
    print(f"- CPU only: {args.cpu}")
    print(f"- Gating enabled: {not args.disable_gate}")
    
    load_model(args)
    app.run(host=args.host, port=args.port, debug=False)