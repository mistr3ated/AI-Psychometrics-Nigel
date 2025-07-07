#!/usr/bin/env python3
"""
GPT-2 Model Checkpoint Tester with Nucleus Sampling
Tests model checkpoints as they're saved during training
Usage: python test_checkpoints_nucleus.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
import glob
import os
import json
import time
from datetime import datetime
import argparse

# ============================================================================
# MODEL ARCHITECTURE (Must match your training script)
# ============================================================================
class GPT2Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.05, seq_len=1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)
    
    def forward(self, x):
        residual = x
        x = self.ln1(x)
        mask = self.causal_mask[:x.size(1), :x.size(1)]
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = residual + attn_out
        
        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        return x

class GPT2Mini(nn.Module):
    def __init__(self, vocab_size=50_000, seq_len=1024, n_layer=8, n_head=8, n_embd=512, dropout=0.05):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_embd = n_embd
        
        # Core model components
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, n_embd))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[GPT2Block(n_embd, n_head, dropout, seq_len) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_emb.weight

    def forward(self, x):
        B, T = x.size()
        
        # Token + positional embeddings
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb[:, :T, :]
        
        x = token_emb + pos_emb
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

# ============================================================================
# TESTING FUNCTIONS
# ============================================================================
def load_tokenizer():
    """Load the tokenizer (same as training script)"""
    vocab_file = "./my_tokenizer/vocab.json"
    merges_file = "./my_tokenizer/merges.txt"
    
    if os.path.exists(vocab_file) and os.path.exists(merges_file):
        tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
        print(f"✅ Loaded existing tokenizer (vocab size: {tokenizer.get_vocab_size()})")
        return tokenizer
    else:
        print(f"❌ Tokenizer files not found in my_tokenizer/ directory.")
        print(f"   Looking for: {vocab_file} and {merges_file}")
        print(f"   Make sure training has created the tokenizer.")
        return None

def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint with _orig_mod prefix handling"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract model config
        config = checkpoint.get('config', {
            'vocab_size': 50_000,
            'seq_len': 1024,
            'n_layer': 8,
            'n_head': 8, 
            'n_embd': 512,
            'dropout': 0.05
        })
        
        # Create model
        model = GPT2Mini(**config)
        
        # Get the model state dict and handle _orig_mod prefix
        model_state = checkpoint.get('model', checkpoint)  # Handle best_model.pt (state_dict only)
        
        # Remove _orig_mod. prefix if present (from torch.compile)
        if any(key.startswith('_orig_mod.') for key in model_state.keys()):
            cleaned_state = {}
            for key, value in model_state.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]  # Remove the prefix
                    cleaned_state[new_key] = value
                else:
                    cleaned_state[key] = value
            model_state = cleaned_state
        
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        
        # Get training info
        epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        
        # Try to get loss from metrics if available
        train_loss = checkpoint.get("metrics", {}).get("current_step_loss", 0.0)
                
        print(f"✅ Loaded checkpoint: epoch {epoch}, step {global_step}, loss {train_loss:.4f}")
        return model, epoch, global_step, train_loss
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None, 0, 0, 0

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, nucleus_p=0.9, device='cuda'):
    """Generate text from the model using nucleus sampling"""
    model.eval()
    
    # Tokenize prompt
    encoded = tokenizer.encode(prompt)
    tokens = torch.tensor([encoded.ids], device=device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = model(tokens)
            
            # Get next token logits and apply temperature
            next_token_logits = logits[0, -1, :] / temperature
            
            
            # Apply repetition penalty
            for token_id in set(generated_tokens):
                next_token_logits[token_id] /= 1.5  # Adjust strength as needed
                       
            # Apply nucleus (top-p) filtering first
            probs = F.softmax(next_token_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            nucleus_mask = cumulative_probs <= nucleus_p
            nucleus_mask[0] = True  # Keep at least one token

            # Apply top-k within nucleus
            valid_probs = sorted_probs[nucleus_mask]
            valid_indices = sorted_indices[nucleus_mask]

            # Sample from the valid tokens
            next_token_idx = torch.multinomial(valid_probs, 1)
            next_token = valid_indices[next_token_idx]

            # Add to sequence
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            generated_tokens.append(next_token.item())
            
            # Stop if we hit max sequence length
            if tokens.size(1) >= model.seq_len:
                break
    
    # Decode generated text
    full_tokens = encoded.ids + generated_tokens
    generated_text = tokenizer.decode(full_tokens)
    
    return generated_text

def test_checkpoint(checkpoint_path, tokenizer, device, test_prompts):
    """Test a single checkpoint with temperature and nucleus_p sweep"""
    print(f"\n{'='*60}")
    print(f"Testing checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Load model
    model, epoch, step, train_loss = load_checkpoint(checkpoint_path, device)
    if model is None:
        return
    
    print(f"Training info: Epoch {epoch}, Step {step}, Loss {train_loss:.4f}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test different sampling parameters
    test_configs = [
        (0.3, 0.95), # (temperature, nucleus_p)
        (0.5, 0.9),
        (0.7, 0.85),
        (0.8, 0.8),
        (0.9, 0.75),  
        (1.0, 0.70),
        (1.2, 0.65),
    ]
    
    for temp, nucleus_p in test_configs:
        print(f"\n🔥 Temperature: {temp}, Nucleus p: {nucleus_p}")
        print("-" * 60)
        
        for i, prompt in enumerate(test_prompts, 1):
            try:
                generated = generate_text(model, tokenizer, prompt, 
                                        max_length=100, temperature=temp, 
                                        nucleus_p=nucleus_p, device=device)
                # Extract just the generated part (after the prompt)
                generated_part = generated[len(prompt):]
                print(f"\n{i}. '{prompt}'")
                print(f"   ➜ {generated_part}")
                
            except Exception as e:
                print(f"\n{i}. '{prompt}'")
                print(f"   ➜ ❌ Generation failed: {e}")
        
        print(f"\n")
    
    # Clean up
    del model
    torch.cuda.empty_cache()

def monitor_checkpoints(checkpoint_dir, tokenizer, device, test_prompts, interval=300):
    """Monitor for new checkpoints and test them"""
    print(f"🔍 Monitoring {checkpoint_dir} for new checkpoints...")
    print(f"⏰ Check interval: {interval} seconds")
    
    tested_checkpoints = set()
    
    while True:
        # Find all checkpoint files
        
        checkpoint_patterns = [
            os.path.join(checkpoint_dir, "ckpt-*.pt"),
            os.path.join(checkpoint_dir, "best_model.pt")
        ]
        checkpoints = []
        for pattern in checkpoint_patterns:
            checkpoints.extend(glob.glob(pattern))
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        # Test new checkpoints
        for checkpoint_path in checkpoints:
            if checkpoint_path not in tested_checkpoints:
                test_checkpoint(checkpoint_path, tokenizer, device, test_prompts)
                tested_checkpoints.add(checkpoint_path)
        
        # Wait before next check
        print(f"\n⏱️  Waiting {interval} seconds for next check...")
        time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="Test GPT-2 model checkpoints with nucleus sampling")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", 
                       help="Directory containing checkpoints")
    parser.add_argument("--checkpoint", default=None, 
                       help="Specific checkpoint file to test")
    parser.add_argument("--list", action="store_true",
                       help="List available checkpoints")
    parser.add_argument("--latest", action="store_true", default=True,
                       help="Test latest checkpoint (default)")
    parser.add_argument("--monitor", action="store_true", 
                       help="Monitor for new checkpoints")
    parser.add_argument("--interval", type=int, default=300, 
                       help="Monitoring interval in seconds")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    args = parser.parse_args()
    
    print(f"🚀 GPT-2 Checkpoint Tester (Nucleus Sampling)")
    print(f"📂 Checkpoint directory: {args.checkpoint_dir}")
    print(f"🔧 Device: {args.device}")
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    if tokenizer is None:
        return
    
    # Test prompts
    test_prompts = [
        "An elephant is bigger than a",
        "2+2=",
        "The sky is",
        "Paris is the capital of",
        "The day after Friday is",
        "Hola, cómo estás?",
        "The month with the fewest days is",
        "Personality is long term patterns",
        "Intelligence is the ability to",
        "Do open-source-LLM's rock?",
        "Psychometrics is",
        "He turned to her and said",
        "She turned to him and said",
        "The best career advice i ever received is"
    ]
    
    device = torch.device(args.device)
    
    # Find available checkpoints
    checkpoint_pattern = os.path.join(args.checkpoint_dir, "ckpt-*.pt")
    checkpoints = glob.glob(checkpoint_pattern)
    checkpoints.sort(key=os.path.getmtime, reverse=True)  # newest first
    
    if args.list:
        # List available checkpoints
        print(f"\n📋 Available checkpoints in {args.checkpoint_dir}:")
        if checkpoints:
            for i, cp in enumerate(checkpoints):
                mtime = datetime.fromtimestamp(os.path.getmtime(cp))
                print(f"  {i+1}. {os.path.basename(cp)} (modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print("  No checkpoints found.")
    elif args.checkpoint:
        # Test specific checkpoint
        if os.path.exists(args.checkpoint):
            test_checkpoint(args.checkpoint, tokenizer, device, test_prompts)
        else:
            print(f"❌ Checkpoint not found: {args.checkpoint}")
    elif args.monitor:
        # Monitor for new checkpoints
        monitor_checkpoints(args.checkpoint_dir, tokenizer, device, 
                          test_prompts, args.interval)
    else:
        # Test latest checkpoint (default behavior)
        if checkpoints:
            latest_checkpoint = checkpoints[0]  # First in sorted list (newest)
            print(f"\n🔍 Testing latest checkpoint: {os.path.basename(latest_checkpoint)}")
            test_checkpoint(latest_checkpoint, tokenizer, device, test_prompts)
        else:
            print(f"❌ No checkpoints found in {args.checkpoint_dir}")
            print(f"💡 Available commands:")
            print(f"   python test_checkpoints_nucleus.py --list                    # List checkpoints")
            print(f"   python test_checkpoints_nucleus.py                          # Test latest")
            print(f"   python test_checkpoints_nucleus.py --checkpoint path.pt     # Test specific")
            print(f"   python test_checkpoints_nucleus.py --monitor                # Auto-monitor")

if __name__ == "__main__":
    main()