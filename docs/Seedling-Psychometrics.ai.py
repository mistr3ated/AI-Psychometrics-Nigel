# Seedling-v102.py
# Everything on https://psychometrics.ai is shared under a Creative Commons Attribution-NonCommercial 4.0 International License. 
# You're very welcome to use the ideas, methods, and tools in your own work, whether that’s product development, internal tools, client services, or research. 
# All I ask is that you don’t sell or repackage this content as-is, and you credit psychometrics.ai when using it: 
# Guenole, N. (2025). Psychometrics.ai: AI for Psychological Measurement. https://psychometrics.ai.

# ============================================================================
# SECTION 0: IMPORTS AND DEPENDENCIES
# ============================================================================
import logging
import psutil
import threading
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import glob
import json
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt

# ============================================================================
# SECTION 1: LOGGING AND SYSTEM MONITORING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

def log_system_usage(interval=60):
    """System monitoring with uptime tracking"""
    start_time = time.time()
    while True:
        try:
            uptime = time.time() - start_time
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory().percent
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                gpu_max = torch.cuda.max_memory_allocated() / 1024**3
                logging.info(f"📊 Uptime: {format_time(uptime)} | CPU: {cpu:.1f}% | RAM: {mem:.1f}% | GPU: {gpu_mem:.1f}GB/{gpu_max:.1f}GB")
            else:
                logging.info(f"📊 Uptime: {format_time(uptime)} | CPU: {cpu:.1f}% | RAM: {mem:.1f}%")
        except Exception as e:
            logging.warning(f"System monitoring error: {e}")
        time.sleep(interval)

# Start system monitoring
monitor_thread = threading.Thread(target=log_system_usage, daemon=True)
monitor_thread.start()

def format_time(seconds):
    """Convert seconds to readable format"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

# ============================================================================
# SECTION 2: CUDA AND DEVICE SETUP
# ============================================================================
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.info(f"✅ CUDA available: {torch.cuda.is_available()}")
logging.info(f"🖥️ GPU count: {torch.cuda.device_count()}")
logging.info(f"💻 Using device: {device}")

if torch.cuda.is_available():
    logging.info(f"🔥 GPU: {torch.cuda.get_device_name()}")
    logging.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# ============================================================================
# SECTION 3: IMPROVED GPT-2 MODEL ARCHITECTURE
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
        # Causal mask
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
    def __init__(self, vocab_size, seq_len=1024, n_layer=8, n_head=8, n_embd=512, dropout=0.05):
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
        
        # FIXED: Improved weight initialization
        self.apply(self._init_weights)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        
        # Log model size
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"🧠 Model size: {total_params:,} parameters ({total_params/1e6:.1f}M)")
        logging.info(f"🔧 Model config: vocab={vocab_size}, layers={n_layer}, heads={n_head}, embd={n_embd}")

    def _init_weights(self, module):
        """FIXED: Better weight initialization to prevent gradient explosion"""
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization with proper scaling
            std = math.sqrt(1.0 / (module.in_features + module.out_features))
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Smaller std for embeddings to prevent initial instability
            nn.init.normal_(module.weight, mean=0.0, std=0.005)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            if module.dim() > 1:
                # Very small initialization for positional embeddings
                nn.init.normal_(module, mean=0.0, std=0.005)

    def forward(self, x):
        B, T = x.size()
        
        # Input validation
        if T > self.seq_len:
            raise ValueError(f"Sequence length {T} exceeds maximum {self.seq_len}")
        
        # Token + positional embeddings
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb[:, :T, :]
        
        x = token_emb + pos_emb
        
        # Check for NaN after embeddings
        if torch.isnan(x).any():
            logging.error("❌ NaN detected after embedding layer!")
            raise ValueError("NaN in embeddings")
        
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Final check before output
        if torch.isnan(x).any():
            logging.error("❌ NaN detected before output head!")
            raise ValueError("NaN before output")
        
        logits = self.head(x)
        
        if torch.isnan(logits).any():
            logging.error("❌ NaN detected in output logits!")
            raise ValueError("NaN in output logits")
        
        return logits

# ============================================================================
# SECTION 4: MEMORY-EFFICIENT DATASET CLASS (UNCHANGED)
# ============================================================================
class MemoryEfficientDataset(IterableDataset):
    def __init__(self, file_path, seq_len=1024, buffer_size=1000):
        self.file_path = file_path
        self.seq_len = seq_len
        self.buffer_size = buffer_size
        self.min_seq_len = 5
        
    def __iter__(self):
        buffer = []
        skipped_lines = 0
        processed_lines = 0
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    tokens = list(map(int, line.split()))
                    
                    if len(tokens) < self.min_seq_len + 1:
                        skipped_lines += 1
                        continue
                    
                    input_len = min(len(tokens) - 1, self.seq_len)
                    
                    input_seq = tokens[:input_len]
                    target_seq = tokens[1:input_len + 1]
                    
                    buffer.append((
                        torch.tensor(input_seq, dtype=torch.long),
                        torch.tensor(target_seq, dtype=torch.long)
                    ))
                    processed_lines += 1
                    
                    if len(buffer) >= self.buffer_size:
                        if processed_lines % 10000 == 0:
                            logging.info(f"📊 Processed {processed_lines} sequences, skipped {skipped_lines}")
                        
                        import random
                        random.shuffle(buffer)
                        for item in buffer:
                            yield item
                        buffer = []
                        
                except (ValueError, MemoryError, RuntimeError) as e:
                    logging.warning(f"Skipping problematic line {line_num}: {e}")
                    skipped_lines += 1
                    continue
                except Exception as e:
                    logging.error(f"Unexpected error at line {line_num}: {e}")
                    skipped_lines += 1
                    continue
            
            if buffer:
                import random
                random.shuffle(buffer)
                for item in buffer:
                    yield item
        
        logging.info(f"📊 Dataset complete: {processed_lines} sequences processed, {skipped_lines} skipped")

# ============================================================================
# SECTION 5: DATA PREPARATION (UNCHANGED)
# ============================================================================
def setup_data_and_tokenizer():
    """Setup data and tokenizer - only if needed"""
    
    # 1. Download dataset if needed
    if not os.path.exists("samples/pile_sample.txt"):
        logging.info("📥 Downloading dataset...")
        import subprocess
        import sys
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade",
            "datasets[streaming]>=2.16.0", "zstandard", "pyarrow", "transformers"
        ])
        
        from datasets import load_dataset
        
        # Set cache directories
        for var in ("HF_DATASETS_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE"):
            os.environ.pop(var, None)
        os.environ["HF_DATASETS_CACHE"] = "./hf_cache"
        os.environ["HF_HOME"] = "./hf_home"
        os.environ["HUGGINGFACE_HUB_CACHE"] = "./hf_cache"
        
        os.makedirs("samples", exist_ok=True)
        output_file = "samples/pile_sample.txt"
        
        dataset = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
        target_bytes = int(10.0 * 1024**3)  # 10GB
        current_size = 0
        chunks = 0
        
        with open(output_file, "w", encoding="utf-8") as out_f:
            for item in tqdm(dataset, desc="📥 Sampling dataset"):
                text = item.get("text", "").strip()
                if len(text) < 200:
                    continue
                out_f.write(text + "\n\n")
                current_size += len(text.encode("utf-8"))
                chunks += 1
                if current_size >= target_bytes:
                    break
        
        logging.info(f"✅ Downloaded {current_size/1024**2:.1f}MB in {chunks} chunks")
    else:
        logging.info("✅ Dataset already exists")
    
    # 2. Train tokenizer if needed
    clean_file = "samples/pile_sample_clean.txt"
    os.makedirs("samples", exist_ok=True)
    if not os.path.exists(clean_file):
        logging.info("🧹 Cleaning dataset for tokenizer training...")
        with open("samples/pile_sample.txt", "r", encoding="utf-8") as fin, open(clean_file, "w", encoding="utf-8") as fout:
            import string, re
            for line in fin:
                line = line.strip()
                if (
                    len(line) < 100 or 
                    line.count(" ") < 10 or 
                    sum(c.isalpha() for c in line) / len(line) < 0.5 or
                    sum(c in string.punctuation for c in line) / len(line) > 0.2 or
                    re.search(r"(.)\1{5,}", line)
                ):
                    continue
                fout.write(line + "\n")
        logging.info("✅ Cleaned dataset saved")
    else:
        logging.info("✅ Cleaned tokenizer dataset already exists")

    output_dir = "my_tokenizer"
    if not os.path.exists(os.path.join(output_dir, "vocab.json")):
        logging.info("🔤 Training tokenizer...")
        os.makedirs(output_dir, exist_ok=True)
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[clean_file],
            vocab_size=50_000,
            min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
        )
        tokenizer.save_model(output_dir)

        with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
            json.dump({"add_prefix_space": True, "unk_token": "<unk>"}, f)

        logging.info("✅ Tokenizer trained and saved")
    else:
        logging.info("✅ Tokenizer already exists")
    
    
   
    
    # 3. Tokenize dataset if needed
    if not os.path.exists("pile_tokenized_train.txt"):
        logging.info("🔤 Tokenizing dataset (memory efficient)...")
        tokenizer = ByteLevelBPETokenizer("my_tokenizer/vocab.json", "my_tokenizer/merges.txt")
        
        MAX_SEQ_LEN = 1024
        
        train_count = 0
        val_count = 0
        
        with open("samples/pile_sample.txt", "r", encoding="utf-8") as fin, \
            open("pile_tokenized_train.txt", "w", encoding="utf-8") as train_out, \
            open("pile_tokenized_val.txt", "w", encoding="utf-8") as val_out:
            
            for line_idx, line in enumerate(tqdm(fin, desc="🔤 Tokenizing")):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    ids = tokenizer.encode(line).ids
                    for i in range(0, len(ids), MAX_SEQ_LEN):
                        chunk = ids[i:i+MAX_SEQ_LEN]
                        if len(chunk) < 10:  # Skip very short sequences
                            continue
                        chunk_str = " ".join(map(str, chunk))
                        
                        if line_idx % 5 == 0:
                            val_out.write(chunk_str + "\n")
                            val_count += 1
                        else:
                            train_out.write(chunk_str + "\n")
                            train_count += 1
                            
                except Exception as e:
                    logging.warning(f"Tokenization error on line {line_idx}: {e}")
                    continue
        
        logging.info(f"✅ Tokenized: {train_count} train, {val_count} val sequences")
    else:
        logging.info("✅ Tokenized datasets already exist")

def filter_tokenized_data():
    """Minimal preprocessing: filter out punctuation-heavy sequences"""
    
    def is_good_sequence(token_ids):
        """Check if sequence has reasonable token distribution"""
        if len(token_ids) < 10:
            return False
            
        # Count "bad" tokens (adjust ranges based on your vocab inspection)
        bad_count = 0
        for token_id in token_ids:
            # Typical ranges for punctuation/numbers in BPE tokenizers
            if (token_id < 50 or  # Very early tokens (often punctuation)
                token_id > 45000 or  # Very late tokens (often rare/broken)
                token_id in [40, 41, 91, 93]):  # Common punct token IDs
                bad_count += 1
        
        # Reject if >30% bad tokens
        return (bad_count / len(token_ids)) < 0.3
    
    # Filter training data
    print("🔧 Filtering training data...")
    good_sequences = 0
    total_sequences = 0
    
    with open("pile_tokenized_train.txt", "r") as infile, \
        open("pile_tokenized_train_clean.txt", "w") as outfile:
        
        for line in infile:
            total_sequences += 1
            try:
                token_ids = list(map(int, line.strip().split()))
                if is_good_sequence(token_ids):
                    outfile.write(line)
                    good_sequences += 1
            except:
                continue
                
            if total_sequences % 10000 == 0:
                print(f"  Processed {total_sequences}, kept {good_sequences}")
    
    # Filter validation data  
    print("🔧 Filtering validation data...")
    with open("pile_tokenized_val.txt", "r") as infile, \
        open("pile_tokenized_val_clean.txt", "w") as outfile:
        
        for line in infile:
            try:
                token_ids = list(map(int, line.strip().split()))
                if is_good_sequence(token_ids):
                    outfile.write(line)
            except:
                continue
    
    print(f"✅ Filtered data: kept {good_sequences}/{total_sequences} sequences ({100*good_sequences/total_sequences:.1f}%)")
    
# ============================================================================
# SECTION 6:  TRAINING FUNCTION
# ============================================================================
SAVE_EVERY_STEPS = 1000


def train_model():
    """FIXED: Main training function with better stability"""

    # Initialize metrics at the start of the function
    metrics = {
        "train_losses": [],
        "val_losses": [],
        "learning_rates": []
    }
    training_start_time = time.time()
    epoch_times = []
    logging.info(f"⏰ Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    CKPT_DIR = "./checkpoints"
    os.makedirs(CKPT_DIR, exist_ok=True)
    checkpoint = None
    
    # Check for existing checkpoints
    ckpts = glob.glob(os.path.join(CKPT_DIR, "ckpt-*.pt"))
    start_epoch = 0
     
    if ckpts:
        latest_ckpt = max(ckpts, key=os.path.getmtime)
        logging.info(f"🔄 Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location="cpu")
        
        if checkpoint:
            metrics = checkpoint.get("metrics", {
                "train_losses": [],
                "val_losses": [],
                "learning_rates": []
            })
        else:
            metrics = {
                "train_losses": [],
                "val_losses": [],
                "learning_rates": []
            }
       
        logging.info(f"📍 Resuming from epoch {start_epoch}")
    else:
        logging.info("🚀 Starting fresh training")

    # Create datasets
    train_dataset = MemoryEfficientDataset("pile_tokenized_train_clean.txt", seq_len=1024, buffer_size=500)
    val_dataset = MemoryEfficientDataset("pile_tokenized_val_clean.txt", seq_len=1024, buffer_size=100)

    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(batch):
        input_seqs, target_seqs = zip(*batch)
        
        valid_pairs = []
        for inp, tgt in zip(input_seqs, target_seqs):
            if len(inp) >= 5 and len(tgt) >= 5 and len(inp) <= 1024 and len(tgt) <= 1024:
                valid_pairs.append((inp, tgt))
        
        if not valid_pairs:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
            
        input_seqs, target_seqs = zip(*valid_pairs)
        
        input_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
        target_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0)
        
        max_len = min(input_padded.size(1), target_padded.size(1))
        input_padded = input_padded[:, :max_len]
        target_padded = target_padded[:, :max_len]
        
        return input_padded, target_padded

    # Better batch size and gradient accumulation
    batch_size = 8 if torch.cuda.is_available() else 1  # Reduced batch size
    gradient_accumulation_steps = 32  # Increased accumulation

    def worker_init_fn(worker_id):
        import random
        random.seed(42 + worker_id)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=6, 
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=6,
        pin_memory=True, 
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )

    logging.info(f"🔧 Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    logging.info(f"🔧 Effective batch size: {batch_size * gradient_accumulation_steps}")

    model = GPT2Mini(vocab_size=50_000)
    model.to(device)
    model = torch.compile(model)

# FIXED: Better optimizer configuration
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,  # FIXED: Reduced for better stability
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    # Re-enable mixed precision with better settings
    scaler = GradScaler(enabled=torch.cuda.is_available())
    if torch.cuda.is_available():
        logging.info("✅ Mixed precision ENABLED")
    else:
        logging.info("⚠️ Mixed precision DISABLED (CPU training)")

    # This would ideally come from a count of lines in 'pile_tokenized_train_clean.txt'
    # For now, we use the assumed 29_171_354
    num_training_sequences = 29_171_354

    batch_size = 8 # defined in the script
    gradient_accumulation_steps = 32 # defined in the script

    max_epochs = 2
    # Number of actual data batches processed per epoch
    batches_per_epoch = num_training_sequences // batch_size
    # Number of optimizer steps per epoch (after accumulation)
    gradient_steps_per_epoch = batches_per_epoch // gradient_accumulation_steps

    # Total optimizer steps over all epochs
    total_steps = gradient_steps_per_epoch * max_epochs

    # Warmup steps calculation remains sound relative to total_steps
    warmup_steps = min(5000, total_steps // 8)

    logging.info(f"Calculated batches per epoch: {batches_per_epoch}")
    logging.info(f"Calculated gradient steps per epoch: {gradient_steps_per_epoch}")
    logging.info(f"Calculated total steps: {total_steps}")
    logging.info(f"Calculated warmup steps: {warmup_steps}") 
    

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5
    )
    
    logging.info(f"📅 Training plan: {max_epochs} epochs, {gradient_steps_per_epoch} steps/epoch")
    logging.info(f"🔥 Warmup steps: {warmup_steps}, Total steps: {total_steps}")
    logging.info(f"🎯 Initial learning rate: {optimizer.param_groups[0]['lr']:.2e}")

    global_step = 0
    
    if checkpoint:
        model.load_state_dict(checkpoint.get("model", {}))
        optimizer.load_state_dict(checkpoint.get("optimizer", {}))
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        global_step = checkpoint.get("global_step", 0)  # MOVE THIS LINE HERE
        logging.info("✅ Checkpoint loaded successfully")
        
    tokenizer = ByteLevelBPETokenizer("my_tokenizer/vocab.json", "my_tokenizer/merges.txt")

    # Optional: Print top 30 tokens by ID (for debugging purposes)
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    print("Top 30 tokens by frequency (lowest IDs):")
    for token, idx in sorted_vocab[:30]:
        print(f"{idx:5d} ➜ {repr(token)}")

    pad_id = tokenizer.token_to_id("<pad>")

    if pad_id is None:
        logging.error("❌ <pad> token not found in tokenizer!")
        pad_id = 0
    
    metrics_file = os.path.join(CKPT_DIR, "metrics.json")


#################

    def save_checkpoint(epoch, global_step, current_loss=None, current_lr=None, is_best=False):
        # Create a copy of metrics and add current step info
        current_metrics = metrics.copy()
        if current_loss is not None and current_lr is not None:
            current_metrics["current_step_loss"] = current_loss
            current_metrics["current_step_lr"] = current_lr
            current_metrics["current_step"] = global_step
            current_metrics["current_epoch"] = epoch

        checkpoint_data = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "metrics": current_metrics,
            "best_val_loss": best_val_loss  # FIXED: Save best_val_loss
        }
        
        if global_step % SAVE_EVERY_STEPS == 0:
            path = os.path.join(CKPT_DIR, f"ckpt-step{global_step:06d}.pt")
        else:
            path = os.path.join(CKPT_DIR, f"ckpt-epoch{epoch:02d}.pt")
        torch.save(checkpoint_data, path)
        if is_best:
            best_path = os.path.join(CKPT_DIR, "best_model.pt")
            torch.save(checkpoint_data, best_path)
            logging.info(f"💾 Saved best model (epoch {epoch})")
        logging.info(f"💾 Checkpoint saved: {path}")

    def quick_val():
        model.eval()
        loss, steps = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                if steps >= 50: break
                if x.numel() == 0 or y.numel() == 0: continue
                x, y = x.to(device), y.to(device)
                with autocast(enabled=scaler.is_enabled()):
                    logits = model(x)
                    l = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=pad_id)
                if not torch.isnan(l):
                    loss += l.item()
                    steps += 1
        model.train()
        return loss / steps if steps > 0 else float('inf')

    best_val_loss = checkpoint.get("best_val_loss", float("inf")) if checkpoint else float("inf")
    patience = 3
    patience_counter = 0
    nan_count = 0
    max_nan_tolerance = 5

    logging.info("🚀 Starting training loop...")

##################

    # FIXED: Proper epoch range when resuming
    start_epoch = checkpoint.get("epoch", 0) + 1 if checkpoint else 0
    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.time()
        logging.info(f"🧪 EPOCH {epoch + 1}/{max_epochs} STARTED")
        
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_nan_count = 0
        optimizer.zero_grad()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")

        for batch_idx, batch in enumerate(train_pbar):
            if batch_idx >= batches_per_epoch:
                break
            try:
                input_ids, labels = batch
                if input_ids.numel() == 0 or labels.numel() == 0:
                    continue
                
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # FIXED: Proper mixed precision usage
                with autocast(enabled=scaler.is_enabled()):
                    logits = model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=pad_id,
                        reduction='mean',
                        label_smoothing=0.0
                    ) / gradient_accumulation_steps

                # FIXED: Better NaN detection
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 10.0:
                    nan_count += 1
                    epoch_nan_count += 1
                    logging.error(f"❌ Bad loss detected at step {global_step}: {loss.item()}")
                    
                    optimizer.zero_grad()
                    
                    if nan_count > max_nan_tolerance:
                        logging.error(f"🚨 Too many bad losses ({nan_count}), aborting!")
                        return None
                    continue

                # FIXED: Proper gradient scaling
                scaler.scale(loss).backward()

##################

                # FIXED: Better gradient accumulation check
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Check gradients before stepping
                    scaler.unscale_(optimizer)
                    
                    # FIXED: More aggressive gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.50)
                    
                    # Log gradient norm less frequently
                    if grad_norm > 2.0 and global_step % 10 == 0:
                        logging.warning(f"⚠️ Large gradient norm: {grad_norm:.4f} at step {global_step}")
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    
                    # MID-EPOCH VALIDATION
                    if global_step % 250 == 0:
                        val = quick_val()
                        is_new_best = val < best_val_loss
                        current_lr = scheduler.get_last_lr()[0]
                        if is_new_best:
                            best_val_loss = val
                            logging.info(f"🏆 NEW BEST MODEL at step {global_step}: {val:.4f}")
                        current_lr = scheduler.get_last_lr()[0]
                        save_checkpoint(epoch, global_step, val, current_lr, is_best=is_new_best)
                      
                    # Add step-based checkpoint saving
                    if global_step % SAVE_EVERY_STEPS == 0:
                        logging.info(f"💾 Saving checkpoint at step {global_step}")
                        current_avg_loss = sum(train_pbar.loss_window) / len(train_pbar.loss_window) if hasattr(train_pbar, 'loss_window') and train_pbar.loss_window else loss.item() * gradient_accumulation_steps
                        current_lr = scheduler.get_last_lr()[0]
                        save_checkpoint(epoch, global_step, current_avg_loss, current_lr, is_best=False)
                        

                # FIXED: Ensure loss is always a scalar
                epoch_loss += loss.item() * gradient_accumulation_steps
                epoch_steps += 1



                # Track running averages
                if not hasattr(train_pbar, 'loss_window'):
                    train_pbar.loss_window = []
                    train_pbar.update_freq = 100  # Average over 100 batches

                train_pbar.loss_window.append(loss.item() * gradient_accumulation_steps)
                if len(train_pbar.loss_window) > train_pbar.update_freq:
                    train_pbar.loss_window.pop(0)


##################

                # Update display every 10 batches with running average
                if batch_idx % 10 == 0:
                    avg_loss = sum(train_pbar.loss_window) / len(train_pbar.loss_window)
                    prev_val_loss = best_val_loss
                    current_lr = scheduler.get_last_lr()[0]
                    train_pbar.set_postfix({
                        'avg_loss': f'{avg_loss:.4f}',
                        'cur_loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                        'lr': f'{current_lr:.2e}', 
                        'step': global_step,
                        'prev_val': f'{prev_val_loss:.4f}',
                        'nan': epoch_nan_count
                    })

            except Exception as e:
                logging.error(f"❌ Training step failed at batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                optimizer.zero_grad()
                continue

        train_pbar.close()

        avg_train_loss = epoch_loss / epoch_steps if epoch_steps > 0 else float('inf')
        train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item() if avg_train_loss < 10 else float('inf')
        current_lr = scheduler.get_last_lr()[0]

        logging.info(f"📈 EPOCH {epoch + 1} TRAINING COMPLETE")
        logging.info(f"📊 Average Training Loss: {avg_train_loss:.4f}")
        logging.info(f"🧮 Training Perplexity: {train_perplexity:.2f}")
        logging.info(f"🎯 Learning Rate: {current_lr:.2e}")
        logging.info(f"🔢 Steps: {epoch_steps}, Global Step: {global_step}")

        # Quick validation (shortened)
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 200:  # Better validation sample size
                    break
                    
                try:
                    input_ids, labels = batch
                    if input_ids.numel() == 0 or labels.numel() == 0:
                        continue

                    input_ids = input_ids.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with autocast(enabled=scaler.is_enabled()):
                        logits = model(input_ids)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            ignore_index=pad_id,
                            reduction='mean'
                        )

                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss += loss.item()
                        val_steps += 1

                except Exception as e:
                    logging.error(f"❌ Validation step failed: {e}")
                    continue

        avg_val_loss = val_loss / val_steps if val_steps > 0 else float("inf")
       
        is_best = avg_val_loss < best_val_loss

        if is_best:
            logging.info(f"🏆 NEW BEST MODEL at epoch end: {best_val_loss:.4f} → {avg_val_loss:.4f}")
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        logging.info(f"🔍 Validation Loss: {avg_val_loss:.4f}")
        logging.info(f"🏆 Best Validation Loss: {best_val_loss:.4f}")
        if is_best:
            logging.info("🎉 NEW BEST MODEL!")
           
        # FIXED: Sample generation to check coherence
        if (epoch + 1) % 2 == 0:  # Every 2 epochs
            model.eval()
            try:
                test_prompt = "The quick brown fox"
                prompt_ids = tokenizer.encode(test_prompt).ids
                input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        
                with torch.no_grad():
                    for _ in range(20):  # Generate 20 tokens
                        if input_ids.size(1) >= 100:
                            break
                        logits = model(input_ids)
                        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        input_ids = torch.cat([input_ids, next_token], dim=1)
        
                generated = tokenizer.decode(input_ids[0].tolist())
                logging.info(f"🎯 Generation sample: {generated}")
            except Exception as e:
                logging.warning(f"⚠️ Generation test failed: {e}")
        model.train()
        
        # Save metrics
        metrics["train_losses"].append(avg_train_loss)
        metrics["val_losses"].append(avg_val_loss)
        metrics["learning_rates"].append(current_lr)
        
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
            
        current_lr = scheduler.get_last_lr()[0]
        save_checkpoint(epoch, global_step, avg_val_loss, current_lr, is_best=is_best)

        # Time tracking
        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)
        total_elapsed = time.time() - training_start_time
        
        logging.info(f"⏱️ Epoch duration: {format_time(epoch_duration)}")
        logging.info(f"⏱️ Total elapsed: {format_time(total_elapsed)}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info(f"✅ EPOCH {epoch + 1} COMPLETE")
        logging.info("="*50)

        # Early stopping
        if patience_counter >= patience:
            logging.info(f"🛑 Early stopping after {patience} epochs without improvement")
            break

    total_training_time = time.time() - training_start_time
    logging.info("🎉 TRAINING COMPLETE!")
    logging.info(f"⏱️ Total training time: {format_time(total_training_time)}")
    logging.info(f"📊 Total NaN encounters: {nan_count}")
    
    return model

# ============================================================================
# SECTION 7: TEXT GENERATION TESTING
# ============================================================================
def test_generation():
    """Test generation with proper error handling"""
    logging.info("🧪 Starting generation tests...")
    
    model = GPT2Mini(vocab_size=50_000)
    best_model_path = "./checkpoints/best_model.pt"
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)
        logging.info("✅ Loaded best trained model")
    else:
        logging.warning("⚠️ No trained model found, using random weights")
    
    model.to(device)
    model.eval()
    
    tokenizer = ByteLevelBPETokenizer("my_tokenizer/vocab.json", "my_tokenizer/merges.txt")
    
    test_prompts = [
        "The quick brown fox",
        "Nigel thinks",
        "Metallica are",
        "Do open-source LLMs rock?",
        "What happened In the beginning?",
        "Once upon a time"
    ]
    
    results = []
    
    for prompt in test_prompts:
        logging.info(f"🎯 Testing prompt: '{prompt}'")
        
        try:
            prompt_ids = tokenizer.encode(prompt).ids
            input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
            
            with torch.no_grad():
                for i in range(30):
                    if input_ids.size(1) >= 1024:
                        break
                    
                    logits = model(input_ids)
                    probs = F.softmax(logits[:, -1, :] / 0.07, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
            
            generated_text = tokenizer.decode(input_ids[0].tolist())
            generated_new = tokenizer.decode(input_ids[0][len(prompt_ids):].tolist())
            
            logging.info(f"📝 Generated: {generated_text}")
            logging.info(f"✨ New part: {generated_new}")
            
            results.append({
                "prompt": prompt,
                "generated_full": generated_text,
                "generated_new": generated_new
            })
            
        except Exception as e:
            logging.error(f"❌ Generation failed for '{prompt}': {e}")
            results.append({
                "prompt": prompt,
                "error": str(e)
            })
    
    with open("generated_outputs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info("✅ Generation test complete")

# ============================================================================
# SECTION 8: TRAINING VISUALIZATION
# ============================================================================
def plot_training_curves():
    """Plot training curves with proper error handling"""
    metrics_path = "./checkpoints/metrics.json"
    
    if not os.path.exists(metrics_path):
        logging.warning("⚠️ No metrics file found, skipping plots")
        return
    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = range(1, len(metrics["train_losses"]) + 1)
        ax1.plot(epochs, metrics["train_losses"], label="Training Loss", marker='o')
        
        if "val_losses" in metrics and len(metrics["val_losses"]) > 0:
            val_epochs = range(1, len(metrics["val_losses"]) + 1)
            ax1.plot(val_epochs, metrics["val_losses"], label="Validation Loss", marker='s')
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Progress")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if "learning_rates" in metrics and len(metrics["learning_rates"]) > 0:
            ax2.plot(epochs, metrics["learning_rates"], label="Learning Rate", color='red', marker='x')
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Learning Rate")
            ax2.set_title("Learning Rate Schedule")
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = "./checkpoints/training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"📈 Training curves saved to {plot_path}")
        
    except Exception as e:
        logging.error(f"❌ Plotting failed: {e}")

# ============================================================================
# SECTION 9: MAIN EXECUTION PIPELINE
# ============================================================================
def main():
    """Main execution function"""
    logging.info("🚀 GPT-2 Training Pipeline Started")
    logging.info("="*60)
    
    try:
        # Step 1: Setup data and tokenizer
        logging.info("📋 Step 1: Setting up data and tokenizer...")
        setup_data_and_tokenizer()
        if not os.path.exists("pile_tokenized_train_clean.txt"):
            filter_tokenized_data()        
        
        # Step 2: Check if we need to train
        best_model_path = "./checkpoints/best_model.pt"
        
        if not os.path.exists(best_model_path):
            logging.info("📋 Step 2: No trained model found, starting training...")
            train_model()
        else:
            logging.info("📋 Step 2: Found existing trained model, skipping training")
        
        # Step 3: Test generation
        logging.info("📋 Step 3: Testing generation...")
        test_generation()
        
        # Step 4: Plot training curves
        logging.info("📋 Step 4: Creating training plots...")
        plot_training_curves()
        
        logging.info("🎉 PIPELINE COMPLETE!")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")
        raise

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================
if __name__ == "__main__":
    main()