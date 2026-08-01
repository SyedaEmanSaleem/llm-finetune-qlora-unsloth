"""
================================================================================
LLM Fine-Tuning with QLoRA + Unsloth  –  Complete Colab / Jupyter Notebook
================================================================================
Copy each section into successive cells of a Google Colab notebook.
Runtime → Change runtime type → GPU → T4 (free tier is enough for 3B–8B models).

Estimated time on sample data: 10–20 minutes end-to-end.
"""

# %% [markdown]
# # LLM Fine-Tuning with QLoRA and Unsloth
#
# **Goal**: Fine-tune a small LLM on a domain-specific instruction dataset so it
# answers questions in that domain more accurately than the base model.
#
# **Stack**: Unsloth + QLoRA (4-bit) + TRL SFTTrainer + Hugging Face.
#
# This notebook is self-contained. Run the cells in order.

# %% [markdown]
# ## 0. Check GPU

# %%
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")

# %% [markdown]
# ## 1. Install dependencies (Colab)

# %%
# Run this cell once. On local machines with proper CUDA you can use requirements.txt instead.
import os
if "COLAB_" in "".join(os.environ.keys()):
    # Colab-friendly install
    !pip install -q unsloth
    !pip install -q --no-deps bitsandbytes accelerate xformers peft trl datasets transformers
else:
    print("Not on Colab – assume packages are already installed via requirements.txt")

# %% [markdown]
# ## 2. Load base model in 4-bit (QLoRA)

# %%
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048          # Choose any; Unsloth handles RoPE scaling
dtype = None                   # Auto detection
load_in_4bit = True            # QLoRA

# Good starting models for free Colab T4:
#   "unsloth/Llama-3.2-3B-Instruct"          ← fastest / lowest VRAM
#   "unsloth/Meta-Llama-3.1-8B-Instruct"
#   "unsloth/Qwen2.5-7B-Instruct"
#   "unsloth/Mistral-7B-Instruct-v0.3"

model_name = "unsloth/Llama-3.2-3B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print("Model loaded successfully.")

# %% [markdown]
# ## 3. Add LoRA adapters
#
# We train only a tiny fraction of the parameters (~1%). This is what makes
# fine-tuning feasible on consumer GPUs.

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                          # Rank – higher = more capacity, more VRAM
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,                # 0 is optimized in Unsloth
    bias = "none",
    use_gradient_checkpointing = "unsloth",   # 30% less VRAM, longer context
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# Print trainable parameters
trainable, total = 0, 0
for _, p in model.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()
print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# %% [markdown]
# ## 4. Prepare dataset
#
# We use a small medical-instruction sample that ships with the project.
# You can replace it with any Alpaca-style JSONL or a Hugging Face dataset.

# %%
from datasets import Dataset, load_dataset
import json
from pathlib import Path

# ---------- Option A: local JSONL (recommended for the project) ----------
sample_path = Path("data/sample_instructions.jsonl")
if not sample_path.exists():
    # Fallback: create a tiny in-memory set so the notebook still runs
    print("Sample file not found – using built-in mini dataset.")
    raw = [
        {"instruction": "What is hypertension?", "input": "", "output": "Hypertension is persistently high blood pressure..."},
        {"instruction": "Difference between Type 1 and Type 2 diabetes?", "input": "", "output": "Type 1 is autoimmune insulin deficiency; Type 2 is insulin resistance..."},
    ]
else:
    raw = []
    with open(sample_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw.append(json.loads(line))

dataset = Dataset.from_list(raw)

# ---------- Option B: Hugging Face dataset (uncomment to use) ----------
# dataset = load_dataset("yahma/alpaca-cleaned", split="train[:500]")

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples.get("input", [""] * len(instructions))
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = ALPACA_PROMPT.format(
            instruction=instruction,
            input=input_text or "",
            output=output,
        )
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
print(f"Dataset size: {len(dataset)}")
print("Example text:\n", dataset[0]["text"][:500], "...")

# %% [markdown]
# ## 5. Train with SFTTrainer

# %%
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False,                 # True can speed up short sequences
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,   # effective batch size = 8
        warmup_steps = 5,
        max_steps = 60,                    # quick experiment; use num_train_epochs=1 for full run
        # num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

print("Starting training...")
trainer_stats = trainer.train()
print(trainer_stats)

# %% [markdown]
# ## 6. Inference – chat with your fine-tuned model

# %%
FastLanguageModel.for_inference(model)   # 2× faster native inference

def ask(instruction: str, input_text: str = "", max_new_tokens: int = 256):
    prompt = ALPACA_PROMPT.format(
        instruction=instruction,
        input=input_text,
        output="",               # generation starts after this
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
    )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return only the response part
    if "### Response:" in full:
        return full.split("### Response:")[-1].strip()
    return full

# Try a few questions
print("Q: What is hypertension and what are its common risk factors?")
print("A:", ask("What is hypertension and what are its common risk factors?"))
print()
print("Q: Explain the difference between Type 1 and Type 2 diabetes.")
print("A:", ask("Explain the difference between Type 1 and Type 2 diabetes."))

# %% [markdown]
# ## 7. Save the LoRA adapter

# %%
output_dir = "outputs/lora_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Adapter saved to {output_dir}")

# Optional: also save a merged 16-bit model (needs more disk)
# model.save_pretrained_merged("outputs/merged_16bit", tokenizer, save_method="merged_16bit")

# Optional: push to Hugging Face Hub
# model.push_to_hub_merged("your-username/my-medical-llama", tokenizer, save_method="lora", token="hf_...")

# %% [markdown]
# ## 8. (Optional) Load the adapter later

# %%
# from unsloth import FastLanguageModel
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "outputs/lora_model",
#     max_seq_length = 2048,
#     load_in_4bit = True,
# )
# FastLanguageModel.for_inference(model)

# %% [markdown]
# ## Next steps for students
#
# 1. Replace the sample dataset with your own domain data (see `data/README_data.md`).
# 2. Increase `max_steps` or switch to `num_train_epochs=1–3`.
# 3. Experiment with LoRA rank (8, 16, 32, 64) and learning rate.
# 4. Evaluate on a held-out set of questions (qualitative + optional LLM-as-judge).
# 5. Merge the adapter and convert to GGUF for Ollama / llama.cpp.
# 6. Read `docs/THEORY.md` for the conceptual background.
#
# Congratulations – you have completed a full QLoRA fine-tuning pipeline! 🦥
