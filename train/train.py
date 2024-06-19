from huggingface_hub import login
login()

import torch
from datasets import load_dataset, Dataset
import peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments, BitsAndBytesConfig, GenerationConfig
from trl import SFTTrainer
import os, wandb, platform, gradio, warnings
import pandas as pd
import json
import tqdm

dataset = []
# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('ai_logs_for_website_scraper_6-16-24.csv', header=None)
print(df.head())


def print_system_specs():
    # Check if CUDA is available
    is_cuda_available = torch.cuda.is_available()
    print("CUDA Available:", is_cuda_available)
# Get the number of available CUDA devices
    num_cuda_devices = torch.cuda.device_count()
    print("Number of CUDA devices:", num_cuda_devices)
    if is_cuda_available:
        for i in range(num_cuda_devices):
            # Get CUDA device properties
            device = torch.device('cuda', i)
            print(f"--- CUDA Device {i} ---")
            print("Name:", torch.cuda.get_device_name(i))
            print("Compute Capability:", torch.cuda.get_device_capability(i))
            print("Total Memory:", torch.cuda.get_device_properties(i).total_memory, "bytes")
    # Get CPU information
    print("--- CPU Information ---")
    print("Processor:", platform.processor())
    print("System:", platform.system(), platform.release())
    print("Python Version:", platform.python_version())
print_system_specs()


# data split
def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

# come from book
train_df, validation_df, test_df = random_split(df, 0.7, 0.1)

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

def create_prompt(row):
    # video check
    prompt = f"Instruction: {row['instruction']}\nContext: {row['prompt']}\nResponse: {row['response']}"
    return prompt

train_df['text'] = train_df.apply(create_prompt, axis=1)
data_df = train_df

data = Dataset.from_pandas(data_df)

data = Dataset.from_pandas(data_df)

model_id="llama-2-13b-chat-hf-weights-local-path"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


# Configure quantization for memory efficiency
quantization_config_loading = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Load LLAMA2 13B model with quantization configurations
model = AutoModelForCausalLM.from_pretrained(
                                model_id,
                                quantization_config=quantization_config_loading,
                                device_map="auto"
                            )

# Modify model configuration parameters
model.config.use_cache=False
model.config.pretraining_tp=1
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Configure PEFT (Parameter Efficient Fine-Tuning)
peft_config = LoraConfig(
                    r=16,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
                )

# Apply PEFT configurations to the model
model = get_peft_model(model, peft_config)

# Define training arguments
training_arguments = TrainingArguments(
                            output_dir="LLaMa2_13B_Chat-finetuned-exp",
                            per_device_train_batch_size=8,
                            gradient_accumulation_steps=1,
                            optim="paged_adamw_32bit",
                            learning_rate=2e-4,
                            lr_scheduler_type="cosine",
                            save_strategy="epoch",
                            logging_steps=50,
                            num_train_epochs=1,
                            max_steps=500,
                            fp16=True,
                            push_to_hub=True
                        )

# Initialize SFTTrainer for training
trainer = SFTTrainer(
            model=model,
            train_dataset=data,
            peft_config=peft_config,
            dataset_text_field="text",
            args=training_arguments,
            tokenizer=tokenizer,
            packing=False,
            max_seq_length=512
    )

# Train the model
trainer.train()

# Push the trained model to Hugging Face Hub
trainer.push_to_hub()