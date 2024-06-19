import json
import pandas as pd
from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import tqdm

model_id="LLaMa2_13B_Chat-finetuned-exp"
# tokenizer.pad_token = tokenizer.eos_token
tokenizer = AutoTokenizer.from_pretrained(model_id)

test= pd.read_csv("test.csv")

def create_test_prompt(row):
    # Customize the prompt format based on your requirements
    prompt = f"Instruction: {row['instruction']}\nContext: {row['prompt']}\nResponse: "
    return prompt

# Apply the function to each row of the DataFrame
test['text'] = test.apply(create_test_prompt, axis=1)
val_data_df = test