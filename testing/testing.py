import json
import pandas as pd
from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import tqdm
import time
import re

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

# Load the model and the tokenizer. Set generation config
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto")


generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.1,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id
)


st_time = time.time()
example = val_data_df['text'][3]
inputs = tokenizer(example, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, generation_config=generation_config)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
# print(re.search(r"###Response: (\d+)", answer).group(1))
print(time.time()-st_time)


def solve_question(question_prompt):
    inputs = tokenizer(question_prompt, return_tensors="pt", padding=True, truncation=True,max_length= 2048).to("cuda")
    outputs = model.generate(**inputs, generation_config=generation_config)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return answer

all_answers = []
val_data_prompts = list(val_data_df['text'])

for i in tqdm.tqdm(range(0, len(val_data_prompts), 4)):
    question_prompts = val_data_prompts[i:i+4]
    ans = solve_question(question_prompts)
    all_answers.extend(ans)
    torch.cuda.empty_cache()

val_data_df['Finetuned_results'] = all_answers

val_data_df.to_csv('test.csv', index=False)






