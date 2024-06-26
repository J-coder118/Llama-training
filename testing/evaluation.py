import csv
import matplotlib.pyplot as plt
from transformers import pipeline
from tqdm import tqdm

# Load the fine-tuned model
generator = pipeline('text-generation', model='J-coder118/llama-2-7b-finetuned')

# Define a function to evaluate the model
def evaluate_model(instruction, prompt):
    # Generate the response
    response = generator(instruction + prompt, max_length=1024, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, num_beams=4)[0]['generated_text']
    return response

# Load the testing data from the CSV file
test_data = []
with open('ai_logs.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        test_data.append((row['instruction'], row['prompt'], row['response']))

# Evaluate the model on the testing data and compare with ground truth
matches = 0
total = 0
match_percentages = []

for instruction, prompt, ground_truth in tqdm(test_data[:1000], desc="Evaluating model"):
    model_response = evaluate_model(instruction, prompt)
    print(f"Model response: {model_response}")
    if model_response == ground_truth:
        matches += 1
    total += 1
    match_percentage = matches / total * 100
    match_percentages.append(match_percentage)

# Plot the match percentage over the 1000 rows
plt.figure(figsize=(12, 6))
plt.plot(match_percentages)
plt.title("Model Response Match Percentage")
plt.xlabel("Row")
plt.ylabel("Match Percentage (%)")
plt.grid()
plt.show()