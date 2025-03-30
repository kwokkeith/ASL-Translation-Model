
import pandas as pd
import subprocess
import os
from bert_score import score
import re
import time

cpu_count = os.cpu_count()

# Define model configurations (update paths!)
MODEL_CONFIGS = {
    "LLaMA3": {
        "model_path": "./model/ggml-model-Q4_K_M.gguf"
    },
    "Gemma": {
        "model_path": "./model/gemma-1.1-7b-it.Q4_K_M.gguf"
    },
}

# Load CSV
df = pd.read_csv("llm_data.csv")

# Function to normalize text
def normalize(text):
    return re.sub(r'[^a-z0-9 ]', '', text.lower().strip())

# Function to run llama-cli
def run_llm(prompt, model_path):
    print(f"Running llm with prompt: {prompt}")
    result = subprocess.run([
        "./llama-cli/llama-cli",
        "-m", model_path,
        "--jinja",
        "--single-turn",
        "-sys", "You are an interpreter trying to convert ASL phrases and fingerspelled names into English sentences. You must strictly follow the grammar provided",
        "-p", prompt,
        "--grammar-file", "./grammar/english.gbnf", 
        "-t", str(cpu_count)
    ], capture_output=True, text=True)

    s = result.stdout
    
    # Find the index of '[end of text]'
    end_marker = "[end of text]"
    end_index = s.find(end_marker)
    
    if end_index != -1:
        # Get the substring before '[end of text]'
        before_end = s[:end_index]
    
        # Find the last newline before '[end of text]'
        last_newline_index = before_end.rfind('\n')
    
        # Extract the line just before '[end of text]'
        line_before_end = before_end[last_newline_index + 1:].strip()
        print(line_before_end)
    
        # return the sentence
        return line_before_end
    else:
        print("[end of text] not found.")
        return ""
    
normalised_accuracies = {}
avg_runtimes = {}

# 3 runs per gesture per model
for model_name, config in MODEL_CONFIGS.items():
    outputs = {f"{model_name}_output_{i+1}": [] for i in range(3)}
    precision_scores = []
    recall_scores = []
    f1_scores = []
    runtimes = []
    correct = 0
    total = len(df)

    for prompt, ref in zip(df['gesture_sequence'], df['ground_truth']):
        model_outputs = []
        for i in range(3):
            start = time.time()
            out = run_llm(prompt, config["model_path"])
            end = time.time()
            runtime = end - start
            runtimes.append(runtime)

            outputs[f"{model_name}_output_{i+1}"].append(out)
            model_outputs.append(out)

        # Compute BERTScore for all 3 outputs
        P, R, F1 = score(model_outputs, [ref] * 3, lang="en", verbose=False)
        precision_scores.append(float(sum(P).item()) / 3)
        recall_scores.append(float(sum(R).item()) / 3)
        f1_scores.append(float(sum(F1).item()) / 3)

        norm_gt = normalize(ref)
        if norm_gt in [normalize(o) for o in model_outputs]:
            correct += 1

    # Add outputs and scores to dataframe
    for key in outputs:
        df[key] = outputs[key]
    df[f"{model_name}_bertscore_precision"] = precision_scores
    df[f"{model_name}_bertscore_recall"] = recall_scores
    df[f"{model_name}_bertscore_f1"] = f1_scores

    normalised_accuracies[model_name] = round((correct / total)*100, 2)
    avg_runtimes[model_name] = round(sum(runtimes) / len(runtimes), 2)

# Save full output
print("Saving output...")
df.to_csv("asl_llm_model_comparison_3runs_new.csv", index=False)

# Print summary
for model in MODEL_CONFIGS:
    avg_precision = df[model + "_bertscore_precision"].mean()
    avg_recall = df[model + "_bertscore_recall"].mean()
    avg_f1 = df[model + "_bertscore_f1"].mean()
    print(f"{model} - Avg BERTScore Precision: {avg_precision:.4f}")
    print(f"{model} - Avg BERTScore Recall: {avg_recall:.4f}")
    print(f"{model} - Avg BERTScore F1: {avg_f1:.4f}")
    print(f"{model} - Normalised Accuracy: {normalised_accuracies[model]}%")
    print(f"{model} - Avg Runtime: {avg_runtimes[model]}s")
