import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

prompt_version = "v2"
prompt_lan = "english"
prompts = json.load(open(f"prompts/{prompt_lan}.json", "r", encoding="utf-8"))[prompt_version]

models = {
    "gpt": "models/gpt2",
    "llama": "models/llama-2-7b-hf",
    "deepseek": "models/deepseek-aiDeepSeek-R1-Distill-Qwen-1.5B" 
}

results = []

for label, repo in models.items():
    print(f"\n\n===== Start: {label.upper()} | {repo} =====")
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    model_device = next(model.parameters()).device


    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )
        end_time = time.time()
        decoded_outputs = tokenizer.decode(outputs[0]).strip().replace(prompt, '')
        duration = end_time - start_time

        print(f"\nPrompt: {prompt}")
        print(f"Time: {duration:.2f}s")
        print(f"Output: {decoded_outputs[:30]}")

        results.append({
            "model": label,
            "prompt": prompt.replace('\n',' '),
            "latency": duration,
            "output": decoded_outputs.replace('\n',' ')[:200]
        })

    print(f"\n===== Complete: {label.upper()} =====")


print(f"\n===== Finsihed all models! =====")


print("\n\n============ 🔍 Benchmark Results ============\n")
print("| Model | Prompt | Latency(s) | Output Snippet |")
print("|-------|--------|------------|----------------|")
for row in results:
    print(f"| {row['model']} | {row['prompt']} | {row['latency']} | {row['output']} |")
