#!/usr/bin/env python3
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 🚀 Activate Virtual Environment
venv_python = "/home/ubuntu/eden_env/bin/python3"
if os.path.exists(venv_python):
    os.environ["VIRTUAL_ENV"] = "/home/ubuntu/eden_env"
    os.environ["PATH"] = f"/home/ubuntu/eden_env/bin:{os.environ['PATH']}"

# 🚀 Load GPT-Neo 1.3B
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("\n🚀 GPT-Neo 1.3B Loaded Successfully!\n")

# 🚀 Initialize Variables
entropy_value = np.random.uniform(5.0, 10.0)
prev_entropy = entropy_value
entropy_stagnation_counter = 0
shockwave_energy = 1.0
alpha, beta, gamma, delta = [0.25] * 4

# 🚀 Exponential Moving Average (EMA)
alpha_smoothing = max(0.05, min(0.3, abs(entropy_value - prev_entropy)))
smoothed_entropy = alpha_smoothing * entropy_value + (1 - alpha_smoothing) * prev_entropy
prev_entropy = entropy_value

# 🚀 Logarithmic Adaptive Entropy Scaling
entropy_scaling_factor = max(0.4, min(2.2, 1 / (1 + np.log1p(entropy_value) * 0.015)))

# 🚀 Monitor Entropy Change
entropy_diff = abs(entropy_value - prev_entropy)
entropy_stagnation_counter = entropy_stagnation_counter + 1 if entropy_diff < 0.0001 else 0

# 🚀 Adaptive Instability Factor
if entropy_stagnation_counter > 2 and abs(entropy_diff) < 0.001:
    entropy_value += np.random.uniform(-0.03, 0.03)

# 🚀 Sigmoid-Based Shockwave Perturbation
if entropy_stagnation_counter > 5:
    shockwave_energy = min(7.0, shockwave_energy * 1.1)
    entropy_shift = 2 * (1 / (1 + np.exp(-shockwave_energy))) - 1
    entropy_value += np.random.uniform(-entropy_shift, entropy_shift)

# 🚀 Fully Adaptive Entropy Floor
min_entropy_threshold = smoothed_entropy * (0.75 + entropy_value / (50.0 + entropy_stagnation_counter))

# 🚀 Normalize Weights
weight_sum = alpha + beta + gamma + delta
oscillation = np.random.uniform(0.995, 1.005)
if weight_sum > 0:
    alpha, beta, gamma, delta = [(x / weight_sum) * oscillation for x in [alpha, beta, gamma, delta]]
    total_weight = alpha + beta + gamma + delta
    alpha, beta, gamma, delta = [x / total_weight for x in [alpha, beta, gamma, delta]]

# 🚀 Print Optimization Metrics
print("\n📊 EDEN-Neo Optimization Metrics:")
print(f"Smoothed Entropy Score: {entropy_value:.4f}")
print(f"Entropy Change: {entropy_diff:.4f}, Stagnation Counter: {entropy_stagnation_counter}")
print(f"Shockwave Energy: {shockwave_energy:.4f}")
print(f"Momentum-Optimized Weights -> Alpha: {alpha:.4f}, Beta: {beta:.4f}, Gamma: {gamma:.4f}, Delta: {delta:.4f}")
print("═════════════════════════════════════════════\n")

# 🚀 Generate Text Output
tokenizer.pad_token = tokenizer.eos_token
input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors="pt", padding=True, truncation=True)
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    min_length=50,
    max_length=250,
    return_dict_in_generate=True,
    output_scores=True
)

# 🚀 Ensure Proper Text Generation
if hasattr(output, "scores") and len(output.scores) > 0:
    logits = output.scores[-1] * entropy_scaling_factor
    probabilities = torch.nn.functional.softmax(torch.tanh(logits), dim=-1)
    new_output = torch.multinomial(probabilities, num_samples=1)
    decoded_output = tokenizer.decode(new_output[0], skip_special_tokens=True)
else:
    decoded_output = "[No response generated due to model error]"

print(f"\n📝 Generated Output: {decoded_output}")
print(f"Final Smoothed Entropy Score: {entropy_value:.4f}")
print("\n═════════════════════════════════════════════\n")
