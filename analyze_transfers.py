"""
Analyze CPU↔GPU transfer times in SmolVLM ONNX backend.
"""
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path('.').absolute() / 'src'))

from models.SmolVLM.onnx_backend import OnnxBackend
from utils.config import VLMChatConfig
from PIL import Image

# Load config and create backend
config = VLMChatConfig(json.load(open('config.json')))
backend = OnnxBackend(config)

# Load test image
image = Image.open('captures/camera0_20251014_180719.jpg')

# Prepare inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do you see?"}
        ]
    }
]

print("Preparing inputs...")
start = time.time()
inputs = backend.prepare_inputs(messages, [image])
print(f"Input preparation: {time.time() - start:.3f}s\n")

print("Analyzing transfer sizes:")
print(f"  pixel_values: {inputs['pixel_values'].shape} = {inputs['pixel_values'].nbytes / 1024 / 1024:.2f} MB")
print(f"  pixel_attention_mask: {inputs['pixel_attention_mask'].shape} = {inputs['pixel_attention_mask'].nbytes / 1024:.2f} KB")
print(f"  input_ids: {inputs['input_ids'].shape} = {inputs['input_ids'].nbytes / 1024:.2f} KB")
print(f"  attention_mask: {inputs['attention_mask'].shape} = {inputs['attention_mask'].nbytes / 1024:.2f} KB\n")

# Measure vision encoder
print("Testing vision encoder (with TensorRT)...")
runs = 3
times = []
for i in range(runs):
    start = time.time()
    image_features = backend._vision_session.run(
        ['image_features'],
        {
            'pixel_values': inputs['pixel_values'],
            'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
        }
    )[0]
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.3f}s")

print(f"  Average: {np.mean(times):.3f}s")
print(f"  Image features shape: {image_features.shape} = {image_features.nbytes / 1024:.2f} KB\n")

# Measure embedding
print("Testing embedding (with TensorRT)...")
input_ids_single = inputs['input_ids'][:, :1]  # Single token
runs = 100
times = []
for i in range(runs):
    start = time.time()
    inputs_embeds = backend._embed_session.run(None, {'input_ids': input_ids_single})[0]
    elapsed = time.time() - start
    times.append(elapsed)

print(f"  Average (100 runs): {np.mean(times)*1000:.3f}ms")
print(f"  Min: {np.min(times)*1000:.3f}ms, Max: {np.max(times)*1000:.3f}ms")
print(f"  Embedding shape: {inputs_embeds.shape} = {inputs_embeds.nbytes / 1024:.2f} KB\n")

# Estimate transfer overhead
print("Transfer overhead estimates:")
pixel_transfer_time = (inputs['pixel_values'].nbytes / 1024 / 1024) / 12.0  # ~12 GB/s PCIe bandwidth
print(f"  pixel_values transfer (~12 GB/s PCIe): {pixel_transfer_time*1000:.3f}ms")

features_transfer_time = (image_features.nbytes / 1024) / (12 * 1024)  # Much smaller
print(f"  image_features transfer: {features_transfer_time*1000:.3f}ms")

embed_transfer_time = (inputs_embeds.nbytes / 1024) / (12 * 1024)
print(f"  embedding transfer: {embed_transfer_time*1000:.3f}ms")

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)
print(f"Vision encoder time: ~{np.mean(times):.3f}s")
print(f"  - Includes: CPU→GPU pixel transfer (~{pixel_transfer_time*1000:.1f}ms)")
print(f"  - Includes: GPU computation")  
print(f"  - Includes: GPU→CPU features transfer (~{features_transfer_time*1000:.2f}ms)")
print(f"\nEmbedding per token: ~{np.mean(times)*1000:.1f}ms")
print(f"  - Most time is likely TensorRT engine selection overhead")
print(f"  - Actual compute + transfer probably < 1ms")
print("\nBottleneck: Vision encoder (first token)")
print("Optimization: Cache image_features in backend for follow-up questions")
