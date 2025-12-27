"""
Run OpenCLIP image+text inference using the exported ONNX image/text encoders.

Prereqs:
- onnxruntime
- open_clip (for tokenizer + transforms)
- numpy, PIL

This script:
- builds ONNX Runtime sessions (preferring TensorRT/CUDA if available),
- runs the image and text encoders,
- L2-normalizes features,
- computes scaled dot-product logits and softmax probabilities.
"""
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import open_clip

# --- Configuration: adjust paths and model name ---
IMAGE_ONNX = "openclip_image_encoder.onnx"
TEXT_ONNX = "openclip_text_encoder.onnx"
MODEL_NAME = "MobileCLIP2-S0"
# If you used the same model_kwargs as when exporting
model_kwargs = {}
if not (MODEL_NAME.endswith("S3") or MODEL_NAME.endswith("S4") or MODEL_NAME.endswith("L-14")):
    model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

# --- Helper: choose providers intelligently ---
available = ort.get_available_providers()
preferred = [p for p in (
    #"TensorrtExecutionProvider", 
    "CUDAExecutionProvider", 
    #"CPUExecutionProvider"
    ) if p in available]
if not preferred:
    raise RuntimeError(f"No ONNX Runtime providers available. Found: {available}")

# Optionally set provider options (example for CUDA)
provider_options = None
if "CUDAExecutionProvider" in preferred:
    # provider_options should be a list of dicts corresponding to providers when passed,
    # but many builds are fine without provider options. Uncomment and adapt if needed.
    # provider_options = [{"device_id": 0} if p == "CUDAExecutionProvider" else {} for p in preferred]
    provider_options = None

print("Available ORT providers:", available)
print("Using provider order:", preferred)

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

def make_session(path: str):
    # Create a session using preferred providers. Falls back automatically when provider fails.
    if provider_options is not None:
        return ort.InferenceSession(path, sess_options, providers=preferred, provider_options=provider_options)
    return ort.InferenceSession(path, sess_options, providers=preferred)

# --- Create ONNX Runtime sessions ---
print("Loading ONNX sessions...")
sess_img = make_session(IMAGE_ONNX)
sess_txt = make_session(TEXT_ONNX)
print("Image session providers (active):", sess_img.get_providers())
print("Text session providers (active):", sess_txt.get_providers())

# --- Build transforms/tokenizer (no model weights needed) ---
# create_model_and_transforms(..., pretrained=False) returns transforms without loading heavy pretrained weights
_, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=False, **model_kwargs)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# --- Input examples ---
image_path = "cat.jpeg"  # adjust path
texts = ["a diagram", "a dog", "a cat"]

# Preprocess image -> numpy float32 (shape: [B, C, H, W])
img_pil = Image.open(image_path).convert("RGB")
img_torch = preprocess(img_pil).unsqueeze(0)  # torch tensor, shape [1,3,H,W]
img_input = img_torch.cpu().numpy().astype(np.float32)

# Tokenize text -> numpy int64 (shape: [B, seq_len])
text_tokens = tokenizer(texts)  # returns a torch.LongTensor or similar
text_input = text_tokens.cpu().numpy().astype(np.int64)

# --- Run encoders ---
print("Running image encoder ONNX...")
ort_img_out = sess_img.run(None, {sess_img.get_inputs()[0].name: img_input})[0]  # shape [B, D]
print("Running text encoder ONNX...")
ort_txt_out = sess_txt.run(None, {sess_txt.get_inputs()[0].name: text_input})[0]  # shape [B_text, D]

# --- Normalize embeddings ---
def l2_normalize(x, axis=-1, eps=1e-12):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)

img_feats = l2_normalize(ort_img_out, axis=-1)
txt_feats = l2_normalize(ort_txt_out, axis=-1)

# --- Similarity and probabilities ---
scale = 100.0  # same scaling CLIP uses; adjust if your exported model expects different
logits = scale * (img_feats @ txt_feats.T)  # shape [B_image, B_text]
# softmax along text axis
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

probs = softmax(logits, axis=-1)

print("Logits shape:", logits.shape)
print("Probs:\n", probs)

# --- Interpret results ---
for i, p in enumerate(probs):
    sorted_idx = np.argsort(-p)
    print(f"\nImage {i} top predictions:")
    for rank, idx in enumerate(sorted_idx):
        print(f"  rank {rank+1}: '{texts[idx]}'  prob={p[idx]:.4f}")
