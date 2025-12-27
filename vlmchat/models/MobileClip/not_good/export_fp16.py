"""
Export OpenCLIP image + text encoders to FP16 ONNX and verify with ONNX Runtime.

Usage:
  python export_fp16_onnx.py

Adjust paths and MODEL_NAME as needed.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import open_clip

# CONFIG
MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED_PATH = "/home/patrick/.cache/huggingface/hub/models--apple--MobileCLIP2-S0/snapshots/3136ea51c8ed56b9f9abfab04cb816735aaad6cb/mobileclip2_s0.pt"
IMAGE_ONNX_FP16 = "openclip_image_encoder_fp16.onnx"
TEXT_ONNX_FP16 = "openclip_text_encoder_fp16.onnx"
OPSET = 14
SIMPLEFY_IF_AVAILABLE = True   # requires onnxsim

# Choose device for tracing/export. Prefer CUDA when available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Export device:", device)

# Load model and transforms (pretrained path is the local checkpoint file)
model_kwargs = {}
if not (MODEL_NAME.endswith("S3") or MODEL_NAME.endswith("S4") or MODEL_NAME.endswith("L-14")):
    model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_PATH, **model_kwargs)

# If your model requires reparameterization (MobileOne), do it before export
try:
    from mobileclip.modules.common.mobileone import reparameterize_model
    model = reparameterize_model(model)
except Exception:
    # not critical if not present
    pass

model.eval()

# Helper: convert module to FP16 but keep numeric-sensitive layers in FP32
KEEP_FLOAT_LAYERS = (nn.LayerNorm, nn.Embedding, nn.BatchNorm2d, nn.GroupNorm)

def convert_module_to_fp16(mod: nn.Module, keep_types=KEEP_FLOAT_LAYERS):
    # First cast entire module to half
    mod.half()
    # Then cast specified types back to float
    for m in mod.modules():
        if isinstance(m, keep_types):
            m.float()
    return mod

# Separate image and text encoders to avoid accidentally casting tokenizer/other ops
image_encoder = model.visual
text_encoder = model.text

# Move to device for tracing
image_encoder.to(device)
text_encoder.to(device)

# Convert to FP16 while retaining safe layers as FP32
convert_module_to_fp16(image_encoder)
convert_module_to_fp16(text_encoder)

# Prepare dummy inputs (move to same device and dtype)
# Get input resolution from the model (visual.image_size)
input_resolution = getattr(model.visual, "image_size", 224)

# Use a real image to make preprocess identical to export time; fallback to rand if not found
img_path = "cat.jpeg"
if os.path.exists(img_path):
    pil = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(pil).unsqueeze(0)  # torch tensor [1,3,H,W] in float32 per preprocess
else:
    print(f"Warning: {img_path} not found - using random image tensor")
    img_tensor = torch.randn(1, 3, input_resolution, input_resolution)

# For FP16 export we must cast inputs consistent with model expected types
img_tensor = img_tensor.to(device=device, dtype=torch.float16)

# Dummy text tokens (long ints; embeddings remain float32 internally)
seq_len = getattr(model.text, "context_length", 77)
dummy_text = torch.randint(low=0, high=model.text.vocab_size if hasattr(model.text, "vocab_size") else 10000,
                           size=(1, seq_len), dtype=torch.long, device=device)

# Export image encoder to FP16 ONNX
print("Exporting image encoder to FP16 ONNX:", IMAGE_ONNX_FP16)
torch.onnx.export(
    image_encoder,
    img_tensor,
    IMAGE_ONNX_FP16,
    export_params=True,
    opset_version=OPSET,
    input_names=["image_input"],
    output_names=["image_features"],
    dynamic_axes={"image_input": {0: "batch_size"}, "image_features": {0: "batch_size"}},
    do_constant_folding=True,
    verbose=False,
)

# Export text encoder to FP16 ONNX
print("Exporting text encoder to FP16 ONNX:", TEXT_ONNX_FP16)
torch.onnx.export(
    text_encoder,
    dummy_text,
    TEXT_ONNX_FP16,
    export_params=True,
    opset_version=OPSET,
    input_names=["text_input"],
    output_names=["text_features"],
    dynamic_axes={"text_input": {0: "batch_size"}, "text_features": {0: "batch_size"}},
    do_constant_folding=True,
    verbose=False,
)

# Optionally simplify ONNX models (onnx-simplifier)
if SIMPLEFY_IF_AVAILABLE:
    try:
        from onnxsim import simplify
        print("Trying to simplify image ONNX...")
        model_onnx = onnx.load(IMAGE_ONNX_FP16)
        model_simp, check = simplify(model_onnx)
        if check:
            onnx.save(model_simp, IMAGE_ONNX_FP16)
            print("Image ONNX simplified.")
        print("Trying to simplify text ONNX...")
        model_onnx = onnx.load(TEXT_ONNX_FP16)
        model_simp, check = simplify(model_onnx)
        if check:
            onnx.save(model_simp, TEXT_ONNX_FP16)
            print("Text ONNX simplified.")
    except Exception as e:
        print("onnx-simplifier not available or simplification failed:", e)

# Quick verification using ONNX Runtime
print("Verifying FP16 ONNX with ONNX Runtime (may use TRT/CUDA providers if available)...")
providers = ort.get_available_providers()
preferred = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if p in providers]
print("Available ORT providers:", providers)
print("Preferred order:", preferred)

sess_img = None
sess_txt = None
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

try:
    sess_img = ort.InferenceSession(IMAGE_ONNX_FP16, sess_options, providers=preferred)
    sess_txt = ort.InferenceSession(TEXT_ONNX_FP16, sess_options, providers=preferred)
    print("Session image providers:", sess_img.get_providers())
    print("Session text providers:", sess_txt.get_providers())

    # Run inference: ONNX Runtime may return float16 arrays; cast to float32 for comparison
    ort_in_img = img_tensor.cpu().numpy().astype(np.float16)
    ort_out_img = sess_img.run(None, {sess_img.get_inputs()[0].name: ort_in_img})[0].astype(np.float32)

    ort_in_txt = dummy_text.cpu().numpy().astype(np.int64)
    ort_out_txt = sess_txt.run(None, {sess_txt.get_inputs()[0].name: ort_in_txt})[0].astype(np.float32)

    # Compute Torch reference (move original models to device and compute)
    # Use mixed dtype: image encoder expects FP16 inputs, so image_encoder currently uses .half()
    with torch.no_grad():
        torch_img_out = image_encoder(img_tensor).cpu().numpy().astype(np.float32)
        torch_txt_out = text_encoder(dummy_text).cpu().numpy().astype(np.float32)

    # Compare with relaxed tolerance because FP16 loses precision
    img_close = np.allclose(torch_img_out, ort_out_img, atol=1e-2, rtol=1e-2)
    txt_close = np.allclose(torch_txt_out, ort_out_txt, atol=1e-2, rtol=1e-2)

    print("Image outputs close:", img_close)
    print("Text outputs close:", txt_close)
    if not img_close:
        print("Max abs diff image:", np.abs(torch_img_out - ort_out_img).max())
    if not txt_close:
        print("Max abs diff text:", np.abs(torch_txt_out - ort_out_txt).max())
except Exception as e:
    print("ONNX Runtime verification failed:", e)
    print("You can still try converting ONNX to a TensorRT engine with trtexec --fp16 (recommended on Jetson).")

print("FP16 export complete. Files:", IMAGE_ONNX_FP16, TEXT_ONNX_FP16)
