#!/usr/bin/env python3
import time
import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

import torch
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model

# Hard-coded paths / names (edit if needed)
ENGINE_PATH = "image_fp16.engine"
IMAGE_PATH = "cat.jpeg"
MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED_PATH = "/home/patrick/mobileclip2_s0.pt"  # keep as provided in your snippet
TEXT_PROMPT = ["a diagram", "a dog", "a cat"]

# Load TRT engine
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

# Create context and get tensor info (name-based tensor API)
ctx = engine.create_execution_context()
INPUT_TENSOR_NAME = "image_input"
OUTPUT_TENSOR_NAME = "image_features"
in_shape = tuple(ctx.get_tensor_shape(INPUT_TENSOR_NAME))   # e.g. (1,3,256,256)
out_shape = tuple(ctx.get_tensor_shape(OUTPUT_TENSOR_NAME)) # e.g. (1,512)
in_dtype = trt.nptype(engine.get_tensor_dtype(INPUT_TENSOR_NAME))
out_dtype = trt.nptype(engine.get_tensor_dtype(OUTPUT_TENSOR_NAME))

print("TRT engine input shape, dtype:", in_shape, in_dtype)
print("TRT engine output shape, dtype:", out_shape, out_dtype)

# Load MobileCLIP model and preprocess (as in your example)
model_kwargs = {}
if not (MODEL_NAME.endswith("S3") or MODEL_NAME.endswith("S4") or MODEL_NAME.endswith("L-14")):
    model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_PATH, **model_kwargs)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model.eval()
model = reparameterize_model(model)

# Prepare image with the model's preprocess
pil = Image.open(IMAGE_PATH).convert("RGB")
torch_input = preprocess(pil).unsqueeze(0)  # (1,C,H,W), float32 on CPU

# Convert to numpy for TRT and cast to engine dtype
np_input = torch_input.cpu().numpy().reshape(in_shape).astype(in_dtype, copy=False)

# Allocate TRT host/device buffers and bind by tensor name
in_elems = int(np.prod(in_shape))
out_elems = int(np.prod(out_shape))

host_in = cuda.pagelocked_empty(in_elems, in_dtype)
host_out = cuda.pagelocked_empty(out_elems, out_dtype)

host_in[:] = np_input.ravel().astype(in_dtype, copy=False)

d_in = cuda.mem_alloc(host_in.nbytes)
d_out = cuda.mem_alloc(host_out.nbytes)

ctx.set_tensor_address(INPUT_TENSOR_NAME, int(d_in))
ctx.set_tensor_address(OUTPUT_TENSOR_NAME, int(d_out))

stream = cuda.Stream()
cuda.memcpy_htod_async(d_in, host_in, stream)

t0 = time.time()
ctx.execute_async_v3(stream_handle=stream.handle)
t1 = time.time()

cuda.memcpy_dtoh_async(host_out, d_out, stream)
stream.synchronize()
t2 = time.time()

trt_out = np.array(host_out).reshape(out_shape)
if trt_out.dtype == np.float16:
    trt_out = trt_out.astype(np.float32)
trt_vec = trt_out.ravel().astype(np.float32)

print(f"TRT exec ms: {(t1-t0)*1000:.2f}, total ms: {(t2-t0)*1000:.2f}")
print("TRT output shape:", trt_out.shape)

# Run MobileCLIP model directly (image + text) for comparison
# (keeps to your template: no extra guards; may fail depending on device)
with torch.no_grad():
    # image features
    image_tensor = torch_input  # CPU tensor as returned by preprocess
    image_feats = model.encode_image(image_tensor)
    image_feats = image_feats.cpu().numpy().astype(np.float32).ravel()
    # text features
    tokenized = tokenizer(TEXT_PROMPT)
    text_feats = model.encode_text(tokenized)
    text_feats = text_feats.cpu().numpy().astype(np.float32).ravel()
    # normalize
    image_feats /= np.linalg.norm(image_feats)
    text_feats /= np.linalg.norm(text_feats)

# Normalize TRT vector as well
trt_norm = np.linalg.norm(trt_vec)
if trt_norm != 0:
    trt_unit = trt_vec / trt_norm
else:
    trt_unit = trt_vec

# similarity helper
def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print("Norms: TRT", np.linalg.norm(trt_vec), "Torch_image", np.linalg.norm(image_feats), "Torch_text", np.linalg.norm(text_feats))
print("Cosine TRT vs Torch_image:", cos(trt_vec, image_feats))
print("Cosine Torch_image vs Torch_text:", cos(image_feats, text_feats))
print("Cosine TRT vs Torch_text:", cos(trt_vec, text_feats))

# Save outputs for inspection
np.save("trt_image_features.npy", trt_vec)
np.save("torch_image_features.npy", image_feats)
np.save("torch_text_features.npy", text_feats)

print("Saved trt_image_features.npy, torch_image_features.npy, torch_text_features.npy")
