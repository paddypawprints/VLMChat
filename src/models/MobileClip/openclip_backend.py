import torch
import open_clip
from PIL import Image
from mobileclip.modules.common.mobileone import reparameterize_model

model_name = "MobileCLIP2-S0"
model_kwargs = {}
if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
    model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="./mobileclip2_s0.pt", **model_kwargs)
tokenizer = open_clip.get_tokenizer(model_name)

# Model needs to be in eval mode for inference because of batchnorm layers unlike ViTs
model.eval()

# For inference/model exporting purposes, please reparameterize first
model = reparameterize_model(model)

image = preprocess(Image.open("trail-riders.jpg").convert("RGB")).unsqueeze(0)
text = tokenizer(["a horse", "a person riding a horse", "a man wearing a hat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
