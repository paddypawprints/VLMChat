import open_clip
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionSigLIP')

import torch
from PIL import Image

image = preprocess_val(Image.open("hat.jpg")).unsqueeze(0)
text = tokenizer(["a hat", "a t-shirt", "shoes"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image, normalize=True)
    text_features = model.encode_text(text, normalize=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
