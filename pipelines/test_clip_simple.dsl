# Simple CLIP test - just encode some text
# Tests that CLIP model loads from HuggingFace Hub

{
  # Collect text input
  input() -> :break_on(code=1):
  
  # Encode with CLIP
  -> clip_text_encoder()
  
  # Display the embeddings
  -> output(types="embeddings")
}
