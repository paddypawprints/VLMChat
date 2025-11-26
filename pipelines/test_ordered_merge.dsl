# Test ordered_merge syntax

{
  input() -> :break_on(code=1):
  
  -> [
      clip_text_encoder(),
      clip_vision()
      :ordered_merge(order="0,1"):
    ]
  
  -> output(types="embeddings")
}
