#!/usr/bin/env python3
"""
SmolVLM batch processor with YOLO + Clusterer pipeline.

Usage:
    python smol_batch.py --image <path> [--model 256M|500M] [--prompt "<prompt>"]
    
Runs YOLO detection, clusters objects, extracts crops, and sends all crops
as a batch to SmolVLM for description.
"""
import sys
import logging
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import gc
import time

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
models_path = Path(__file__).parent / "src" / "vlmchat" / "models"
sys.path.insert(0, str(models_path))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Memory tracking utility
def print_gpu_memory(label=""):
    """Print current GPU memory usage (Jetson-compatible)."""
    try:
        # Try jetson-stats first (jtop)
        try:
            import jtop
            with jtop.jtop() as jetson:
                if jetson.ok():
                    gpu_mem = jetson.memory['GPU']
                    used_mb = gpu_mem['used'] / (1024 * 1024)  # Convert to MB
                    total_mb = gpu_mem['tot'] / (1024 * 1024)
                    print(f"[GPU Memory {label}] {used_mb:.0f}MB / {total_mb:.0f}MB ({used_mb/total_mb*100:.1f}%)")
                    return
        except (ImportError, Exception):
            pass
        
        # Try nvidia-smi fallback
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            used, total = result.stdout.strip().split(',')
            used_mb = int(used.strip())
            total_mb = int(total.strip())
            print(f"[GPU Memory {label}] {used_mb}MB / {total_mb}MB ({used_mb/total_mb*100:.1f}%)")
    except Exception:
        pass  # GPU monitoring not available

# Timing utility
class Timer:
    """Simple context manager for timing code blocks."""
    def __init__(self, name):
        self.name = name
        self.start = None
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"[Timing] {self.name}: {elapsed:.2f}s")

# Pipeline imports
from vlmchat.pipeline.models.yolo_tensorrt import YoloTensorRT
from vlmchat.pipeline.tasks.clusterer import ClustererTask
from vlmchat.pipeline.cache.image import ImageContainer
from vlmchat.pipeline.image.formats import ImageFormat
from vlmchat.pipeline.core.task_base import Context, ContextDataType

# SmolVLM imports
from vlmchat.pipeline.models.smolvlm_vision_onnx import SmolVLMVisionOnnx
from vlmchat.pipeline.models.smolvlm_embed_onnx import SmolVLMEmbedOnnx
from vlmchat.pipeline.models.smolvlm_decoder_onnx import SmolVLMDecoderOnnx
from typing import List, Dict, Generator

try:
    from transformers import AutoProcessor, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("ERROR: transformers not available")
    sys.exit(1)


class BatchSmolVLM:
    """SmolVLM wrapper optimized for batch processing of image crops."""
    
    def __init__(self, model_path: str, model_size: str = "256M", sequential_load: bool = False):
        self.model_path = Path(model_path)
        self.model_size = model_size
        self.sequential_load = sequential_load
        
        # Determine HuggingFace model name based on size
        hf_model_name = f"HuggingFaceTB/SmolVLM2-{model_size}-Instruct"
        
        # Load processor and config
        self.config = AutoConfig.from_pretrained(hf_model_name)
        self.processor = AutoProcessor.from_pretrained(hf_model_name)
        self.tokenizer = self.processor.tokenizer
        
        # Extract model parameters
        text_config = self.config.text_config
        self.num_hidden_layers = text_config.num_hidden_layers
        self.num_key_value_heads = text_config.num_key_value_heads
        self.head_dim = text_config.head_dim
        self.image_token_id = self.config.image_token_id
        
        # Handle EOS token IDs
        cfg_eos = text_config.eos_token_id
        if cfg_eos is not None:
            if isinstance(cfg_eos, (list, tuple)):
                self.eos_token_id = list(cfg_eos)
            else:
                self.eos_token_id = [int(cfg_eos)]
        else:
            self.eos_token_id = []
        
        # Add tokenizer EOS if different
        if self.tokenizer.eos_token_id not in self.eos_token_id:
            self.eos_token_id.append(self.tokenizer.eos_token_id)
        
        # Load backends based on mode
        if sequential_load:
            # Only load vision encoder initially, defer others until needed
            self.vision = SmolVLMVisionOnnx(str(self.model_path / "vision_encoder.onnx"), device="cuda")
            self.embed = None
            self.decoder = None
        else:
            # Load all backends upfront
            self.vision = SmolVLMVisionOnnx(str(self.model_path / "vision_encoder.onnx"), device="cuda")
            self.embed = SmolVLMEmbedOnnx(str(self.model_path / "embed_tokens.onnx"), device="cuda")
            self.decoder = SmolVLMDecoderOnnx(
                str(self.model_path / "decoder_model_merged.onnx"),
                num_hidden_layers=self.num_hidden_layers,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                device="cuda",
                model_size=self.model_size  # Pass model size for GPU decision
            )
    
    def encode_images_batch(self, images: List[Image.Image]) -> List[np.ndarray]:
        """Encode multiple images and return list of features (one per image)."""
        all_features = []
        for image in images:
            pixel_inputs = self.processor(images=[image], return_tensors="np")
            pixel_values = pixel_inputs['pixel_values']
            pixel_attention_mask = pixel_inputs['pixel_attention_mask']
            
            image_features = self.vision.encode(pixel_values, pixel_attention_mask)
            all_features.append(image_features)
        
        # Return list of features, not concatenated
        # Each image may have different number of patches
        return all_features
    
    def unload_vision_encoder(self):
        """Unload vision encoder to free GPU memory."""
        if self.vision is not None:
            del self.vision
            self.vision = None
            import gc
            gc.collect()
            print("✓ Vision encoder unloaded")
    
    def load_text_backends(self):
        """Load embed and decoder backends (for sequential loading mode)."""
        if self.embed is None:
            print("Loading embed layer...")
            self.embed = SmolVLMEmbedOnnx(str(self.model_path / "embed_tokens.onnx"), device="cuda")
        
        if self.decoder is None:
            print("Loading decoder...")
            self.decoder = SmolVLMDecoderOnnx(
                str(self.model_path / "decoder_model_merged.onnx"),
                num_hidden_layers=self.num_hidden_layers,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                device="cuda",
                model_size=self.model_size  # Pass model size for GPU decision
            )
        print("✓ Text backends loaded")
    
    def prepare_inputs_batch(self, prompt: str, images: List[Image.Image]) -> Dict[str, np.ndarray]:
        """Prepare inputs for multiple images with prompt."""
        # Create structured message format with multiple images
        content = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt_text, images=images, return_tensors="np")
        
        return inputs
    
    def generate_with_cached_features(self, prompt: str, cached_features: List[np.ndarray], 
                                     num_images: int, max_new_tokens: int = 128) -> str:
        """Generate text using pre-cached image features.
        
        Supports image tags like <image1>, <image2>, <image3> to control which images
        are included and where. If no tags are found, all images are prepended.
        """
        import re
        
        # Check for image tags like <image1>, <image2>, etc.
        image_tag_pattern = r'<image(\d+)>'
        image_tags = re.findall(image_tag_pattern, prompt)
        
        if image_tags:
            # User specified which images to use
            # Convert to 0-based indices
            selected_indices = [int(idx) - 1 for idx in image_tags]
            
            # Validate indices
            valid_indices = [idx for idx in selected_indices if 0 <= idx < num_images]
            if len(valid_indices) != len(selected_indices):
                print(f"Warning: Some image indices out of range (available: 1-{num_images})")
            
            # Replace <imageN> with <image> in order
            prompt_text_clean = re.sub(image_tag_pattern, '<image>', prompt)
            
            # Build content with images in the positions specified
            # Split by <image> tags to interleave text and images
            parts = prompt_text_clean.split('<image>')
            
            content = []
            for i, part in enumerate(parts):
                if part:  # Add text if non-empty
                    if i == 0:
                        # First part - prepend images
                        for idx in valid_indices[:i+1] if i < len(valid_indices) else []:
                            content.append({"type": "image"})
                    content.append({"type": "text", "text": part})
                if i < len(parts) - 1 and i < len(valid_indices):
                    # Add image between parts
                    content.append({"type": "image"})
            
            # Simpler approach: just build content with <image> markers in position
            content = []
            img_idx = 0
            for part in prompt_text_clean.split('<image>'):
                if img_idx > 0:
                    # Add image before this text part
                    content.append({"type": "image"})
                if part.strip():
                    content.append({"type": "text", "text": part})
                img_idx += 1
            
            selected_features = [cached_features[idx] for idx in valid_indices]
        else:
            # No image tags - treat as text-only query (no images)
            selected_indices = []
            selected_features = []
            
            content = [{"type": "text", "text": prompt}]
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt_text, return_tensors="np", padding=True)
        
        batch_size = inputs['input_ids'].shape[0]
        
        # Initialize past key-values
        past_key_values = self.decoder.initialize_past_key_values(batch_size)
        
        # Prepare initial inputs
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = np.cumsum(attention_mask, axis=-1)
        
        # Generation loop
        generated_tokens = []
        for i in range(max_new_tokens):
            # Get token embeddings
            inputs_embeds = self.embed.embed(input_ids)
            
            # Inject cached image features on first step
            if i == 0 and self.image_token_id is not None:
                image_mask = input_ids == self.image_token_id
                if image_mask.any():
                    # Get positions of <image> tokens
                    image_positions = np.where(image_mask[0])[0]
                    
                    # Each <image> token gets replaced by one image's embeddings
                    # Build new embeddings array with images expanded
                    new_embeds_list = []
                    img_idx = 0
                    
                    for pos in range(inputs_embeds.shape[1]):
                        if image_mask[0, pos]:
                            # This is an <image> token - replace with that image's features
                            if img_idx < len(selected_features):
                                features = selected_features[img_idx]
                                # Flatten: (num_patches, 64, hidden_dim) -> (num_patches*64, hidden_dim)
                                num_patches, embeddings_per_patch, hidden_dim = features.shape
                                flat_features = features.reshape(num_patches * embeddings_per_patch, hidden_dim)
                                new_embeds_list.append(flat_features)
                                img_idx += 1
                            else:
                                # Shouldn't happen, but keep original embedding
                                new_embeds_list.append(inputs_embeds[0:1, pos])
                        else:
                            # Regular text token
                            new_embeds_list.append(inputs_embeds[0:1, pos])
                    
                    # Concatenate along sequence dimension
                    inputs_embeds = np.concatenate(new_embeds_list, axis=0)[np.newaxis, :, :]
                    
                    # Update attention mask and position ids to match new sequence length
                    new_seq_len = inputs_embeds.shape[1]
                    attention_mask = np.ones((1, new_seq_len), dtype=attention_mask.dtype)
                    position_ids = np.arange(new_seq_len, dtype=position_ids.dtype)[np.newaxis, :]
            
            # Decoder step
            logits, past_key_values = self.decoder.decode(
                inputs_embeds,
                attention_mask,
                position_ids,
                past_key_values
            )
            
            # Sample next token (greedy)
            next_token_id = int(logits[:, -1].argmax(-1)[0])
            
            # Check for EOS
            if next_token_id in self.eos_token_id:
                break
            
            generated_tokens.append(next_token_id)
            
            # Update for next step
            input_ids = np.array([[next_token_id]], dtype=np.int64)
            attention_mask = np.ones_like(input_ids)
            position_ids = position_ids[:, -1:] + 1
        
        # Decode tokens to text
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response


def run_yolo_clusterer(image_path: Path, prompts: List[str], cleanup: bool = False) -> tuple:
    """
    Run YOLO + Clusterer pipeline on image.
    
    Returns:
        List of (crop_image, detection) tuples
    """
    print(f"\n[1] Loading image: {image_path}")
    print_gpu_memory("before YOLO")
    image = Image.open(image_path)
    print(f"    Image size: {image.size}")
    
    # Run YOLO inference
    print(f"\n[2] Running YOLO detection...")
    model_path = Path("/home/patrick/Dev/model-rt-build/platform/jetson/release_artifacts/yolov8n_fp16.engine")
    
    img_container = ImageContainer(
        cache_key=image_path.stem,
        source_data=image,
        source_format=ImageFormat.PIL
    )
    
    with Timer("YOLO load"):
        yolo_model = YoloTensorRT(engine_path=str(model_path))
        print_gpu_memory("after YOLO load")
    
    with Timer("YOLO detection"):
        detections = yolo_model.detect(img_container, confidence=0.25, iou=0.45)
    
    print(f"    Found {len(detections)} objects")
    
    # Create context and add detections
    print(f"\n[3] Running clustering with prompts: {prompts}")
    ctx = Context()
    for det in detections:
        ctx.add_data(ContextDataType.IMAGE, det, "detections")
    
    for prompt in prompts:
        ctx.add_data(ContextDataType.TEXT, prompt, "prompts")
    
    # Run clusterer
    clusterer = ClustererTask(
        task_id="smol_clusterer",
        input_label="detections",
        output_label="clustered",
        max_clusters=10,
        max_detections=8,
        merge_threshold=0.65,
        proximity_weight=0.5,
        size_weight=0.5,
        semantic_weight=1.5,
        visual_weight=0.0,  # Disabled - not needed for watch list matching
        prob_gain=8.0,
        enable_semantic=True,
        enable_visual=False,  # Disabled - saves ~500MB GPU memory
        enable_cluster_validation=True,
        cluster_prompt_threshold=0.35
    )
    print_gpu_memory("after Clusterer load")
    ctx = clusterer.run(ctx)
    
    # Get clustered detections
    clustered = ctx.data.get(ContextDataType.IMAGE, {}).get("clustered", [])
    print(f"    Result: {len(detections)} objects → {len(clustered)} clusters")
    
    # Extract crops
    print(f"\n[4] Extracting {len(clustered)} crops...")
    crops = []
    max_width = 0
    max_height = 0
    
    with Timer("Crop extraction"):
        for i, det in enumerate(clustered):
            # Materialize the detection as a PIL image
            crop_img = det.materialize(format=ImageFormat.PIL)
            
            if crop_img is not None:
                crops.append((crop_img, det))
                print(f"    Crop {i+1}: {det.category.label} (conf={det.confidence:.2f}) size={crop_img.size}")
                max_width = max(max_width, crop_img.width)
                max_height = max(max_height, crop_img.height)
    
    # Add scaled-down full image as first item
    if crops and max_width > 0 and max_height > 0:
        # Calculate scale to match largest crop dimension
        scale_w = max_width / image.width
        scale_h = max_height / image.height
        scale = max(scale_w, scale_h)  # Use larger scale to fit largest dimension
        
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        full_image_scaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a dummy detection object for the full image
        class FullImageMarker:
            def __init__(self):
                self.category = type('obj', (object,), {'label': 'full_scene'})()
                self.confidence = 1.0
        
        crops.insert(0, (full_image_scaled, FullImageMarker()))
        print(f"    Full image (scaled): size={full_image_scaled.size}")
    
    # Cleanup function to free memory
    def cleanup_models():
        nonlocal yolo_model, clusterer
        if yolo_model is not None:
            del yolo_model
            yolo_model = None
        if clusterer is not None:
            del clusterer
            clusterer = None
        
        # Force garbage collection and CUDA memory release
        import gc
        gc.collect()
        
        # Explicitly clear CUDA cache (critical for TensorRT models)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        
        # Force ONNX Runtime to release GPU memory
        try:
            import onnxruntime as ort
            # Trigger any pending cleanup
            gc.collect()
        except ImportError:
            pass
        
        print("✓ YOLO and Clusterer unloaded")
        print_gpu_memory("after cleanup")
    
    if cleanup:
        return crops, cleanup_models
    else:
        return crops, None


def main():
    parser = argparse.ArgumentParser(
        description='SmolVLM Batch Processor with YOLO + Clusterer',
        epilog='''
Memory requirements (Jetson Orin Nano 8GB):
  256M model: ~3.5GB GPU (recommended - works reliably with all features)
  500M model: ~5-6GB GPU (may OOM during vision encoding - use at your own risk)
  
For security camera use: 256M model provides excellent performance with stable memory usage.
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file')
    parser.add_argument('--model', type=str, default='256M', choices=['256M', '500M'],
                        help='Model size to use (default: 256M - recommended for Jetson)')
    parser.add_argument('--prompt', type=str, default='person riding horse,person wearing white hat',
                        help='Comma-separated clustering prompts (default: person riding horse,person wearing white hat)')
    parser.add_argument('--sequential-load', action='store_true',
                        help='Unload YOLO and CLIP after clustering to save GPU memory, but keep SmolVLM loaded (recommended for all models)')
    parser.add_argument('--warmup', action='store_true',
                        help='Run warmup inference to build TensorRT engines and cache optimizations before actual processing')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SmolVLM Batch Processor ({args.model}) - Interactive Mode")
    print(f"{'='*70}")
    
    # Check paths
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1
    
    model_path = Path.home() / "onnx" / f"SmolVLM2-{args.model}-Instruct"
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        print(f"Please ensure the SmolVLM2-{args.model}-Instruct model is downloaded to ~/onnx/")
        return 1
    
    # Warn about 500M memory requirements
    if args.model == '500M':
        print(f"\n{'⚠'*35}")
        print(f"WARNING: 500M model requires ~5-6GB GPU memory")
        print(f"On Jetson Orin Nano (8GB), vision encoding may fail with OOM")
        print(f"and fall back to slow CPU execution (~250s vs ~25s for 4 images).")
        print(f"\nRecommendation: Use 256M model for reliable performance.")
        print(f"{'⚠'*35}")
        if not args.sequential_load:
            print(f"\nTip: Add --sequential-load to free YOLO/CLIP memory (may help)")
        import time
        time.sleep(2)  # Give user time to read warning
    
    # Parse prompts
    prompts = [p.strip() for p in args.prompt.split(',')]
    
    # Warmup phase if requested
    if args.warmup:
        print(f"\n{'='*70}")
        print(f"WARMUP PHASE - Building TensorRT engines and caching optimizations")
        print(f"{'='*70}")
        
        # Create a small dummy image for warmup
        dummy_image = Image.new('RGB', (640, 640), color='red')
        dummy_path = Path("/tmp/warmup_image.jpg")
        dummy_image.save(dummy_path)
        
        print("\n[Warmup] Running YOLO + Clusterer...")
        with Timer("Warmup: YOLO + Clusterer"):
            warmup_crops, _ = run_yolo_clusterer(dummy_path, prompts, cleanup=False)
        
        print("\n[Warmup] Loading and running SmolVLM vision encoder...")
        with Timer("Warmup: SmolVLM vision"):
            warmup_model = BatchSmolVLM(str(model_path), model_size=args.model, sequential_load=False)
            if warmup_crops:
                warmup_images = [crop_img for crop_img, _ in warmup_crops[:1]]  # Just first crop
            else:
                warmup_images = [dummy_image]
            _ = warmup_model.encode_images_batch(warmup_images)
        
        print("\n[Warmup] Testing text generation...")
        with Timer("Warmup: Text generation"):
            _ = warmup_model.generate_with_cached_features(
                "test", 
                warmup_model.encode_images_batch([dummy_image]),
                num_images=1,
                max_new_tokens=10
            )
        
        # Cleanup warmup
        del warmup_model
        gc.collect()
        dummy_path.unlink()
        
        print(f"\n{'='*70}")
        print(f"WARMUP COMPLETE - All engines cached and ready")
        print(f"{'='*70}")
    
    # Run YOLO + Clusterer pipeline
    print(f"\n[2] Running YOLO + Clusterer pipeline...")
    if args.sequential_load:
        print("    (Sequential loading mode: will unload models after each phase)")
    crops_data, cleanup_fn = run_yolo_clusterer(image_path, prompts, cleanup=args.sequential_load)
    
    if not crops_data:
        print("\n⚠ No clusters found")
        return 0
    
    # Extract just the crop images
    crop_images = [crop_img for crop_img, _ in crops_data]
    crop_labels = [f"{det.category.label} (conf={det.confidence:.2f})" for _, det in crops_data]
    
    # Cleanup YOLO/Clusterer if sequential loading
    if args.sequential_load and cleanup_fn is not None:
        print(f"\n[4] Freeing GPU memory...")
        with Timer("Cleanup YOLO/Clusterer"):
            cleanup_fn()
        
        # Aggressive memory cleanup and synchronization
        import gc
        import time
        
        # Multiple GC passes help with circular references
        gc.collect()
        gc.collect()
        
        # Force CUDA synchronization if torch available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        
        # Give GPU driver time to complete memory releases
        time.sleep(1.0)
        print_gpu_memory("after cleanup and sync")
    
    # Load SmolVLM model
    print(f"\n[5] Loading SmolVLM-{args.model} model...")
    print_gpu_memory("before SmolVLM load")
    with Timer(f"SmolVLM-{args.model} load"):
        # Load all backends upfront, even in sequential mode
        model = BatchSmolVLM(str(model_path), model_size=args.model, sequential_load=False)
    print("✓ Model loaded")
    print_gpu_memory("after SmolVLM load")
    
    # Extra memory cleanup before vision encoding (critical for 500M model)
    if args.sequential_load:
        print("\n[6] Final memory cleanup before vision encoding...")
        import gc
        gc.collect()
        time.sleep(0.5)
        print_gpu_memory("before vision encoding")
    
    # Encode all images once and cache the features
    print(f"\n[7] Encoding {len(crop_images)} crop images...")
    print(f"    Crops: {', '.join(crop_labels)}")
    with Timer("Vision encoding"):
        cached_features = model.encode_images_batch(crop_images)
    print(f"✓ Image features cached: {len(cached_features)} images")
    print(f"    Shapes: {[f.shape for f in cached_features]}")
    print_gpu_memory("after vision encoding")
    
    # Note: In sequential mode, we keep SmolVLM loaded but cleaned up YOLO/CLIP
    if args.sequential_load:
        print(f"\n[8] Sequential mode: Keeping SmolVLM models loaded for optimal performance")
    
    # Interactive loop
    print(f"\n{'='*70}")
    print(f"Interactive Mode - Ask questions about the {len(crop_images)} detected clusters")
    print(f"Clusters: {', '.join(crop_labels)}")
    print(f"\nTip: Use <image1>, <image2>, <image3> to reference specific images")
    print(f"     Example: 'describe <image1>' or 'what is in <image2>?'")
    print(f"     No tags = text-only query (no images sent to model)")
    print(f"Type your question (or 'exit' to quit)")
    print(f"{'='*70}")
    
    try:
        while True:
            try:
                question = input("\nYou: ").strip()
            except EOFError:
                print("\n\nExiting...")
                break
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                break
            
            # Reduce max tokens for 500M to save memory
            max_tokens = 128 if args.model == '500M' else 256
            
            # Generate response using cached features
            print("\nSmolVLM: ", end='', flush=True)
            response = model.generate_with_cached_features(
                question, 
                cached_features, 
                num_images=len(crop_images),
                max_new_tokens=max_tokens
            )
            print(response)
    
    except KeyboardInterrupt:
        print("\n\nExiting...")
    
    # After exiting interactive mode, test warmed vision encoding performance
    print(f"\n{'='*70}")
    print(f"Testing warmed vision encoder performance...")
    print(f"Re-encoding the same {len(crop_images)} images to measure speed")
    print(f"{'='*70}\n")
    
    with Timer("Vision encoding (warmed)"):
        warmed_features = model.encode_images_batch(crop_images)
    
    print(f"✓ Warmed encoding complete: {len(warmed_features)} images")
    avg_time = 251.24 / len(crop_images)  # Will be overwritten by actual timing
    print(f"  Expected improvement: First run had cold TensorRT cache")
    print(f"  Warmed run should be ~10x faster per image\n")
    
    print(f"{'='*70}")
    print("✓ Session complete")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
