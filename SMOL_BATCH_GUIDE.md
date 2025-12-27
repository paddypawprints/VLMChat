# SmolVLM Batch Processor - Deployment Guide

## Quick Start (Jetson Orin Nano 8GB)

**Recommended configuration for security camera:**
```bash
python smol_batch.py \
  --image path/to/frame.jpg \
  --model 256M \
  --sequential-load \
  --prompt "person with backpack,person wearing red jacket,suspicious activity"
```

## Model Comparison

### 256M Model (✅ RECOMMENDED)
- **GPU Memory**: ~3.5GB total
- **Performance**: Stable and reliable
- **Vision Encoding**: ~25s for 4 images (warmed)
- **Status**: Works perfectly with all features
- **Use for**: Production deployments, security cameras

### 500M Model (⚠️ NOT RECOMMENDED)
- **GPU Memory**: ~5-6GB total
- **Issue**: Vision encoder OOMs, falls back to CPU
- **Vision Encoding**: ~250s for 4 images (CPU fallback)
- **Status**: 10x slower, unreliable on 8GB GPU
- **Use for**: Only if you have >12GB GPU memory

## Memory Management Strategy

### Sequential Loading (`--sequential-load`)
1. **YOLO Detection**: Load → Detect objects → Unload (~1GB freed)
2. **CLIP Clustering**: Load → Cluster semantics → Unload (~1GB freed)
3. **SmolVLM Analysis**: Load once → Keep loaded → Fast repeated queries

**Why this works:**
- Detection models only needed at frame start
- VLM stays loaded for multiple watch list queries per frame
- Preserves TensorRT runtime optimizations (warm cache)
- Avoids 10x performance penalty from reloading

### Memory Timeline (256M Sequential Mode)
```
Baseline:           ~500MB
+ YOLO:            ~1500MB
+ CLIP:            ~2500MB
- Cleanup:         ~500MB  ← YOLO/CLIP freed
+ SmolVLM (all):   ~3500MB  ← Vision + Embed + Decoder
During encoding:   ~4000MB
During generation: ~4500MB
```

### Memory Timeline (500M Sequential Mode)
```
Baseline:           ~500MB
+ YOLO:            ~1500MB
+ CLIP:            ~2500MB
- Cleanup:         ~500MB
+ SmolVLM (all):   ~5000MB  ← Vision + Embed + Decoder
During encoding:   ~6500MB  ❌ OOM! Falls back to CPU
```

## Performance Characteristics

### First Run (Cold Start)
- **TensorRT Engine Building**: 5-6 minutes
- Builds optimized engines for each model
- Cached to disk for subsequent runs

### Subsequent Runs (Warmed)
- **YOLO Detection**: 0.45s
- **CLIP Clustering**: <1s
- **Vision Encoding**: ~6s per image (GPU), ~60s per image (CPU)
- **Text Generation**: ~1-2s per query

### Interactive Mode
- Features cached after initial encoding
- Multiple questions use same cached features
- No re-encoding needed between queries
- Ideal for watch list checking

## Security Camera Workflow

### Frame Processing Loop
```python
1. Capture camera frame
2. Run YOLO detection (0.45s)
3. Run CLIP clustering (1s)
4. Cleanup YOLO/CLIP models
5. Encode clusters with SmolVLM vision (6s/image, one-time)
6. Check watch list items:
   - "Is there a person with a backpack?" (2s)
   - "Is anyone wearing a red jacket?" (2s)
   - "Describe suspicious activity" (2s)
7. Alert if matches found
8. Next frame (keep SmolVLM loaded)
```

**Total per frame**: ~15-20s with 3-4 clusters and 3 watch list queries

## Selective Image Querying

Use `<imageN>` tags to query specific detections:

```
You: describe <image1>
SmolVLM: A person riding a horse through the forest.

You: what is <image2> wearing?
SmolVLM: The person is wearing a white hat and jeans.

You: are <image1> and <image3> the same person?
SmolVLM: No, they appear to be different people.
```

**No tags = all images sent** (memory intensive with 500M model)

## TensorRT Caching Explained

### Two-Level Cache System
1. **Disk Cache** (persistent)
   - Compiled engine files (`.engine`, `.trt`)
   - Survives reboots and process restarts
   - Loaded quickly after first build

2. **Runtime Cache** (volatile)
   - CUDA execution context and optimizations
   - Lives in GPU memory while model loaded
   - Destroyed when model unloaded
   - **10x performance difference** (6s vs 60s per image)

### Why Keep SmolVLM Loaded
- Preserve runtime cache = fast inference
- Unloading → reloading loses runtime optimizations
- Detection models (YOLO/CLIP) not needed after clustering
- Perfect for security camera: analyze once, query many times

## Troubleshooting

### Vision Encoding Falls Back to CPU
**Symptom**: 250s for 4 images instead of 25s
**Cause**: Insufficient GPU memory for TensorRT execution buffer
**Solution**: 
- Use 256M model instead of 500M
- Enable `--sequential-load` flag
- Reduce number of images per batch

### Out of Memory During Generation
**Symptom**: RuntimeException during text generation
**Cause**: Too many images in context (500M model)
**Solution**:
- Use selective queries: `<image1>`, `<image2>`
- Reduce max_tokens (automatically done for 500M)
- Use 256M model

### Slow First Inference
**Symptom**: 5-6 minute wait on first run
**Cause**: TensorRT building optimized engines
**Solution**: This is normal! Use `--warmup` flag to build ahead of time

## Best Practices

1. **Always use `--sequential-load`** for production
2. **Stick with 256M model** on 8GB GPU
3. **Use selective queries** (`<imageN>`) when possible
4. **Keep SmolVLM loaded** between frames
5. **First run warmup** before production deployment
6. **Monitor GPU memory** with `jtop` or `tegrastats`

## Example Commands

### Basic usage (256M)
```bash
python smol_batch.py --image frame.jpg --model 256M --sequential-load
```

### Custom watch list
```bash
python smol_batch.py \
  --image frame.jpg \
  --model 256M \
  --sequential-load \
  --prompt "person with weapon,person in distress,broken window"
```

### First-time setup with warmup
```bash
python smol_batch.py \
  --image frame.jpg \
  --model 256M \
  --sequential-load \
  --warmup
```

### Help
```bash
python smol_batch.py --help
```

## Summary

For reliable security camera deployment on Jetson Orin Nano:
- ✅ Use **256M model**
- ✅ Enable **`--sequential-load`**
- ✅ Keep **SmolVLM loaded** between frames
- ✅ Use **selective queries** for efficiency
- ❌ Avoid **500M model** (insufficient memory)
- ❌ Don't **unload SmolVLM** (loses optimizations)

**Performance**: ~15-20s per frame with 3-4 detections and multiple watch list queries.
