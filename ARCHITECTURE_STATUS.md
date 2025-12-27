# VLMChat Architecture Status

## Current Architecture (December 2025)

### ✅ Modern Pipeline Architecture (`vlmchat.pipeline.*`)

**USE THIS FOR ALL NEW CODE**

The pipeline architecture is the current, maintained system:

#### Core Components
- `vlmchat.pipeline.core.*` - Pipeline runner, task base, context
- `vlmchat.pipeline.detection` - Detection objects (with bbox, category, confidence)
- `vlmchat.pipeline.cache.*` - Image caching and container system
- `vlmchat.pipeline.services.semantic` - ClipSemanticService for CLIP matching

#### Pipeline Tasks (All Current)
- `vlmchat.pipeline.tasks.yolo_tensorrt` - YOLO object detection
- `vlmchat.pipeline.tasks.clusterer` - Object clustering with semantic similarity
- `vlmchat.pipeline.tasks.viewer` - Visualization with bounding boxes
- `vlmchat.pipeline.tasks.camera` - Camera input
- `vlmchat.pipeline.tasks.smolvlm` - Vision-language model
- Plus many others in `vlmchat.pipeline.tasks/*`

#### Detection Object Structure
```python
from vlmchat.pipeline.detection import Detection, CocoCategory

detection = Detection(
    bbox=(x1, y1, x2, y2),           # Bounding box coordinates
    confidence=0.95,                  # Detection confidence
    category=CocoCategory.PERSON,     # COCO category enum
    source_image=image_container      # Source ImageContainer
)

# Access attributes
detection.bbox           # (x1, y1, x2, y2)
detection.confidence     # float
detection.category       # CocoCategory enum
detection.category.label # "person" (string)
detection.children       # List[Detection] for clusters
```

#### Usage Example
```python
from vlmchat.pipeline.tasks.yolo_tensorrt import YoloTensorRT
from vlmchat.pipeline.tasks.clusterer import ClustererTask
from vlmchat.pipeline.tasks.viewer import ViewerTask
from vlmchat.pipeline.core.task_base import Context, ContextDataType
from vlmchat.pipeline.cache.image import ImageContainer
from vlmchat.pipeline.image.formats import ImageFormat

# Load image
image = Image.open("test.jpg")
img_container = ImageContainer(
    cache_key="test",
    source_data=image,
    source_format=ImageFormat.PIL
)

# Run YOLO
yolo = YoloTensorRT(engine_path="/path/to/yolo.engine")
detections = yolo.detect(img_container, confidence=0.5)

# Cluster detections
ctx = Context()
for det in detections:
    ctx.add_data(ContextDataType.IMAGE, det, "detections")

clusterer = ClustererTask(
    input_label="detections",
    prompts=["person riding a horse"],
    merge_threshold=0.6
)
ctx = clusterer.run(ctx)

# Visualize
viewer = ViewerTask(input_label="detections", save_dir="/tmp")
ctx = viewer.run(ctx)
```

---

### ⚠️ DEPRECATED: Legacy Object Detector (`vlmchat.object_detector.*`)

**DO NOT USE FOR NEW CODE - EXISTS ONLY FOR BACKWARD COMPATIBILITY**

#### Deprecated Modules
- ❌ `vlmchat.object_detector.detection_base` - Old Detection class
- ❌ `vlmchat.object_detector.object_clusterer` - Old ObjectClusterer
- ❌ `vlmchat.object_detector.yolo_object_detector` - Old YOLO wrapper
- ❌ `vlmchat.object_detector.detection_viewer` - Old viewer
- ❌ `vlmchat.object_detector.semantic_provider` - Old semantic interface

#### Why Deprecated?
1. **Non-standard architecture** - Doesn't use pipeline task system
2. **Legacy Detection format** - Uses `.box`, `.object_category`, `.conf` instead of `.bbox`, `.category`, `.confidence`
3. **Poor integration** - Can't use with modern pipeline tasks
4. **No longer maintained** - Bug fixes and features only added to pipeline version

#### Old Detection Object Structure (Don't Use)
```python
from vlmchat.object_detector.detection_base import Detection

# OLD FORMAT - Don't use this!
detection = Detection(
    box=(x1, y1, x2, y2),          # ❌ Should be "bbox"
    object_category="person",       # ❌ Should be CocoCategory enum
    conf=0.95                       # ❌ Should be "confidence"
)
```

---

## Migration Guide

### If You Have Code Using Legacy `object_detector`

**Replace this:**
```python
from vlmchat.object_detector.object_clusterer import ObjectClusterer
from vlmchat.object_detector.detection_base import Detection

clusterer = ObjectClusterer(source=detector, ...)
detections = clusterer.detect(image)
```

**With this:**
```python
from vlmchat.pipeline.tasks.clusterer import ClustererTask
from vlmchat.pipeline.detection import Detection
from vlmchat.pipeline.core.task_base import Context, ContextDataType

# Create context with detections
ctx = Context()
for det in detections:
    ctx.add_data(ContextDataType.IMAGE, det, "detections")

# Run clusterer
clusterer = ClustererTask(input_label="detections", prompts=["..."])
ctx = clusterer.run(ctx)

# Extract results
results = ctx.data[ContextDataType.IMAGE]["detections"]
```

### Key Differences

| Aspect | Legacy (object_detector) | Modern (pipeline) |
|--------|-------------------------|-------------------|
| Detection class | `detection_base.Detection` | `pipeline.detection.Detection` |
| Bbox attribute | `.box` | `.bbox` |
| Category attribute | `.object_category` (str) | `.category` (CocoCategory) |
| Confidence attribute | `.conf` | `.confidence` |
| Clustering | `ObjectClusterer` class | `ClustererTask` task |
| Execution model | Direct method calls | Context-based task.run() |
| Integration | Standalone | Works with all pipeline tasks |

---

## Status Summary

✅ **Use `vlmchat.pipeline.*` for all new code**
- Modern architecture
- Actively maintained
- Full feature set
- Consistent interface

❌ **Avoid `vlmchat.object_detector.*`**
- Legacy code only
- No new features
- Will be removed eventually
- Marked as deprecated

---

## Questions?

See:
- `PIPELINE.md` - Full pipeline architecture documentation
- `src/vlmchat/pipeline/tasks/` - Task implementation examples
- `test_*.py` files - Usage examples
- Code comments with ⚠️ DEPRECATED warnings
