"""
CLIP Vision task that generates embeddings for detection crops.

This task extracts image crops for each detection bounding box and generates
CLIP vision embeddings using a pre-initialized CLIPModel. The embeddings can
be used downstream for similarity matching, ranking, or VLM processing.
"""

from typing import List, Optional, Dict
import numpy as np
from PIL import Image
import logging

from ..task_base import BaseTask, Context, ContextDataType, register_task
from ...object_detector.detection_base import Detection

logger = logging.getLogger(__name__)


@register_task('clip_vision')
class ClipVisionTask(BaseTask):
    """
    Generates CLIP vision embeddings for detection crops.
    
    This task:
    1. Takes IMAGE and DETECTIONS from context
    2. Crops the image for each TOP-LEVEL detection bounding box (ignores children)
    3. Creates a batch of cropped images (numpy arrays)
    4. Runs CLIP vision encoder on the batch
    5. Stores embeddings in context as [detection_id, embedding] pairs
    6. Passes through IMAGE and DETECTIONS unchanged
    
    Output format in ContextDataType.EMBEDDINGS:
    [
        ['detection_0', np.ndarray],
        ['detection_1', np.ndarray],
        ...
    ]
    
    The embeddings can be used for:
    - Similarity matching between detections
    - Ranking/scoring detections against text queries
    - Feature vectors for downstream VLM processing
    
    Example:
        clip_task = ClipVisionTask(
            task_id="clip_vision",
            clip_model=clip_model
        )
        
        # After execution, context contains:
        # - ContextDataType.DETECTIONS: Original detections
        # - ContextDataType.IMAGE: Original image
        # - ContextDataType.EMBEDDINGS: List of [id, embedding] pairs
    """
    
    def __init__(self, clip_model=None, task_id: str = "clip_vision", image_format: str = "pil"):
        """
        Initialize CLIP vision task.
        
        Args:
            clip_model: Pre-initialized CLIPModel instance (can be injected later)
            task_id: Unique identifier for this task
            image_format: Expected image format ("pil" or "torch_gpu")
        """
        super().__init__(task_id)
        self.clip_model = clip_model
        self.image_format = image_format.lower()
        
        # Select backend based on required image format
        if self.clip_model is not None:
            if self.image_format == "torch_gpu":
                self.clip_model.set_runtime("tensorrt")
            elif self.image_format == "pil":
                self.clip_model.set_runtime("openclip")
            else:
                raise ValueError(f"Unsupported image format: {self.image_format}")
        
        # Define contracts
        self.input_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.DETECTIONS: list
        }
        self.output_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.DETECTIONS: list,
            ContextDataType.EMBEDDINGS: list  # List of numpy arrays
        }
        
        # Query backend's native format and set our input format requirement
        from ..image_format import ImageFormat
        if self.clip_model and hasattr(self.clip_model, '_runtime') and hasattr(self.clip_model._runtime, 'native_image_format'):
            backend_format = self.clip_model._runtime.native_image_format
            # Map string format to ImageFormat enum
            format_map = {
                "pil": ImageFormat.PIL,
                "numpy": ImageFormat.NUMPY,
                "torch_cpu": ImageFormat.TORCH_CPU,
                "torch_gpu": ImageFormat.TORCH_GPU
            }
            self.native_input_format = format_map.get(backend_format, ImageFormat.PIL)
        else:
            # Default to PIL if backend not initialized or doesn't declare format
            self.native_input_format = ImageFormat.PIL
    
    def configure(self, **kwargs) -> None:
        """
        Configure from DSL parameters.
        
        Args:
            **kwargs: Configuration parameters (currently unused)
        """
        pass  # CLIP model is set at initialization
    
    def _crop_detection_from_image(self, image: Image.Image, detection: Detection) -> np.ndarray:
        """
        Crop image region for a detection box.
        
        Args:
            image: PIL Image
            detection: Detection with bounding box
            
        Returns:
            Cropped image as numpy array
        """
        x1, y1, x2, y2 = detection.box
        
        # Ensure coordinates are within image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image.width, int(x2))
        y2 = min(image.height, int(y2))
        
        # Crop and convert to numpy array
        cropped = image.crop((x1, y1, x2, y2))
        return np.array(cropped)
    
    def run(self, context: Context) -> Context:
        """
        Generate CLIP embeddings for detection crops.
        
        Args:
            context: Input context with IMAGE and DETECTIONS
            
        Returns:
            Context with EMBEDDINGS added as list of [detection_id, embedding] pairs
        """
        # Get or initialize CLIP model from environment
        clip_model = self.clip_model
        if not clip_model:
            from ...pipeline_environment import Environment
            env = Environment.get_instance()
            clip_model = env.get("services", "clip", "model")
            
            # If still not found, try to initialize it
            if not clip_model:
                logger.info(f"Task '{self.task_id}': Initializing CLIP image model...")
                from ...models.MobileClip.clip_image_model import ClipImageModel
                from ...utils.config import VLMChatConfig
                
                config = getattr(context, 'config', None) or VLMChatConfig()
                clip_model = ClipImageModel(config=config)
                env.set("services", "clip", "model", clip_model)
                logger.info(f"Task '{self.task_id}': CLIP model initialized and cached")
        
        if not clip_model:
            raise ValueError(f"Task '{self.task_id}': clip_model required to encode images")
        
        # Get image from context (may be ImageContainer or raw image)
        from ..image_container import ImageContainer
        from ..image_format import ImageFormat
        image_list = context.data.get(ContextDataType.IMAGE)
        if not image_list:
            raise ValueError(f"Task {self.task_id}: IMAGE not found in context")
        
        image_item = image_list[0] if isinstance(image_list, list) else image_list
        
        # Extract actual image data from ImageContainer
        # Note: PipelineRunner will ensure format matches native_input_format
        image: Image.Image
        if isinstance(image_item, ImageContainer) and self.native_input_format:
            # Get the format that runner converted for us
            cached_image = image_item.get_format(self.native_input_format)
            if cached_image is None:
                # Fallback to any available format
                for fmt in image_item.get_cached_formats():
                    cached_image = image_item.get_format(fmt)
                    if cached_image is not None:
                        break
            if cached_image is None:
                raise ValueError(f"Task {self.task_id}: No image format available in container")
            if not isinstance(cached_image, Image.Image):
                if isinstance(cached_image, np.ndarray):
                    image = Image.fromarray(cached_image)
                else:
                    raise TypeError(f"Unexpected image type: {type(cached_image)}")
            else:
                image = cached_image
        else:
            # Fallback for old code paths - convert to PIL if needed
            if isinstance(image_item, Image.Image):
                image = image_item
            elif isinstance(image_item, np.ndarray):
                image = Image.fromarray(image_item)
            else:
                raise TypeError(f"Unexpected image type: {type(image_item)}")
        
        # Get detections from context (only top-level, ignore children)
        detections = context.data.get(ContextDataType.DETECTIONS, [])
        
        if not detections:
            # No detections - encode the entire image instead
            try:
                emb = clip_model.encode(image)
                # Store single embedding for entire image with label
                context.data[ContextDataType.EMBEDDINGS] = [['full_image', emb.cpu().numpy()]]
                return context
            except Exception as e:
                print(f"Error encoding entire image: {e}")
                context.data[ContextDataType.EMBEDDINGS] = []
                return context
        
        # Crop images for each TOP-LEVEL detection only (ignore children)
        cropped_images = []
        detection_ids = []
        
        for det in detections:
            try:
                # Check if detection has an enhanced crop in metadata
                if (hasattr(det, 'metadata') and det.metadata is not None and
                    det.metadata.get('use_enhanced_crop', False) and
                    'enhanced_crop' in det.metadata):
                    
                    # Use the enhanced/swapped crop from metadata
                    enhanced_crop = det.metadata['enhanced_crop']
                    crop_array = np.array(enhanced_crop)
                    cropped_images.append(crop_array)
                    
                    logger.debug(f"Task {self.task_id}: Using enhanced crop for detection {det.id}")
                else:
                    # Standard crop from original image
                    crop = self._crop_detection_from_image(image, det)
                    cropped_images.append(crop)
                
                # Store actual detection ID
                detection_ids.append(det.id)
                    
            except Exception as e:
                logger.warning(f"Task {self.task_id}: Failed to crop detection {det.id}: {e}")
                # Add placeholder for failed crops
                cropped_images.append(np.zeros((224, 224, 3), dtype=np.uint8))
                detection_ids.append(det.id)
        
        # Generate CLIP embeddings for batch
        try:
            embeddings = []
            for crop_array in cropped_images:
                # Convert numpy array to PIL Image
                crop_pil = Image.fromarray(crop_array)
                # Encode single image using ClipImageModel.encode()
                emb = clip_model.encode(crop_pil)
                # Convert torch tensor to numpy and squeeze to get (D,) shape
                raw_emb = emb.cpu().numpy().squeeze()
                embeddings.append(raw_emb)
            
            # Store as list of [detection_id, embedding] pairs
            embedding_pairs = [[det_id, emb] for det_id, emb in zip(detection_ids, embeddings)]
            context.data[ContextDataType.EMBEDDINGS] = embedding_pairs
            
            logger.info(f"Task '{self.task_id}': Generated {len(embedding_pairs)} embeddings, "
                       f"passing through {len(detections)} detections")
            
        except Exception as e:
            logger.error(f"Error generating CLIP embeddings: {e}", exc_info=True)
            # Return empty embeddings on error
            context.data[ContextDataType.EMBEDDINGS] = []
        
        # Pass through IMAGE and DETECTIONS unchanged
        return context
    
    def __str__(self) -> str:
        """String representation."""
        model_name = getattr(self.clip_model, 'model_name', 'unknown')
        return f"ClipVisionTask(id={self.task_id}, model={model_name})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()


if __name__ == "__main__":
    # Example usage (requires actual CLIPModel)
    print("\n--- ClipVisionTask Example ---\n")
    
    # Create mock CLIP model for demonstration
    class MockCLIPModel:
        def __init__(self):
            self.model_name = "MobileCLIP2-S0"
        
        def encode_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
            """Mock encoding - returns random embeddings."""
            print(f"  Encoding {len(images)} image crops...")
            # Return random embeddings (512-dim for MobileCLIP)
            return [np.random.randn(512).astype(np.float32) for _ in images]
    
    # Create test data
    ctx = Context()
    
    # Create a small test image
    test_image = Image.new('RGB', (640, 480), color=(0, 0, 255))  # type: ignore[arg-type]  # Blue
    ctx.data[ContextDataType.IMAGE] = [test_image]
    
    # Create test detections
    det1 = Detection(box=(10, 10, 100, 100), object_category="person", conf=0.92)
    det2 = Detection(box=(200, 200, 300, 300), object_category="horse", conf=0.89)
    det3 = Detection(box=(400, 100, 500, 200), object_category="dog", conf=0.95)
    ctx.data[ContextDataType.DETECTIONS] = [det1, det2, det3]
    
    print(f"Input context:")
    print(f"  IMAGE: {test_image.width}x{test_image.height}")
    print(f"  DETECTIONS: {len(ctx.data[ContextDataType.DETECTIONS])}")
    
    # Create CLIP vision task with mock model
    mock_clip = MockCLIPModel()
    clip_task = ClipVisionTask(clip_model=mock_clip, task_id="clip_vision")
    print(f"\n{clip_task}")
    
    # Run task
    output_ctx = clip_task.run(ctx)
    
    embeddings = output_ctx.data.get(ContextDataType.EMBEDDINGS, [])
    print(f"\nOutput context:")
    print(f"  IMAGE: {bool(output_ctx.data.get(ContextDataType.IMAGE))}")
    print(f"  DETECTIONS: {len(output_ctx.data[ContextDataType.DETECTIONS])}")
    print(f"  EMBEDDINGS: {len(embeddings)}")
    
    if embeddings:
        print(f"\nEmbedding details:")
        for i, emb in enumerate(embeddings):
            print(f"  [{i}] shape={emb.shape}, dtype={emb.dtype}")
    
    print("\n✓ ClipVisionTask generated embeddings for detection crops")
