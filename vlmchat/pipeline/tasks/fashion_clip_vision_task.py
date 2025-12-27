"""
FashionClip Vision task that generates embeddings for detection crops.

This task extracts image crops for each detection bounding box and generates
FashionClip vision embeddings (768-dimensional, optimized for fashion domain).
The embeddings can be used downstream for fashion-specific similarity matching.
"""

from typing import List, Optional, Dict
import numpy as np
from PIL import Image
import logging

from ..task_base import BaseTask, Context, ContextDataType, register_task
from ...object_detector.detection_base import Detection

logger = logging.getLogger(__name__)


@register_task('fashion_clip_vision')
class FashionClipVisionTask(BaseTask):
    """
    Generates FashionClip vision embeddings for detection crops.
    
    This task:
    1. Takes IMAGE and DETECTIONS from context
    2. Crops the image for each TOP-LEVEL detection bounding box (ignores children)
    3. Creates a batch of cropped images (PIL Images)
    4. Runs FashionClip vision encoder on the batch (768-dim embeddings)
    5. Stores embeddings in context (one per top-level detection)
    6. Passes through IMAGE and DETECTIONS unchanged
    
    The embeddings can be used for:
    - Fashion-specific similarity matching
    - Ranking/scoring fashion items against text queries
    - Feature vectors for fashion-aware downstream processing
    
    Example:
        fashion_clip_task = FashionClipVisionTask(
            task_id="fashion_clip_vision",
            fashion_clip_model=fashion_clip_model
        )
        
        # After execution, context contains:
        # - ContextDataType.DETECTIONS: Original detections
        # - ContextDataType.IMAGE: Original image
        # - ContextDataType.EMBEDDINGS: List of FashionClip embeddings (768-dim)
    """
    
    def __init__(self, fashion_clip_model=None, task_id: str = "fashion_clip_vision"):
        """
        Initialize FashionClip vision task.
        
        Args:
            fashion_clip_model: Pre-initialized FashionClipModel instance (can be injected later)
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.fashion_clip_model = fashion_clip_model
        
        # Define contracts
        self.input_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.DETECTIONS: list
        }
        self.output_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.DETECTIONS: list,
            ContextDataType.EMBEDDINGS: list  # List of numpy arrays (768-dim)
        }
    
    def configure(self, **kwargs) -> None:
        """
        Configure from DSL parameters.
        
        Args:
            **kwargs: Configuration parameters (currently unused)
        """
        pass  # FashionClip model is set at initialization
    
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
        Generate FashionClip embeddings for detection crops.
        
        Args:
            context: Input context with IMAGE and DETECTIONS
            
        Returns:
            Context with EMBEDDINGS added
        """
        # Get or initialize FashionClip model
        fashion_clip_model = self.fashion_clip_model
        if not fashion_clip_model:
            from ..environment import Environment
            env = Environment.get_instance()
            fashion_clip_model = env.get("services", "fashion_clip", "model")
            
            # If still not found, try to initialize it
            if not fashion_clip_model:
                logger.info(f"Task '{self.task_id}': Initializing FashionClip model...")
                from ...models.FashionClip.fashion_clip_model import FashionClipModel
                from ...utils.config import VLMChatConfig
                
                # Get config from context if available
                config = getattr(context, 'config', None)
                if not config:
                    config = VLMChatConfig()
                
                fashion_clip_model = FashionClipModel(config=config)
                env.set("services", "fashion_clip", "model", fashion_clip_model)
                logger.info(f"Task '{self.task_id}': FashionClip model initialized and cached")
        
        if not fashion_clip_model:
            raise ValueError(f"Task '{self.task_id}': fashion_clip_model required")
        
        # Use the obtained model
        self.fashion_clip_model = fashion_clip_model
        
        # Get image from context
        image_list = context.data.get(ContextDataType.IMAGE)
        if not image_list:
            raise ValueError(f"Task {self.task_id}: IMAGE not found in context")
        
        image = image_list[0] if isinstance(image_list, list) else image_list
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Get detections from context (only top-level, ignore children)
        detections = context.data.get(ContextDataType.DETECTIONS, [])
        
        if not detections:
            # No detections - encode the entire image instead
            logger.info(f"Task {self.task_id}: No detections found, encoding entire image")
            try:
                runtime = self.fashion_clip_model._runtime_as_fashion_clip()
                emb = runtime.encode_image(image)
                # Store single embedding for entire image
                context.data[ContextDataType.EMBEDDINGS] = [emb.cpu().numpy()]
                return context
            except Exception as e:
                logger.error(f"Task {self.task_id}: Error encoding entire image: {e}")
                context.data[ContextDataType.EMBEDDINGS] = []
                return context
        
        logger.info(f"Task {self.task_id}: Processing {len(detections)} detections")
        
        # Collect detection IDs for pairing with embeddings
        detection_ids = [det.id for det in detections]
        
        # Crop images for each TOP-LEVEL detection only (ignore children)
        cropped_images = []
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
                    
            except Exception as e:
                logger.warning(f"Task {self.task_id}: Failed to crop detection {det.id}: {e}")
                # Add placeholder for failed crops
                cropped_images.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Generate FashionClip embeddings for batch
        try:
            # FashionClipModel uses a runtime backend
            runtime = self.fashion_clip_model._runtime_as_fashion_clip()
            
            embeddings = []
            for crop_array in cropped_images:
                # Convert numpy array to PIL Image
                crop_pil = Image.fromarray(crop_array)
                # Encode single image using runtime
                emb = runtime.encode_image(crop_pil)
                # Convert torch tensor to numpy and squeeze to get (D,) shape
                raw_emb = emb.cpu().numpy().squeeze()
                embeddings.append(raw_emb)
            
            logger.info(f"Task {self.task_id}: Generated {len(embeddings)} FashionClip embeddings (768-dim)")
            
            # Store as list of [detection_id, embedding] pairs (matching clip_vision format)
            embedding_pairs = [[det_id, emb] for det_id, emb in zip(detection_ids, embeddings)]
            context.data[ContextDataType.EMBEDDINGS] = embedding_pairs
            
        except Exception as e:
            logger.error(f"Task {self.task_id}: Error generating FashionClip embeddings: {e}")
            # Return empty embeddings on error
            context.data[ContextDataType.EMBEDDINGS] = []
        
        # Pass through IMAGE and DETECTIONS unchanged
        return context
    
    def __str__(self) -> str:
        """String representation."""
        model_name = getattr(self.fashion_clip_model, 'model_name', 'FashionClip')
        return f"FashionClipVisionTask(id={self.task_id}, model={model_name})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()


if __name__ == "__main__":
    # Example usage (requires actual FashionClipModel)
    print("\n--- FashionClipVisionTask Example ---\n")
    
    # Create mock FashionClip model for demonstration
    class MockFashionClipModel:
        def __init__(self):
            self.model_name = "Marqo/marqo-fashionSigLIP"
        
        def _runtime_as_fashion_clip(self):
            return self
        
        def encode_image(self, image: Image.Image) -> 'MockTensor':
            """Mock encoding - returns random embeddings."""
            print(f"  Encoding image crop...")
            import torch
            # Return random embeddings (768-dim for FashionClip)
            return torch.randn(1, 768)
    
    class MockTensor:
        def __init__(self, data):
            self.data = data
        def cpu(self):
            return self
        def numpy(self):
            return self.data
    
    # Create test data
    ctx = Context()
    
    # Create a small test image
    test_image = Image.new('RGB', (640, 480), color='blue')
    ctx.data[ContextDataType.IMAGE] = [test_image]
    
    # Create test detections (fashion items)
    det1 = Detection(box=(10, 10, 100, 100), object_category="dress", conf=0.92)
    det2 = Detection(box=(200, 200, 300, 300), object_category="shoes", conf=0.89)
    det3 = Detection(box=(400, 100, 500, 200), object_category="hat", conf=0.95)
    ctx.data[ContextDataType.DETECTIONS] = [det1, det2, det3]
    
    print(f"Input context:")
    print(f"  IMAGE: {test_image.width}x{test_image.height}")
    print(f"  DETECTIONS: {len(ctx.data[ContextDataType.DETECTIONS])} fashion items")
    
    # Create FashionClip vision task with mock model
    mock_fashion_clip = MockFashionClipModel()
    fashion_clip_task = FashionClipVisionTask(fashion_clip_model=mock_fashion_clip, 
                                              task_id="fashion_clip_vision")
    print(f"\n{fashion_clip_task}")
    
    # Run task
    output_ctx = fashion_clip_task.run(ctx)
    
    embeddings = output_ctx.data.get(ContextDataType.EMBEDDINGS, [])
    print(f"\nOutput context:")
    print(f"  IMAGE: {bool(output_ctx.data.get(ContextDataType.IMAGE))}")
    print(f"  DETECTIONS: {len(output_ctx.data[ContextDataType.DETECTIONS])}")
    print(f"  EMBEDDINGS: {len(embeddings)}")
    
    if embeddings:
        print(f"\nEmbedding details:")
        for i, emb in enumerate(embeddings):
            print(f"  [{i}] shape={emb.shape}, dtype={emb.dtype}")
    
    print("\n✓ FashionClipVisionTask generated 768-dim embeddings for fashion item crops")
