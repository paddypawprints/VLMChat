"""
SmolVLM vision-language model task.

Provides image captioning and visual question answering capabilities
using SmolVLM with ONNX Runtime backend.
"""

import logging
from typing import Optional, List
from pathlib import Path

from ..core.task_base import BaseTask, Context, ContextDataType, register_task
from ..detection import Detection
from ..image.formats import ImageFormat
from ..models.smolvlm_onnx import SmolVLMOnnx

logger = logging.getLogger(__name__)


@register_task('smolvlm')
class SmolVLMTask(BaseTask):
    """
    SmolVLM vision-language model for image understanding.
    
    Provides two modes:
    1. Captioning: Generate descriptions of images/detections
    2. VQA: Answer questions about images/detections
    
    Contract:
        Input: IMAGE[input_label] - ImageContainer or Detection objects
        Output: TEXT[output_label] - Generated text (captions or answers)
        Optional Input: TEXT[prompt_label] - Questions/prompts
    
    Usage:
        # Caption mode: generate descriptions
        smolvlm(mode=caption, output=captions)
        
        # VQA mode: answer questions from prompts
        smolvlm(mode=vqa, input=detections, prompts=questions, output=answers)
        
        # With specific model
        smolvlm(model=/path/to/model.onnx, mode=caption)
    """
    
    def __init__(self,
                 task_id: str = "smolvlm",
                 model_path: Optional[str] = None,
                 mode: str = "caption",
                 input_label: str = "frame",
                 output_label: str = "captions",
                 prompt_label: str = "prompts",
                 default_prompt: str = "Describe this image in detail.",
                 max_new_tokens: int = 100,
                 device: str = "cuda"):
        """
        Initialize SmolVLM task.
        
        Args:
            task_id: Unique task identifier
            model_path: Path to ONNX model directory (containing vision_encoder.onnx, etc.)
            mode: Operation mode - "caption" or "vqa"
            input_label: Label to read images from (default: "frame")
            output_label: Label to write text outputs to (default: "captions")
            prompt_label: Label to read prompts from for VQA mode (default: "prompts")
            default_prompt: Default prompt when none provided
            max_new_tokens: Maximum tokens to generate
            device: Device for inference ('cuda' or 'cpu')
        """
        super().__init__(task_id)
        
        # Parameters
        self.model_path = Path(model_path).expanduser() if model_path else Path("~/onnx/SmolVLM2-256M-Instruct").expanduser()
        self.mode = mode.lower()
        self.input_label = input_label
        self.output_label = output_label
        self.prompt_label = prompt_label
        self.default_prompt = default_prompt
        self.max_new_tokens = max_new_tokens
        self.device = device
        
        # Validate mode
        if self.mode not in ["caption", "vqa"]:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'caption' or 'vqa'")
        
        # Model (initialized on first run)
        self.model: Optional[SmolVLMOnnx] = None
        
        # Declare contracts
        from ...cache.types import CachedItemType
        self.input_contract = {
            ContextDataType.IMAGE: {
                input_label: (CachedItemType.IMAGE, "numpy")  # Need numpy arrays
            }
        }
        if mode == "vqa":
            self.input_contract[ContextDataType.TEXT] = {
                prompt_label: (None, None)  # Text prompts/questions
            }
        
        self.output_contract = {
            ContextDataType.TEXT: {
                output_label: (None, None)  # Generated text
            }
        }
    
    def configure(self, **kwargs) -> None:
        """
        Configure from DSL parameters.
        
        Args:
            **kwargs: Configuration parameters
                - model: Model path (str)
                - mode: Operation mode ("caption" or "vqa")
                - input: Input label (str)
                - output: Output label (str)
                - prompts: Prompt label (str)
                - default_prompt: Default prompt text (str)
                - max_new_tokens: Max tokens to generate (int)
                - device: Device ('cuda' or 'cpu')
        """
        if "model" in kwargs:
            self.model_path = Path(kwargs["model"]).expanduser()
        
        if "mode" in kwargs:
            self.mode = kwargs["mode"].lower()
            if self.mode not in ["caption", "vqa"]:
                raise ValueError(f"Invalid mode: {self.mode}")
        
        if "input" in kwargs:
            self.input_label = kwargs["input"]
        
        if "output" in kwargs:
            self.output_label = kwargs["output"]
        
        if "prompts" in kwargs:
            self.prompt_label = kwargs["prompts"]
        
        if "default_prompt" in kwargs:
            self.default_prompt = kwargs["default_prompt"]
        
        if "max_new_tokens" in kwargs:
            self.max_new_tokens = int(kwargs["max_new_tokens"])
        
        if "device" in kwargs:
            self.device = kwargs["device"]
    
    def _initialize_model(self, context: Context) -> None:
        """Initialize SmolVLM model on first run."""
        if self.model is not None:
            return
        
        logger.info(f"[{self.task_id}] Initializing SmolVLM model from: {self.model_path}")
        
        self.model = SmolVLMOnnx(
            model_path=str(self.model_path),
            device=self.device
        )
        
        logger.info(f"[{self.task_id}] SmolVLM model initialized")
    def execute(self, context: Context) -> None:
        """
        Execute SmolVLM inference.
        
        Args:
            context: Pipeline context with input images/detections
        """
        # Initialize model on first run
        self._initialize_model(context)
        
        # Get input images/detections from context
        if ContextDataType.IMAGE not in context.data:
            logger.warning(f"[{self.task_id}] No IMAGE data in context")
            return
        
        if self.input_label not in context.data[ContextDataType.IMAGE]:
            logger.warning(f"[{self.task_id}] No input images found at label '{self.input_label}'")
            return
        
        frames = context.data[ContextDataType.IMAGE][self.input_label]
        if not frames:
            logger.warning(f"[{self.task_id}] Empty image list in label '{self.input_label}'")
            return
        
        logger.info(f"[{self.task_id}] Processing {len(frames)} images in {self.mode} mode")
        
        # Get prompts if in VQA mode
        if self.mode == "vqa":
            prompts_data = context.data.get(ContextDataType.TEXT, {}).get(self.prompt_label, [])
            if prompts_data:
                prompts = prompts_data
            else:
                logger.info(f"[{self.task_id}] No prompts found, using default: '{self.default_prompt}'")
                prompts = [self.default_prompt] * len(frames)
        else:
            # Caption mode - use default prompt for all
            prompts = [self.default_prompt] * len(frames)
        
        # Adjust prompt count to match input count
        if len(prompts) == 1:
            prompts = prompts * len(frames)
        elif len(prompts) != len(frames):
            logger.warning(f"[{self.task_id}] Prompt count ({len(prompts)}) doesn't match input count ({len(frames)})")
            if len(prompts) < len(frames):
                prompts = prompts + [self.default_prompt] * (len(frames) - len(prompts))
            else:
                prompts = prompts[:len(frames)]
        
        # Process each input
        outputs = []
        for idx, (image_item, prompt) in enumerate(zip(frames, prompts)):
            # Get ImageContainer (could be base image or Detection)
            if isinstance(image_item, Detection):
                # Materialize detection's crop as PIL
                pil_image = image_item.materialize(format=ImageFormat.PIL)
            else:
                # Regular ImageContainer
                pil_image = image_item.get(ImageFormat.PIL)
            
            # Prepare inputs
            messages = [{"role": "user", "content": prompt}]
            inputs = self.model.prepare_inputs(messages, [pil_image])
            
            # Generate
            result = self.model.generate(inputs, max_new_tokens=self.max_new_tokens)
            
            outputs.append(result)
            
            prompt_preview = prompt[:50] if len(prompt) > 50 else prompt
            result_preview = result[:100] if len(result) > 100 else result
            logger.debug(f"[{self.task_id}] [{idx}] Prompt: {prompt_preview}... → Result: {result_preview}...")
        
        # Store outputs in context
        if ContextDataType.TEXT not in context.data:
            context.data[ContextDataType.TEXT] = {}
        if self.output_label not in context.data[ContextDataType.TEXT]:
            context.data[ContextDataType.TEXT][self.output_label] = []
        context.data[ContextDataType.TEXT][self.output_label].extend(outputs)
        
        logger.info(f"[{self.task_id}] Generated {len(outputs)} text outputs")
    
    def validate(self, context: Context) -> bool:
        """
        Validate that inputs are available.
        
        Args:
            context: Pipeline context
            
        Returns:
            True if validation passes
        """
        if ContextDataType.IMAGE not in context.data:
            logger.error(f"[{self.task_id}] No IMAGE data in context")
            return False
        
        if self.input_label not in context.data[ContextDataType.IMAGE]:
            logger.error(f"[{self.task_id}] No input images at label '{self.input_label}'")
            return False
        
        frames = context.data[ContextDataType.IMAGE][self.input_label]
        if not frames:
            logger.error(f"[{self.task_id}] Empty image list at label '{self.input_label}'")
            return False
        
        if self.mode == "vqa":
            prompts = context.data.get(ContextDataType.TEXT, {}).get(self.prompt_label, [])
            if not prompts:
                logger.info(f"[{self.task_id}] No prompts found - will use default prompt")
        
        return True
