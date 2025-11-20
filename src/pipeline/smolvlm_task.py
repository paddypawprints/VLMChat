"""
SmolVLM task adapter for pipeline integration.

Wraps a SmolVLM model instance and runs inference on image + prompt.
"""

from typing import Optional
from PIL import Image

from .task_base import BaseTask, Context, ContextDataType
from ..models.SmolVLM.smol_vlm_model import SmolVLMModel
from ..prompt.prompt import Prompt


class SmolVLMTask(BaseTask):
    """
    Pipeline task adapter for SmolVLM inference.
    
    Takes IMAGE and PROMPT from context, runs model inference,
    and stores the response in context as RESPONSE.
    """
    
    def __init__(self, model: SmolVLMModel, prompt: Prompt, task_id: str = "smolvlm"):
        """
        Initialize SmolVLM task with injected model and prompt manager.
        
        Args:
            model: SmolVLM model instance
            prompt: Prompt manager for conversation history
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.model = model
        self.prompt = prompt
        
        # Define contracts
        # Input: Requires IMAGE and PROMPT
        # Output: Produces RESPONSE
        self.input_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.PROMPT: str
        }
        self.output_contract = {ContextDataType.RESPONSE: str}
    
    def run(self, context: Context) -> Context:
        """
        Run SmolVLM inference on image and prompt from context.
        
        Args:
            context: Pipeline context containing IMAGE and PROMPT
            
        Returns:
            Updated context with RESPONSE added
        """
        # Get inputs from context
        image = context.data.get(ContextDataType.IMAGE)
        prompt_text = context.data.get(ContextDataType.PROMPT)
        
        if image is None:
            raise ValueError(f"Task {self.task_id}: IMAGE not found in context")
        if prompt_text is None:
            raise ValueError(f"Task {self.task_id}: PROMPT not found in context")
        
        # Update prompt with current user input and image
        self.prompt._user_input = prompt_text
        self.prompt.current_image = image
        
        # Get formatted messages
        messages = self.model.get_messages(self.prompt)
        
        # Generate response
        response = self.model.generate_response(
            messages=messages,
            images=[image],
            stream_output=False  # Non-streaming for pipeline
        )
        
        # Store response in context
        context.data[ContextDataType.RESPONSE] = response
        
        return context
