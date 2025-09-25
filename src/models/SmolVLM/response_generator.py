import logging
from typing import List, Dict, Any
from src.models.SmolVLM.smol_vlm_model import SmolVLMModel

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Handles generation of responses from the model."""
    
    def __init__(self, model: SmolVLMModel):
        """
        Initialize response generator.
        
        Args:
            model: Instance of SmolVLMModel
        """
        self.model = model
    
    def generate_response(self, 
                         messages: List[Dict[str, Any]], 
                         images: List[Any],
                         stream_output: bool = True) -> str:
        """
        Generate a response using the model.
        
        Args:
            messages: List of formatted message dictionaries
            images: List of images to process
            stream_output: Whether to stream tokens as they're generated
            
        Returns:
            Generated response text
        """
        # Prepare model inputs
        inputs = self.model.prepare_inputs(messages, images)
        
        try:
            if self.model.use_onnx and stream_output:
                return self._generate_streaming_onnx(inputs)
            elif self.model.use_onnx:
                return self._generate_streaming_onnx(inputs)
            else:
                return self.model.generate_transformers(inputs)
                
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def _generate_streaming_onnx(self, inputs: Dict[str, Any]) -> str:
        """Generate response with ONNX streaming."""
        response_tokens = []
        for token_text in self.model.generate_onnx(inputs):
            response_tokens.append(token_text)
        response_tokens.append('\n')
        return ''.join(response_tokens)
    
 