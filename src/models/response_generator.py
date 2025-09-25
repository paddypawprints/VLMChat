import logging
from typing import List, Dict, Any, Generator, Union

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Handles generation of responses from the model."""
    
    def __init__(self, model):
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
                return self._generate_non_streaming_onnx(inputs)
            else:
                return self.model.generate_transformers(inputs)
                
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def _generate_streaming_onnx(self, inputs: Dict[str, Any]) -> str:
        """Generate response with ONNX streaming."""
        response_tokens = []
        for token_text in self.model.generate_onnx(inputs):
            print(token_text, end='', flush=True)
            response_tokens.append(token_text)
        print()  # New line after streaming
        return ''.join(response_tokens)
    
    def _generate_non_streaming_onnx(self, inputs: Dict[str, Any]) -> str:
        """Generate response with ONNX non-streaming."""
        generated_tokens = list(self.model.generate_onnx(inputs))
        return ''.join(generated_tokens)