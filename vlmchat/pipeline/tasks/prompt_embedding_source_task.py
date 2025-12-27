"""
Prompt embedding source task for injecting prompt embeddings into pipeline context.

This task loads prompt embeddings from a file or generates them using the CLIP
text encoder, and injects them into the context for use by downstream tasks.
"""

import logging
import json
import os
from typing import Optional, Dict, List, Any
from ..task_base import BaseTask, Context, ContextDataType, register_task

logger = logging.getLogger(__name__)


@register_task('prompt_embedding_source')
@register_task('prompt_embeddings')
class PromptEmbeddingSourceTask(BaseTask):
    """
    Loads or generates prompt embeddings and injects into context.
    
    This task manages prompt embeddings lifecycle:
    1. Checks if embeddings already exist in context (from previous iteration)
    2. If not, loads from file or generates using CLIP text encoder
    3. Detects file updates and reloads when source changes
    4. Injects into context.data[ContextDataType.PROMPT_EMBEDDINGS]
    
    Designed for continuous pipelines where prompt embeddings persist across
    iterations (via ContextCleanupTask) but need updating when source changes.
    
    Two modes:
    - **File mode**: Loads pre-computed embeddings from JSON file
    - **Onboard mode**: Generates embeddings using CLIP text encoder
    
    Usage:
        # File mode (pre-computed embeddings)
        task = PromptEmbeddingSourceTask(
            embeddings_path="prompt_embeddings.json"
        )
        
        # Onboard mode (generate with CLIP)
        task = PromptEmbeddingSourceTask(
            prompts=["person riding horse", "car on street"],
            clip_model=clip_model
        )
        
        # In continuous pipeline:
        context = Context()
        while True:
            context = pipeline.run(context)  # Task checks for updates
    """
    
    def __init__(self, 
                 embeddings_path: Optional[str] = None,
                 prompts: Optional[List[str]] = None,
                 clip_model: Optional[Any] = None,
                 task_id: str = "prompt_embedding_source"):
        """
        Initialize prompt embedding source task.
        
        Args:
            embeddings_path: Path to JSON file with pre-computed embeddings.
                           If provided, loads from file (file mode).
            prompts: List of text prompts to encode. If provided with clip_model,
                    generates embeddings on-the-fly (onboard mode).
            clip_model: CLIPModel instance for encoding prompts (onboard mode).
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.embeddings_path = embeddings_path
        self.prompts = prompts
        self.clip_model = clip_model
        self._last_mtime = None
        self._version = 0
        
        # Validate configuration
        if embeddings_path and (prompts or clip_model):
            logger.warning(f"{task_id}: Both file path and onboard mode provided. File mode takes precedence.")
        
        if not embeddings_path and not (prompts and clip_model):
            raise ValueError(f"{task_id}: Must provide either embeddings_path (file mode) or prompts+clip_model (onboard mode)")
        
        # Determine mode
        self.mode = "file" if embeddings_path else "onboard"
        
        # Define contracts
        self.input_contract = {}  # No required inputs
        self.output_contract = {
            ContextDataType.PROMPT_EMBEDDINGS: dict
        }
    
    def configure(self, params: Dict[str, str]) -> None:
        """
        Configure from DSL parameters.
        
        Args:
            params: Configuration parameters
                - embeddings_path: Path to embeddings file
                - prompts: Comma-separated list of prompts (for onboard mode)
                - clip_model: Name/reference to clip model (for onboard mode)
        """
        if "embeddings_path" in params:
            self.embeddings_path = params["embeddings_path"]
            self.mode = "file"
            
        if "prompts" in params:
            self.prompts = [p.strip() for p in params["prompts"].split(",")]
            
        # clip_model would need to be injected externally
        logger.info(f"PromptEmbeddingSourceTask configured in {self.mode} mode")
    
    def run(self, context: Context) -> Context:
        """
        Load or generate prompt embeddings and inject into context.
        
        Args:
            context: Input context (may already have embeddings from previous run)
            
        Returns:
            Context with PROMPT_EMBEDDINGS populated
        """
        # Check if already in context (from previous iteration via cleanup task)
        if ContextDataType.PROMPT_EMBEDDINGS in context.data:
            # Check if source has updated
            if self._needs_update():
                logger.info(f"Prompt embeddings source updated, reloading...")
                prompt_data = self._load_or_generate()
                context.data[ContextDataType.PROMPT_EMBEDDINGS] = prompt_data
            else:
                # Use existing embeddings
                logger.debug("Using cached prompt embeddings from context")
                return context
        else:
            # First time - load or generate
            logger.info(f"Loading prompt embeddings ({self.mode} mode)...")
            prompt_data = self._load_or_generate()
            context.data[ContextDataType.PROMPT_EMBEDDINGS] = prompt_data
        
        return context
    
    def _needs_update(self) -> bool:
        """Check if source has been updated."""
        if self.mode == "file":
            if not os.path.exists(self.embeddings_path):
                return False
            current_mtime = os.path.getmtime(self.embeddings_path)
            if current_mtime != self._last_mtime:
                self._last_mtime = current_mtime
                return True
            return False
        else:
            # Onboard mode - prompts don't change at runtime in current design
            # Could add version checking here in future
            return False
    
    def _load_or_generate(self) -> Dict[str, Any]:
        """Load from file or generate embeddings."""
        if self.mode == "file":
            return self._load_from_file()
        else:
            return self._generate_embeddings()
    
    def _load_from_file(self) -> Dict[str, Any]:
        """Load pre-computed embeddings from JSON file."""
        try:
            with open(self.embeddings_path, 'r') as f:
                data = json.load(f)
            
            # Update tracking
            self._last_mtime = os.path.getmtime(self.embeddings_path)
            if "version" in data:
                self._version = data["version"]
            
            logger.info(f"Loaded {len(data.get('prompts', []))} prompt embeddings from {self.embeddings_path}")
            return data
            
        except FileNotFoundError:
            logger.error(f"Embeddings file not found: {self.embeddings_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in embeddings file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading embeddings file: {e}")
            raise
    
    def _generate_embeddings(self) -> Dict[str, Any]:
        """Generate embeddings using CLIP text encoder."""
        if not self.clip_model:
            raise RuntimeError("CLIP model not available for onboard embedding generation")
        
        try:
            # Get CLIP runtime backend
            runtime = self.clip_model._runtime_as_clip()
            
            # Encode prompts
            logger.info(f"Encoding {len(self.prompts)} prompts with CLIP text encoder...")
            embeddings_tensor = runtime.encode_text(self.prompts)
            
            # Convert to list format
            embeddings_list = embeddings_tensor.cpu().numpy().tolist()
            
            # Increment version
            self._version += 1
            
            prompt_data = {
                "prompts": self.prompts,
                "embeddings": embeddings_list,
                "version": self._version,
                "mode": "onboard"
            }
            
            logger.info(f"Generated {len(embeddings_list)} prompt embeddings (version {self._version})")
            return prompt_data
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def __str__(self) -> str:
        """String representation."""
        if self.mode == "file":
            return f"PromptEmbeddingSourceTask(file={self.embeddings_path})"
        else:
            return f"PromptEmbeddingSourceTask(onboard, {len(self.prompts)} prompts)"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()


if __name__ == "__main__":
    # Example usage
    print("\n--- PromptEmbeddingSourceTask Example ---\n")
    
    import tempfile
    
    # Create test embeddings file
    test_data = {
        "prompts": ["a person riding a horse", "a car on the street"],
        "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "version": 1
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name
    
    try:
        # Test 1: Load from file
        print("--- Test 1: File mode ---")
        task = PromptEmbeddingSourceTask(embeddings_path=temp_path)
        print(f"Task: {task}")
        
        ctx = Context()
        result = task.run(ctx)
        
        if ContextDataType.PROMPT_EMBEDDINGS in result.data:
            data = result.data[ContextDataType.PROMPT_EMBEDDINGS]
            print(f"Loaded {len(data['prompts'])} prompts:")
            for prompt in data['prompts']:
                print(f"  - {prompt}")
        
        # Test 2: Use cached (simulate next iteration)
        print("\n--- Test 2: Use cached from context ---")
        result2 = task.run(result)  # Pass result as input
        print("Using cached embeddings (no reload)")
        
        # Test 3: Simulate file update
        print("\n--- Test 3: File update ---")
        import time
        time.sleep(0.1)  # Ensure mtime changes
        test_data["version"] = 2
        test_data["prompts"].append("a bicycle")
        test_data["embeddings"].append([0.7, 0.8, 0.9])
        with open(temp_path, 'w') as f:
            json.dump(test_data, f)
        
        result3 = task.run(result2)
        if ContextDataType.PROMPT_EMBEDDINGS in result3.data:
            data = result3.data[ContextDataType.PROMPT_EMBEDDINGS]
            print(f"Reloaded {len(data['prompts'])} prompts (version {data['version']})")
        
        print("\n✓ PromptEmbeddingSourceTask file mode tests complete")
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Test 4: Onboard mode (mock CLIP model)
    print("\n--- Test 4: Onboard mode (mock) ---")
    
    class MockCLIPModel:
        def _runtime_as_clip(self):
            return self
        
        def encode_text(self, prompts):
            import numpy as np
            # Mock embeddings
            class MockTensor:
                def __init__(self, data):
                    self.data = data
                def cpu(self):
                    return self
                def numpy(self):
                    return self.data
                def tolist(self):
                    return self.data.tolist()
            
            embeddings = np.random.randn(len(prompts), 512).astype(np.float32)
            return MockTensor(embeddings)
    
    mock_clip = MockCLIPModel()
    prompts = ["test prompt 1", "test prompt 2"]
    
    task_onboard = PromptEmbeddingSourceTask(
        prompts=prompts,
        clip_model=mock_clip
    )
    print(f"Task: {task_onboard}")
    
    ctx2 = Context()
    result4 = task_onboard.run(ctx2)
    
    if ContextDataType.PROMPT_EMBEDDINGS in result4.data:
        data = result4.data[ContextDataType.PROMPT_EMBEDDINGS]
        print(f"Generated {len(data['embeddings'])} embeddings")
        print(f"Embedding shape: {len(data['embeddings'][0])} dimensions")
        print(f"Mode: {data['mode']}")
        print(f"Version: {data['version']}")
    
    print("\n✓ PromptEmbeddingSourceTask onboard mode tests complete")
