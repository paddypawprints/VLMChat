"""
Provides a class to build a semantic cost matrix for clustering.

This module defines:
- ISemanticCostProvider: An abstract interface for semantic cost providers.
- ClipSemanticProvider: A concrete implementation that uses a CLIPModel
  to pre-calculate costs based on prompt matching.
"""

import logging
import json
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from enum import Enum # For dummy class
import math # For batching

# --- Import from project ---
from .coco_categories import CocoCategory
from models.MobileClip.clip_model import CLIPModel, CLIPRuntimeBase

logger = logging.getLogger(__name__)

# --- Abstract Interface ---

class ISemanticCostProvider(ABC):
    """
    Abstract interface for a class that provides semantic costs
    between object categories.
    """
    @abstractmethod
    def start(self) -> None:
        """
        Initializes the provider, e.g., by building the cost matrix.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Cleans up resources.
        """
        pass

    @abstractmethod
    def get_pair_cost(self, category_a: str, category_b: str) -> float:
        """
        Gets the semantic cost using the N*N PAIR ("A and B") algorithm.
        """
        pass
        
    @abstractmethod
    def get_single_cost(self, category_a: str, category_b: str) -> float:
        """
        Gets the semantic cost using the N*K SINGLE (A vs P, B vs P) algorithm.
        """
        pass

# --- Concrete Implementation ---

class ClipSemanticProvider(ISemanticCostProvider):
    """
    An implementation of ISemanticCostProvider that uses a CLIPModel
    to pre-calculate semantic cost matrices.
    """
    
    def __init__(self, 
                 clip_model: CLIPModel, 
                 user_prompts: List[str],
                 embeddings_cache_path: str = "category_pair_embeddings.json",
                 batch_size: int = 5):
        """
        Initializes the provider.

        Args:
            clip_model: An initialized CLIPModel instance.
            user_prompts: The list of user-defined text prompts to match against.
            embeddings_cache_path: Path to the JSON file for storing/reading
                                   category pair embeddings.
            batch_size: Number of text pairs to process in one batch.
        """
        self._clip_model = clip_model
        self._user_prompts = user_prompts
        self._embeddings_cache_path = embeddings_cache_path
        self._batch_size = batch_size
        self._is_ready = False
        
        # Get all category labels from the Enum
        self._categories = [member.label for member in CocoCategory]
        
        # --- Data for Algorithm 1 (Pair) ---
        self._pair_cost_matrix: Dict[str, Dict[str, float]] = {}
        
        # --- Data for Algorithm 2 (Single) ---
        self._single_cost_matrix: Dict[str, Dict[str, float]] = {}
        self._similarity_matrix: Dict[str, Dict[str, float]] = {} # cat -> prompt -> sim


    def start(self) -> None:
        """
        Builds the semantic cost matrices for both algorithms.
        """
        #if not self._clip_model.readiness():
        #    logger.error("ClipSemanticProvider: CLIPModel is not ready.")
        #    return

        if not self._categories:
            logger.error("ClipSemanticProvider: No categories loaded from CocoCategory enum.")
            return
            
        if not self._user_prompts:
            logger.warning("ClipSemanticProvider: No user prompts provided. Semantic costs will be 1.0")
            
        logger.info(f"Building semantic cost matrices for {len(self._categories)} categories...")
        
        try:
            runtime = self._clip_model._runtime_as_clip() # Access internal method
            
            # --- Pre-calculate Prompt Embeddings (K) ---
            prompt_features = runtime.encode_text(self._user_prompts)
            
            # --- Algorithm 1: Build Pair Cost Matrix ---
            self._build_pair_cost_matrix(runtime, prompt_features)
            
            # --- Algorithm 2: Build Single Cost Matrix ---
            self._build_single_cost_matrix(runtime, prompt_features)
                        
            self._is_ready = True
            logger.info("Semantic cost matrices built successfully.")
            
        except Exception as e:
            logger.error(f"Failed to build semantic cost matrix: {e}", exc_info=True)
            self._is_ready = False

    def _build_pair_cost_matrix(self, runtime: "CLIPRuntimeBase", prompt_features: torch.Tensor):
        """Builds the matrix for the N*N "A and B" algorithm."""
        logger.info("Building Pair Cost (N*N) Matrix...")
        pair_embeddings = self._get_or_create_pair_embeddings(runtime)
        
        pair_texts = list(pair_embeddings.keys())
        pair_features_tensor = torch.tensor(
            [pair_embeddings[text] for text in pair_texts]
        )

        # Ensure tensors are on the same device (e.g., CPU for this op)
        prompt_features = prompt_features.cpu()
        pair_features_tensor = pair_features_tensor.cpu()

        with torch.no_grad():
            similarity_matrix = pair_features_tensor @ prompt_features.T
            max_sims, _ = torch.max(similarity_matrix, dim=1)

        self._pair_cost_matrix = {label: {} for label in self._categories}
        
        for idx, pair_text in enumerate(pair_texts):
            # Find which categories this text ("cat_a and cat_b") corresponds to
            found = False
            for i, cat_a in enumerate(self._categories):
                for j in range(i, len(self._categories)):
                    cat_b = self._categories[j]
                    current_pair_text = f"{cat_a} and {cat_b}" if cat_a != cat_b else cat_a
                    if current_pair_text == pair_text:
                        cost = 1.0 - max(0.0, max_sims[idx].item())
                        self._pair_cost_matrix[cat_a][cat_b] = cost
                        self._pair_cost_matrix[cat_b][cat_a] = cost
                        found = True
                        break
                if found:
                    break
        logger.info("Pair Cost Matrix built.")

    def _build_single_cost_matrix(self, runtime: "CLIPRuntimeBase", prompt_features: torch.Tensor):
        """Builds the matrix for the N*K "A vs P" * "B vs P" algorithm."""
        logger.info("Building Single Cost (N*K) Matrix...")
        
        # 1. Get N category embeddings
        category_features = runtime.encode_text(self._categories)
        
        # 2. Build N x K similarity matrix (Cat vs Prompt)
        category_features = category_features.cpu()
        prompt_features = prompt_features.cpu()
        
        with torch.no_grad():
            # sim_matrix[i, j] = similarity(category_i, prompt_j)
            sim_matrix = category_features @ prompt_features.T
            # Clamp similarity values (0 to 1)
            sim_matrix = torch.clamp(sim_matrix, 0, 1)

        # 3. Store this N x K matrix for fast lookup
        self._similarity_matrix = {cat: {} for cat in self._categories}
        for i, cat in enumerate(self._categories):
            for j, prompt in enumerate(self._user_prompts):
                self._similarity_matrix[cat][prompt] = sim_matrix[i, j].item()
        
        # 4. Pre-calculate the final N*N cost matrix
        self._single_cost_matrix = {label: {} for label in self._categories}
        
        for i, cat_a in enumerate(self._categories):
            for j in range(i, len(self._categories)):
                cat_b = self._categories[j]
                
                best_cost = 1.0
                # Find the prompt that gives the best (lowest) cost
                for k, prompt in enumerate(self._user_prompts):
                    sim_a = sim_matrix[i, k].item() # Sim(A, P_k)
                    sim_b = sim_matrix[j, k].item() # Sim(B, P_k)
                    
                    #combined_sim = sim_a * sim_b # Multiply probabilities
                    combined_sim = min(sim_a, sim_b) # Multiply probabilities
                    cost = 1.0 - combined_sim
                    
                    best_cost = min(best_cost, cost)
                
                self._single_cost_matrix[cat_a][cat_b] = best_cost
                self._single_cost_matrix[cat_b][cat_a] = best_cost
        logger.info("Single Cost Matrix built.")


    def _get_or_create_pair_embeddings(self, runtime: "CLIPRuntimeBase") -> Dict[str, List[float]]:
        """
        Loads category pair embeddings from the cache file, or creates
        the file if it doesn't exist.
        """
        try:
            with open(self._embeddings_cache_path, 'r') as f:
                embeddings = json.load(f)
                logger.info(f"Loaded category pair embeddings from {self._embeddings_cache_path}")
                return embeddings
        except FileNotFoundError:
            logger.warning(f"Embedding cache not found. Creating new file at {self._embeddings_cache_path}")
            return self.create_embeddings_cache_file(runtime)
        except Exception as e:
            logger.error(f"Error loading embedding cache: {e}. Rebuilding.", exc_info=True)
            return self.create_embeddings_cache_file(runtime)

    def create_embeddings_cache_file(self, runtime: Optional["CLIPRuntimeBase"] = None) -> Dict[str, List[float]]:
        """
        Generates embeddings for all category pairs and saves them to a file.
        
        This is a public method so it can be called explicitly as a build step.
        """
        if runtime is None:
            #if not self._clip_model.readiness():
            #    raise RuntimeError("CLIPModel is not ready. Cannot generate embeddings.")
            runtime = self._clip_model._runtime_as_clip()
            
        logger.info(f"Generating new embeddings cache file at {self._embeddings_cache_path}...")
        
        # 1. Generate all pair texts
        pair_texts: List[str] = []
        for i, cat_a in enumerate(self._categories):
            for j in range(i, len(self._categories)):
                cat_b = self._categories[j]
                pair_text = f"{cat_a} and {cat_b}" if cat_a != cat_b else cat_a
                pair_texts.append(pair_text)
        
        # 2. Get embeddings from the CLIP model in batches
        embeddings_dict = {}
        
        num_batches = math.ceil(len(pair_texts) / self._batch_size)
        
        for i in range(num_batches):
            batch_start = i * self._batch_size
            batch_end = (i + 1) * self._batch_size
            batch_texts = pair_texts[batch_start:batch_end]
            
            if not batch_texts:
                continue
                
            logger.info(f"Processing batch {i+1}/{num_batches} ({len(batch_texts)} pairs)...")
            logger.info(f"  Batch {i+1} texts: {batch_texts}")
            
            try:
                embeddings_tensor = runtime.encode_text(batch_texts)
                
                # 3. Convert to a JSON-serializable format (dict of lists)
                for text, embedding in zip(batch_texts, embeddings_tensor):
                    embeddings_dict[text] = embedding.cpu().numpy().tolist()
                    
            except Exception as e:
                logger.error(f"Failed to process batch {i+1}: {e}", exc_info=True)
        
        # 4. Save to file
        try:
            with open(self._embeddings_cache_path, 'w') as f:
                json.dump(embeddings_dict, f, indent=2)
            logger.info(f"Successfully saved {len(embeddings_dict)} embeddings to {self._embeddings_cache_path}")
            return embeddings_dict
        except Exception as e:
            logger.error(f"Failed to save embeddings file: {e}", exc_info=True)
            return {}

    def stop(self) -> None:
        """Clears the cost matrix."""
        self._pair_cost_matrix.clear()
        self._single_cost_matrix.clear()
        self._similarity_matrix.clear()
        self._is_ready = False
        logger.info("ClipSemanticProvider stopped.")
        
    def get_pair_cost(self, category_a: str, category_b: str) -> float:
        """
        Gets the pre-calculated cost from the PAIR (N*N) matrix.
        """
        if not self._is_ready: return 1.0
        try:
            return self._pair_cost_matrix[category_a][category_b]
        except KeyError:
            return 1.0
            
    def get_single_cost(self, category_a: str, category_b: str) -> float:
        """
        Gets the pre-calculated cost from the SINGLE (N*K) matrix.
        """
        if not self._is_ready: return 1.0
        try:
            return self._single_cost_matrix[category_a][category_b]
        except KeyError:
            return 1.0

# --- Example Usage ---
if __name__ == "__main__":
    
    # --- Imports for __main__ ---
    import sys
    import os
    import traceback
    # Add parent directory to path to find 'models' and 'utils'
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    try:
        from models.MobileClip.clip_model import CLIPModel
        from metrics.metrics_collector import null_collector
        from .coco_categories import CocoCategory
    except ImportError as e:
        print(f"Error: Failed to import dependencies for __main__: {e}")
        print("Please run this script as a module from the project's 'src' directory.")
        print("Example: python -m semantic_provider")
        sys.exit(1)

    # --- End Imports ---

    print("--- ClipSemanticProvider __main__ Example ---")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    class MockConfig:
        def __init__(self):
            self.model = self
            self.clip_model_name = "MobileCLIP2-S0"
            self.clip_pretrained_path = "/Users/patrick/mobileclip2_s0.pt"
            self.clip_model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
    
    mock_config = MockConfig()

    TEST_CATEGORIES = [
        CocoCategory.PERSON.label,
        CocoCategory.HORSE.label,
        CocoCategory.CAR.label,
        CocoCategory.BICYCLE.label
    ]
    
    TEST_PROMPTS = [
        "a person riding a horse",
        "a person on a bicycle",
        "a car"
    ]
    
    TEST_CACHE_FILE = "test_embeddings_cache.json"

    try:
        print("Initializing CLIPModel...")
        clip_model = CLIPModel(config=mock_config, collector=null_collector()) # type: ignore
        
        #if not clip_model.readiness():
        #    raise RuntimeError("Failed to initialize CLIPModel. Check model paths.")

        print("Initializing ClipSemanticProvider...")
        provider = ClipSemanticProvider(
            clip_model=clip_model,
            user_prompts=TEST_PROMPTS,
            embeddings_cache_path=TEST_CACHE_FILE,
            batch_size=5
        )
        
        print("\nCalling create_embeddings_cache_file()...")
        embeddings = provider.create_embeddings_cache_file()
        
        print("\n--- Generated Embeddings (truncated) ---")
        if embeddings:
            for key, value in list(embeddings.items())[:5]:
                print(f"'{key}': {value[:5]}...")
        
        print("\nCalling start() to build cost matrices...")
        provider.start()
        
        if provider._is_ready:
            print("\n--- Calculated Costs (Pair N*N Algorithm) ---")
            print(f"Cost (person, horse): {provider.get_pair_cost('person', 'horse'):.4f}")
            print(f"Cost (person, car): {provider.get_pair_cost('person', 'car'):.4f}")
            print(f"Cost (person, toothbrush): {provider.get_pair_cost('person', 'toothbrush'):.4f}")
            print(f"Cost (person, toothbrush): {provider.get_pair_cost('horse', 'toothbrush'):.4f}")
            
            print("\n--- Calculated Costs (Single N*K Algorithm) ---")
            print(f"Cost (person, horse): {provider.get_single_cost('person', 'horse'):.4f}")
            print(f"Cost (person, car): {provider.get_single_cost('person', 'car'):.4f}")
            print(f"Cost (person, toothbrush): {provider.get_single_cost('person', 'toothbrush'):.4f}")    
            print(f"Cost (horse, toothbrush): {provider.get_single_cost('horse', 'toothbrush'):.4f}")    


    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure 'open_clip' is installed and all model files (e.g., './mobileclip2_s0.pt') are present.")
        traceback.print_exc()
    finally:
        if os.path.exists(TEST_CACHE_FILE):
            os.remove(TEST_CACHE_FILE)
            print(f"\nCleaned up {TEST_CACHE_FILE}")