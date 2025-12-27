"""
⚠️ DEPRECATED - DO NOT USE FOR NEW CODE ⚠️

This module is part of the LEGACY object_detector package which is being phased out.

FOR NEW CODE, USE: vlmchat.pipeline.tasks.clusterer.ClustererTask
  - ClustererTask is the pipeline-native version
  - Works with pipeline Detection objects (vlmchat.pipeline.detection)
  - Integrates with ClipSemanticService (vlmchat.pipeline.services.semantic)
  - Follows pipeline task architecture

This legacy ObjectClusterer still exists for backward compatibility but should
NOT be used in new pipelines or tests. It uses deprecated Detection objects
from vlmchat.object_detector.detection_base.

---

ObjectDetector pipeline stage for clustering detections.

This file contains:
1.  ClusterAuditLog: A class to store and format audit trails.
2.  ObjectClusterer: The main clustering class with auditing logic.
"""

# --- Base Imports ---
from typing import List, Optional, Tuple, Set, Dict, Any
from PIL.Image import Image
from PIL import Image as PILImage
import numpy as np
import math
import time # For __main__
import requests # For __main__
from io import BytesIO # For __main__
import logging
import torch

logger = logging.getLogger(__name__)

# --- Base Detector Classes (Imported) ---
from .detection_base import ObjectDetector, Detection
from .coco_categories import CocoCategory

# Optional import for legacy semantic provider
try:
    from .semantic_provider import ISemanticCostProvider
except ImportError:
    ISemanticCostProvider = None  # type: ignore[misc,assignment]

# --- Helper Classes and Functions for Clustering ---

def _box_area(box: Tuple[int, int, int, int]) -> int:
    """Calculates the area of a bounding box."""
    if not box: return 0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return w * h

def _union_box(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Calculates the union (bounding box) of two boxes."""
    return (
        min(b1[0], b2[0]),
        min(b1[1], b2[1]),
        max(b1[2], b2[2]),
        max(b1[3], b2[3])
    )

class _ClusterNode:
    """Internal helper class to track cluster state during aggregation."""
    def __init__(self, 
                 detection: Detection, 
                 pixel_area: int,
                 categories: Set[CocoCategory], # Use Enum
                 category_areas: Dict[CocoCategory, List[int]], # Use Enum
                 creation_cost: float = 0.0): # <-- ADDED
        
        self.detection = detection
        self.pixel_area = pixel_area
        # Handle both legacy and pipeline Detection objects
        if hasattr(detection, 'bbox'):
            self.box_area = _box_area(detection.bbox)
        else:
            self.box_area = _box_area(detection.box)
        self.categories = categories
        self.category_areas = category_areas
        self.creation_cost = creation_cost # <-- ADDED

# --- End Helpers ---

# --- Audit Log Class ---

class ClusterAuditLog:
    """
    Stores and formats the audit trail for a single clustering run.
    """
    def __init__(self):
        self.comparisons: List[Dict[str, Any]] = []
        self.merges: List[Dict[str, Any]] = []
        self.final_clusters: List[Detection] = []
        self.final_count = 0

    def log_comparison(self, a: _ClusterNode, b: _ClusterNode, similarities: Dict[str, float], total_similarity: float):
        """Logs a single comparison between two nodes."""
        self.comparisons.append({
            "a_id": a.detection.id,
            "b_id": b.detection.id,
            "similarities": similarities,
            "total_similarity": total_similarity
        })

    def log_merge(self, a: _ClusterNode, b: _ClusterNode, new_node: _ClusterNode, similarity: float, component_similarities: Optional[Dict[str, float]] = None, weights: Optional[Dict[str, float]] = None):
        """Logs a successful merge operation."""
        self.merges.append({
            "merged_a_id": a.detection.id,
            "merged_b_id": b.detection.id,
            "new_cluster_id": new_node.detection.id,
            "similarity": similarity,
            "components": component_similarities or {},
            "weights": weights or {}
        })
        
    def log_final_clusters(self, clusters: List[_ClusterNode]):
        """Logs the final state of the clusters."""
        self.final_clusters = [n.detection for n in clusters]
        self.final_count = len(clusters)

    def __str__(self) -> str:
        """Formats the entire audit log as a human-readable string."""
        log_str = "--- Cluster Audit Log ---\n"
        
        if not self.comparisons and not self.merges:
            return log_str + "No activity logged."

        log_str += "\n=== Merge Operations ===\n"
        if not self.merges:
            log_str += "No merges occurred.\n"
        for i, merge in enumerate(self.merges):
            # Compact format: Merge N: A+B->C (similarity)
            log_str += (
                f"Merge {i+1}: {merge['merged_a_id']}+{merge['merged_b_id']}->{merge['new_cluster_id']} "
                f"(sim={merge['similarity']:.4f})\n"
            )
            
            # Display component similarities if available
            if merge.get('components'):
                components = merge['components']
                weights = merge.get('weights', {})
                
                # Proximity
                if 'prox' in components:
                    prox_val = components['prox']
                    weighted_val = prox_val * weights.get('prox', 1.0) if 'prox' in weights else prox_val
                    log_str += f"         proximity={prox_val:.4f} (weighted={weighted_val:.4f})\n"
                
                # Size
                if 'size' in components:
                    size_val = components['size']
                    weighted_val = size_val * weights.get('size', 1.0) if 'size' in weights else size_val
                    log_str += f"         size={size_val:.4f} (weighted={weighted_val:.4f})\n"
                
                # Semantic (CLIP or legacy pair)
                if 'sem_clip' in components:
                    sem_val = components['sem_clip']
                    sem_weight = next(iter(weights.values())) if weights else 1.0
                    weighted_val = sem_val * sem_weight
                    log_str += f"         semantic_clip={sem_val:.4f} (weighted={weighted_val:.4f})\n"
                elif 'sem_pair' in components:
                    sem_val = components['sem_pair']
                    weighted_val = sem_val * weights.get('sem_pair', 1.0) if 'sem_pair' in weights else sem_val
                    log_str += f"         semantic_pair={sem_val:.4f} (weighted={weighted_val:.4f})\n"

        log_str += "\n=== Final Clusters ===\n"
        log_str += f"Total: {self.final_count}\n"
        for det in self.final_clusters:
            log_str += f"  - {str(det)}\n" # Use Detection's __str__

        return log_str

# --- End Audit Log Class ---


# --- Object Clusterer Implementation ---

class ObjectClusterer(ObjectDetector):
    """
    An ObjectDetector that performs hierarchical clustering on its
    source's detections.
    """
    
    def __init__(self, *,
                 source: ObjectDetector,
                 semantic_provider: Optional[ISemanticCostProvider] = None,
                 semantic_service = None,  # ClipSemanticService instance
                 prompts: Optional[List[str]] = None,
                 max_clusters: int = 10,
                 merge_threshold: float = 0.5,  # Similarity threshold (higher=more selective)
                 proximity_weight: float = 1.0,
                 size_weight: float = 0.5,
                 semantic_weights: Dict[str, float] = {"pair": 1.0},
                 prob_gain: float = 10.0
                 ):
        """
        Initializes the clusterer.
        
        Args:
            source: The detector providing the base detections (e.g., YOLO).
            semantic_provider: (Legacy) A pre-built semantic provider instance.
            semantic_service: ClipSemanticService instance for probability-based semantic matching.
            prompts: List of text prompts for semantic matching (e.g., "person riding a horse").
            max_clusters: (K) The number of clusters to return after filtering.
            merge_threshold: Stop merging if best similarity is below this (range [0,1]).
            proximity_weight: Weight for the proximity similarity.
            size_weight: Weight for the size similarity.
            semantic_weights: Dict of weights for semantic algorithms.
            prob_gain: Sigmoid gain parameter for probability conversion (default 10.0).
        """
        super().__init__(source)
        self.semantic_provider = semantic_provider  # Keep for backward compatibility
        self.semantic_service = semantic_service
        self.prompts = prompts or []
        self.max_clusters = max_clusters
        self.merge_threshold = merge_threshold
        self.prob_gain = prob_gain
        # Store weights
        self.w_prox = proximity_weight
        self.w_size = size_weight
        self.semantic_weights = semantic_weights
        
        # Cache for semantic query results
        self._semantic_cache: Optional[Dict] = None
        
        self._ready = False
        self._audit_enabled = False
        self._last_audit_log: ClusterAuditLog = ClusterAuditLog()
        logger.debug("ObjectClusterer: Initialized.")

    def start(self, audit: bool = False) -> None:
        """
        Starts the clusterer and its source.
        
        Args:
            audit: If True, enables detailed logging for the next run.
        """
        super().start(audit) # Pass audit flag to source
        if self.semantic_provider:
            self.semantic_provider.start()
        self._audit_enabled = audit
        
        self._ready = True
        logger.debug(f"ObjectClusterer: Started (Audit Enabled: {self._audit_enabled}).")

    def set_text_prompts(self, prompts: List[str]) -> None:
        """
        Updates the TEXT prompts used for semantic similarity calculations.
        This is called by the pipeline when TEXT context changes.
        
        Args:
            prompts: List of text prompts for semantic matching (e.g., "person riding a horse")
        """
        self.prompts = prompts
        
        # Invalidate semantic cache when prompts change
        self._semantic_cache = None
        
        # Update legacy semantic provider if present
        if self.semantic_provider and hasattr(self.semantic_provider, 'update_filter_prompts'):
            self.semantic_provider.update_filter_prompts(prompts)  # type: ignore[attr-defined]
        
        logger.debug(f"ObjectClusterer: Updated semantic prompts to {len(prompts)} items")

    def stop(self) -> None:
        super().stop()
        if self.semantic_provider:
            self.semantic_provider.stop()
        self._ready = False
        print("ObjectClusterer: Stopped.")

    def get_last_audit_log(self) -> ClusterAuditLog:
        """Returns the audit log from the most recent 'detect' call."""
        return self._last_audit_log

    def readiness(self) -> bool:
        return self._ready and super().readiness()

    def _detect_internal(self, image: Image, detections: List[Detection]) -> List[Detection]:
        """
        Runs the hierarchical clustering algorithm.
        Works with both legacy and pipeline Detection objects.
        """
        self._last_audit_log = ClusterAuditLog()

        if not detections:
            return []
        
        # 1. Initialize nodes from base detections
        nodes: List[_ClusterNode] = []
        for det in detections:
            # Handle both legacy and pipeline Detection objects
            if hasattr(det, 'category'):  # Pipeline Detection
                cat_label = det.category.label
                bbox = det.bbox
                conf = det.confidence
            else:  # Legacy Detection
                cat_label = det.object_category
                bbox = det.box
                conf = det.conf
            
            cat_enum = CocoCategory.from_string(cat_label)
            if not cat_enum:
                continue
                
            area = _box_area(bbox)
            if area > 0:
                nodes.append(_ClusterNode(
                    detection=det,
                    pixel_area=area,
                    categories={cat_enum},
                    category_areas={cat_enum: [area]},
                    creation_cost=0.0 # Base cost is 0
                ))

        # 2. Run aggregation loop
        while True:
            if len(nodes) <= 1: # Stop if only one cluster remains
                break

            max_similarity, best_pair, best_components = self._find_best_merge_pair(nodes, self._last_audit_log)
            
            # Stop if best similarity is below threshold
            if max_similarity < self.merge_threshold or best_pair is None:
                break
                
            idx_a, idx_b = best_pair
            node_a = nodes[idx_a]
            node_b = nodes[idx_b]
            
            # Pass the similarity to the merge function
            new_node = self._merge_nodes(node_a, node_b, max_similarity)
            
            if self._audit_enabled:
                # Build weights dict for audit
                weights_dict = {
                    'prox': self.w_prox,
                    'size': self.w_size
                }
                # Add semantic weights
                for name, weight in self.semantic_weights.items():
                    if name == "pair":
                        weights_dict['sem_pair'] = weight
                    elif name == "single":
                        weights_dict['sem_single'] = weight
                
                self._last_audit_log.log_merge(node_a, node_b, new_node, max_similarity, best_components, weights_dict)
            
            nodes.pop(max(idx_a, idx_b))
            nodes.pop(min(idx_a, idx_b))
            nodes.append(new_node)

        # 3. Final filtering
        if len(nodes) > self.max_clusters:
            # Sort by creation_similarity (highest similarity is best)
            nodes.sort(key=lambda n: n.creation_cost, reverse=True)
            nodes = nodes[:self.max_clusters] # Keep the K best
            
        # 4. Log final state and return
        if self._audit_enabled:
            self._last_audit_log.log_final_clusters(nodes)
            
        return [n.detection for n in nodes]

    def _calculate_merge_similarity(self, a: _ClusterNode, b: _ClusterNode) -> Tuple[float, Dict[str, float]]:
        """
        Calculates the weighted average similarity of merging two nodes.
        Returns the total similarity AND a dictionary of component similarities for auditing.
        All similarities are in [0, 1] where higher = better match.
        """
        
        # Extract boxes for both detection types
        a_box = a.detection.bbox if hasattr(a.detection, 'bbox') else a.detection.box
        b_box = b.detection.bbox if hasattr(b.detection, 'bbox') else b.detection.box
        
        # --- 1. Proximity Similarity (Packing Efficiency) ---
        merged_box = _union_box(a_box, b_box)
        merged_box_area = _box_area(merged_box)
        total_pixel_area = a.pixel_area + b.pixel_area
        
        if merged_box_area == 0:
            sim_prox = 0.0
        else:
            # Packing efficiency: how much of merged box is actual objects
            sim_prox = total_pixel_area / merged_box_area

        # --- 2. Size Similarity (Scale Compatibility) ---
        sim_size = 1.0  # Default: perfect match if no shared categories
        shared_categories = a.categories.intersection(b.categories)
        
        if shared_categories:
            worst_size_sim = 1.0
            for cat_enum in shared_categories:
                areas_a = a.category_areas.get(cat_enum, [])
                areas_b = b.category_areas.get(cat_enum, [])
                if not areas_a or not areas_b: continue
                median_a = np.median(areas_a)
                median_b = np.median(areas_b)
                max_median = max(median_a, median_b, 1)
                size_diff_ratio = abs(median_a - median_b) / max_median
                # Convert difference to similarity
                cat_size_sim = 1.0 - size_diff_ratio
                worst_size_sim = min(worst_size_sim, cat_size_sim)
            sim_size = worst_size_sim
        
        # --- 3. Semantic Similarities ---
        
        similarity_sum = 0.0
        total_weight = 0.0
        audit_sims: Dict[str, float] = {}  # For logging

        similarity_sum += self.w_prox * sim_prox
        total_weight += self.w_prox
        audit_sims['prox'] = sim_prox

        similarity_sum += self.w_size * sim_size
        total_weight += self.w_size
        audit_sims['size'] = sim_size

        # Compute semantic similarity using ClipSemanticService if available
        if self.semantic_service and self.prompts:
            # Query semantic service once and cache results
            if self._semantic_cache is None:
                self._semantic_cache = self.semantic_service.query_all_pairs(
                    self.prompts,
                    as_probabilities=True,
                    prob_center=None,  # Auto-median
                    prob_gain=self.prob_gain
                )
            
            # Get category pair similarity from cache
            sim_sem = self._get_semantic_similarity_from_cache(a, b)
            
            # Use first semantic weight (typically "pair")
            semantic_weight = next(iter(self.semantic_weights.values())) if self.semantic_weights else 0.0
            if semantic_weight > 0:
                similarity_sum += semantic_weight * sim_sem
                total_weight += semantic_weight
                audit_sims['sem_clip'] = sim_sem
        
        # Fall back to legacy semantic provider if no service
        elif self.semantic_provider:
            for name, weight in self.semantic_weights.items():
                if weight == 0: continue
                
                sim_sem = 0.0
                if name == "pair":
                    avg_sim = self._get_avg_semantic_similarity(a, b, self.semantic_provider.get_pair_cost)
                    sim_sem = avg_sim
                    audit_sims['sem_pair'] = sim_sem
                elif name == "single":
                    avg_sim = self._get_avg_semantic_similarity(a, b, self.semantic_provider.get_single_cost)
                    sim_sem = avg_sim
                    audit_sims['sem_single'] = sim_sem
                
                similarity_sum += weight * sim_sem
                total_weight += weight

        # --- 4. Final Weighted Average ---
        if total_weight == 0: 
            return 0.0, audit_sims
        
        total_similarity = similarity_sum / total_weight
        return total_similarity, audit_sims

    def _get_semantic_similarity_from_cache(self, a: _ClusterNode, b: _ClusterNode) -> float:
        """
        Gets the average semantic similarity between two nodes using cached ClipSemanticService results.
        
        For each category pair combination between nodes a and b, looks up the probability
        from the cached query_all_pairs results and averages them.
        
        Returns:
            Average probability in [0, 1] where higher = more semantically similar.
        """
        if not self._semantic_cache:
            return 0.0
        
        prob_sum = 0.0
        pair_count = 0
        
        for cat_a_enum in a.categories:
            for cat_b_enum in b.categories:
                # Look up this category pair in the cache
                # query_all_pairs returns dict[prompt] -> [(cat_a, cat_b, score), ...]
                # We need to find the best match for this category pair across all prompts
                
                max_prob = 0.0
                for prompt in self.prompts:
                    if prompt not in self._semantic_cache:
                        continue
                    
                    # Find matching category pair in results
                    for result_cat_a, result_cat_b, prob in self._semantic_cache[prompt]:
                        if ((result_cat_a == cat_a_enum.label and result_cat_b == cat_b_enum.label) or
                            (result_cat_a == cat_b_enum.label and result_cat_b == cat_a_enum.label)):
                            max_prob = max(max_prob, prob)
                            break
                
                prob_sum += max_prob
                pair_count += 1
        
        return (prob_sum / pair_count) if pair_count > 0 else 0.0

    def _get_avg_semantic_similarity(self, a: _ClusterNode, b: _ClusterNode, cost_func) -> float:
        """Helper to get the average semantic similarity between two nodes.
        Note: cost_func still returns cost (distance), so we invert it to similarity.
        """
        cost_sum = 0.0
        pair_count = 0
        
        for cat_a_enum in a.categories:
            for cat_b_enum in b.categories:
                cost_sum += cost_func(cat_a_enum.label, cat_b_enum.label)
                pair_count += 1
        
        avg_cost = (cost_sum / pair_count) if pair_count > 0 else 1.0
        # Convert cost (distance) to similarity: cost in [0, inf), we map to [0, 1]
        # Using inverse: sim = 1 / (1 + cost)
        return 1.0 / (1.0 + avg_cost)


    def _find_best_merge_pair(self, nodes: List[_ClusterNode], audit_log: ClusterAuditLog) -> Tuple[float, Optional[Tuple[int, int]], Dict[str, float]]:
        """Finds the pair of nodes with the highest merge similarity."""
        max_similarity = -1.0
        best_pair = None
        best_components = {}
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                total_similarity, component_sims = self._calculate_merge_similarity(nodes[i], nodes[j])
                
                if self._audit_enabled:
                    audit_log.log_comparison(nodes[i], nodes[j], component_sims, total_similarity)
                
                if total_similarity > max_similarity:
                    max_similarity = total_similarity
                    best_pair = (i, j)
                    best_components = component_sims
                    
        return max_similarity, best_pair, best_components

    def _merge_nodes(self, a: _ClusterNode, b: _ClusterNode, merge_similarity: float) -> _ClusterNode:
        """Merges two cluster nodes into a new one."""
        
        # Extract attributes for both detection types
        a_box = a.detection.bbox if hasattr(a.detection, 'bbox') else a.detection.box
        b_box = b.detection.bbox if hasattr(b.detection, 'bbox') else b.detection.box
        
        a_label = a.detection.category.label if hasattr(a.detection, 'category') else a.detection.object_category
        b_label = b.detection.category.label if hasattr(b.detection, 'category') else b.detection.object_category
        
        a_conf = a.detection.confidence if hasattr(a.detection, 'confidence') else a.detection.conf
        b_conf = b.detection.confidence if hasattr(b.detection, 'confidence') else b.detection.conf
        
        new_box = _union_box(a_box, b_box)
        new_pixel_area = a.pixel_area + b.pixel_area
        new_categories = a.categories.union(b.categories)
        
        new_category_areas = a.category_areas.copy()
        for cat, areas in b.category_areas.items():
            new_category_areas.setdefault(cat, []).extend(areas)
        
        new_label = a_label if a.pixel_area > b.pixel_area else b_label
        new_conf = max(a_conf, b_conf)
        
        new_detection = Detection(
            box=new_box,
            object_category=new_label,
            conf=new_conf
        )
        new_detection.children = [a.detection, b.detection]
        
        return _ClusterNode(
            detection=new_detection,
            pixel_area=new_pixel_area,
            categories=new_categories,
            category_areas=new_category_areas,
            creation_cost=merge_similarity # Stores similarity score (higher=better)
        )

    # Moved clustering logic below instead of duplicate _detect_internal


# --- Example Usage ---
if __name__ == "__main__":
    
    # --- Imports for __main__ ---
    try:
        from .yolo_detector import YoloV8Detector  # type: ignore[import-not-found]
        from .detection_viewer import DetectionViewer  # type: ignore[import-not-found]
        from .image_viewer import ImageViewer  # type: ignore[import-not-found]
        from ..models.MobileClip.clip_text_model import ClipTextModel  # type: ignore[import-not-found]
        from .semantic_provider import ClipSemanticProvider  # type: ignore[import-not-found]
        from ..utils.config import load_config  # type: ignore[attr-defined]
        from ..metrics.metrics_collector import null_collector  # type: ignore[import-not-found]
    except ImportError as e:
        print(f"\n--- ERROR: Cannot import dependencies for __main__ ---")
        print(f"Failed on: {e}")
        print("Please ensure all modules are in your PYTHONPATH.")
        YoloV8Detector = None
    
    # --- Mock load_image_from_url ---
    def load_image_from_url(url: str) -> Image:
        print(f"Loading image from URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        pil_image = PILImage.open(BytesIO(response.content))
        print(f"Loaded image successfully (Size: {pil_image.size})")
        return pil_image

    if YoloV8Detector:
        print("\n--- Object Clusterer Pipeline Example ---")
        logging.basicConfig(level=logging.INFO) # Enable logging
        
        CONFIG_PATH = "path/to/your/config.ini" 
        IMAGE_PATH = "https://images.pdimagearchive.org/collections/berg-and-hoeg/32856644182_d379f3512b_o.jpg?width=1140&height=800"
        
        try:
            pil_image = load_image_from_url(IMAGE_PATH)
            
            class MockConfig:
                def __init__(self):
                    self.model = self
                    self.clip_model_name = "MobileCLIP2-S0"
                    self.clip_pretrained_path = "./mobileclip2_s0.pt"
                    self.clip_model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
            config = MockConfig()

            print("Initializing models...")
            text_model = ClipTextModel(config=config, collector=null_collector())  # type: ignore[possibly-unbound]
            yolo_detector = YoloV8Detector(model_name='yolov8n.pt')  # type: ignore[possibly-unbound]
            
            USER_PROMPTS = [
                "a person on a horse",
                "a person with a bicycle",
                "a group of people"
            ]
            
            semantic_provider = ClipSemanticProvider(  # type: ignore[possibly-unbound]
                text_model=text_model,
                user_prompts=USER_PROMPTS
            )
            
            print("\nCreating pipeline: YOLO -> Clusterer -> Viewer")
            viewer = ImageViewer(window_name="Object Clusterer Pipeline")  # type: ignore[possibly-unbound]
            
            detector1 = yolo_detector
            detector2 = ObjectClusterer(
                source=detector1,
                semantic_provider=semantic_provider,
                max_clusters=7,
                merge_threshold=1.4, # Allow 'good' merges
                proximity_weight=1.0,
                size_weight=0.5,
                semantic_weights={"pair": 1.0, "single": 0.0} # Use 'pair' only
            )
            detector3 = DetectionViewer(source=detector2, viewer=viewer)  # type: ignore[possibly-unbound]
            
            try:
                # --- Enable auditing ---
                detector3.start(audit=True)  # type: ignore[call-arg]
                
                if detector3.readiness():
                    print("\nRunning detection pipeline...")
                    detections = detector3.detect(pil_image)
                    
                    print(f"\n--- Pipeline Finished ---")
                    print(f"Final cluster count: {len(detections)}")
                    
                    # --- Print the audit log ---
                    print("\n")
                    print(detector2.get_last_audit_log())
                    
                    print("\nDisplaying result for 5 seconds...")
                    start_time = time.time()
                    while time.time() - start_time < 5 and viewer.is_visible():
                        viewer.show(wait_ms=30)
                else:
                    print("Pipeline failed to start.")
            finally:
                print("Stopping pipeline...")
                detector3.stop()
                viewer.close()

        except Exception as e:
            print(f"\nAn error occurred in __main__: {e}")
            print("This may be due to missing models (e.g., './mobileclip2_s0.pt')")
            print("or a missing config file.")
            import traceback
            traceback.print_exc()
            
        print("--- End of Example ---")
    else:
        print("Missing dependencies. Cannot run example.")