"""
ObjectDetector pipeline stage for clustering detections.

This file contains:
1.  ClusterAuditLog: A class to store and format audit trails.
2.  ObjectClusterer: The main clustering class with auditing logic.
"""

# --- Base Imports ---
from typing import List, Optional, Tuple, Set, Dict, Any
from PIL import Image
import numpy as np
import math
import time # For __main__
import requests # For __main__
from io import BytesIO # For __main__
import logging # For __main__

# --- Base Detector Classes (Imported) ---
from .detection_base import ObjectDetector, Detection
from .semantic_provider import ISemanticCostProvider
from .coco_categories import CocoCategory

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

    def log_comparison(self, a: _ClusterNode, b: _ClusterNode, costs: Dict[str, float], total_cost: float):
        """Logs a single comparison between two nodes."""
        self.comparisons.append({
            "a_id": a.detection.id,
            "b_id": b.detection.id,
            "costs": costs,
            "total_cost": total_cost
        })

    def log_merge(self, a: _ClusterNode, b: _ClusterNode, new_node: _ClusterNode, cost: float):
        """Logs a successful merge operation."""
        self.merges.append({
            "merged_a_id": a.detection.id,
            "merged_b_id": b.detection.id,
            "new_cluster_id": new_node.detection.id,
            "cost": cost
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
            log_str += (
                f"Merge {i+1}: Merged [ID {merge['merged_a_id']}] and [ID {merge['merged_b_id']}] "
                f"into [ID {merge['new_cluster_id']}] (Cost: {merge['cost']:.4f})\n"
            )

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
                 semantic_provider: ISemanticCostProvider, # Use Interface
                 max_clusters: int = 10,
                 merge_threshold: float = 1.5, # Updated default
                 proximity_weight: float = 1.0,
                 size_weight: float = 0.5,
                 semantic_weights: Dict[str, float] = {"pair": 1.0}
                 ):
        """
        Initializes the clusterer.
        
        Args:
            source: The detector providing the base detections (e.g., YOLO).
            semantic_provider: A pre-built semantic provider instance.
            max_clusters: (K) The number of clusters to return after filtering.
            merge_threshold: Stop merging if the best cost is above this.
            proximity_weight: Weight for the proximity cost.
            size_weight: Weight for the size cost.
            semantic_weights: Dict of weights for semantic algorithms.
        """
        super().__init__(source)
        self.semantic_provider = semantic_provider
        self.max_clusters = max_clusters
        self.merge_threshold = merge_threshold
        # Store weights
        self.w_prox = proximity_weight
        self.w_size = size_weight
        self.semantic_weights = semantic_weights
        
        self._ready = False
        self._audit_enabled = False
        self._last_audit_log: ClusterAuditLog = ClusterAuditLog()
        print("ObjectClusterer: Initialized.")

    def start(self, audit: bool = False) -> None:
        """
        Starts the clusterer and its source.
        
        Args:
            audit: If True, enables detailed logging for the next run.
        """
        super().start(audit) # Pass audit flag to source
        self.semantic_provider.start()
        self._audit_enabled = audit
        self._ready = True
        print(f"ObjectClusterer: Started (Audit Enabled: {self._audit_enabled}).")


    def stop(self) -> None:
        super().stop()
        self.semantic_provider.stop()
        self._ready = False
        print("ObjectClusterer: Stopped.")

    def get_last_audit_log(self) -> ClusterAuditLog:
        """Returns the audit log from the most recent 'detect' call."""
        return self._last_audit_log

    def readiness(self) -> bool:
        return self._ready and super().readiness()

    def _calculate_merge_cost(self, a: _ClusterNode, b: _ClusterNode) -> Tuple[float, Dict[str, float]]:
        """
        Calculates the weighted geometric mean cost of merging two nodes.
        Returns the total cost AND a dictionary of component costs for auditing.
        """
        
        # --- 1. Proximity Cost (Packing Efficiency) ---
        merged_box = _union_box(a.detection.box, b.detection.box)
        merged_box_area = _box_area(merged_box)
        total_pixel_area = a.pixel_area + b.pixel_area
        
        if merged_box_area == 0:
            cost_prox_raw = 1.0
        else:
            wasted_space_ratio = (merged_box_area - total_pixel_area) / merged_box_area
            cost_prox_raw = max(0.0, wasted_space_ratio)
            
        cost_prox = 1.0 + cost_prox_raw 

        # --- 2. Size Cost ---
        cost_size = 1.0
        shared_categories = a.categories.intersection(b.categories)
        
        if shared_categories:
            worst_size_cost = 1.0
            for cat_enum in shared_categories:
                areas_a = a.category_areas.get(cat_enum, [])
                areas_b = b.category_areas.get(cat_enum, [])
                if not areas_a or not areas_b: continue
                median_a = np.median(areas_a)
                median_b = np.median(areas_b)
                max_median = max(median_a, median_b, 1)
                size_diff_ratio = abs(median_a - median_b) / max_median
                worst_size_cost = max(worst_size_cost, 1.0 + size_diff_ratio)
            cost_size = worst_size_cost
        
        # --- 3. Semantic Costs ---
        
        epsilon = 0.001
        log_cost_sum = 0.0
        total_weight = 0.0
        audit_costs: Dict[str, float] = {} # For logging

        log_cost_sum += self.w_prox * math.log(max(cost_prox, epsilon))
        total_weight += self.w_prox
        audit_costs['prox'] = cost_prox

        log_cost_sum += self.w_size * math.log(cost_size)
        total_weight += self.w_size
        audit_costs['size'] = cost_size

        for name, weight in self.semantic_weights.items():
            if weight == 0: continue
            
            cost_sem = 1.0
            if name == "pair":
                avg_cost = self._get_avg_semantic_cost(a, b, self.semantic_provider.get_pair_cost)
                cost_sem = 1.0 + avg_cost
                audit_costs['sem_pair'] = cost_sem
            elif name == "single":
                avg_cost = self._get_avg_semantic_cost(a, b, self.semantic_provider.get_single_cost)
                cost_sem = 1.0 + avg_cost
                audit_costs['sem_single'] = cost_sem
            
            log_cost_sum += weight * math.log(cost_sem)
            total_weight += weight

        # --- 4. Final Geometric Mean ---
        if total_weight == 0: 
            return 1.0, audit_costs
        
        total_cost = math.exp(log_cost_sum / total_weight)
        return total_cost, audit_costs

    def _get_avg_semantic_cost(self, a: _ClusterNode, b: _ClusterNode, cost_func) -> float:
        """Helper to get the average semantic cost between two nodes."""
        cost_sum = 0.0
        pair_count = 0
        
        for cat_a_enum in a.categories:
            for cat_b_enum in b.categories:
                cost_sum += cost_func(cat_a_enum.label, cat_b_enum.label)
                pair_count += 1
        
        return (cost_sum / pair_count) if pair_count > 0 else 1.0


    def _find_best_merge_pair(self, nodes: List[_ClusterNode], audit_log: ClusterAuditLog) -> Tuple[float, Optional[Tuple[int, int]]]:
        """Finds the pair of nodes with the lowest merge cost."""
        min_cost = float('inf')
        best_pair = None
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                total_cost, component_costs = self._calculate_merge_cost(nodes[i], nodes[j])
                
                if self._audit_enabled:
                    audit_log.log_comparison(nodes[i], nodes[j], component_costs, total_cost)
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_pair = (i, j)
                    
        return min_cost, best_pair

    def _merge_nodes(self, a: _ClusterNode, b: _ClusterNode, merge_cost: float) -> _ClusterNode:
        """Merges two cluster nodes into a new one."""
        
        new_box = _union_box(a.detection.box, b.detection.box)
        new_pixel_area = a.pixel_area + b.pixel_area
        new_categories = a.categories.union(b.categories)
        
        new_category_areas = a.category_areas.copy()
        for cat, areas in b.category_areas.items():
            new_category_areas.setdefault(cat, []).extend(areas)
        
        new_label = a.detection.object_category if a.pixel_area > b.pixel_area else b.detection.object_category
        new_conf = max(a.detection.conf, b.detection.conf)
        
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
            creation_cost=merge_cost # <-- Pass the merge cost
        )

    def _detect_internal(self, image: Image, detections: List[Detection]) -> List[Detection]:
        """
        Runs the hierarchical clustering algorithm.
        """
        self._last_audit_log = ClusterAuditLog()

        if not detections:
            return []
            
        # 1. Initialize nodes from base detections
        nodes: List[_ClusterNode] = []
        for det in detections:
            cat_enum = CocoCategory.from_string(det.object_category)
            if not cat_enum:
                continue
                
            area = _box_area(det.box)
            if area > 0:
                nodes.append(_ClusterNode(
                    detection=det,
                    pixel_area=area,
                    categories={cat_enum},
                    category_areas={cat_enum: [area]},
                    creation_cost=0.0 # Base cost is 0
                ))

        # 2. Run aggregation loop (CHANGED LOGIC)
        while True:
            if len(nodes) <= 1: # Stop if only one cluster remains
                break

            min_cost, best_pair = self._find_best_merge_pair(nodes, self._last_audit_log)
            
            # This is now the only stop condition
            if min_cost > self.merge_threshold or best_pair is None:
                break
                
            idx_a, idx_b = best_pair
            node_a = nodes[idx_a]
            node_b = nodes[idx_b]
            
            # Pass the cost to the merge function
            new_node = self._merge_nodes(node_a, node_b, min_cost)
            
            if self._audit_enabled:
                self._last_audit_log.log_merge(node_a, node_b, new_node, min_cost)
            
            nodes.pop(max(idx_a, idx_b))
            nodes.pop(min(idx_a, idx_b))
            nodes.append(new_node)

        # 3. Final filtering (CHANGED LOGIC)
        if len(nodes) > self.max_clusters:
            # Sort by creation_cost (lowest cost is best)
            nodes.sort(key=lambda n: n.creation_cost)
            nodes = nodes[:self.max_clusters] # Keep the K best
            
        # 4. Log final state and return
        if self._audit_enabled:
            self._last_audit_log.log_final_clusters(nodes)
            
        return [n.detection for n in nodes]


# --- Example Usage ---
if __name__ == "__main__":
    
    # --- Imports for __main__ ---
    try:
        from .yolo_detector import YoloV8Detector
        from .detection_viewer import DetectionViewer
        from .image_viewer import ImageViewer
        from ..models.MobileClip.clip_model import CLIPModel # Corrected relative path
        from .semantic_provider import ClipSemanticProvider
        from ..utils.config import load_config # Corrected relative path
        from ..metrics.metrics_collector import null_collector # Corrected relative path
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
        pil_image = Image.open(BytesIO(response.content))
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
            clip_model = CLIPModel(config=config, collector=null_collector()) # type: ignore
            yolo_detector = YoloV8Detector(model_name='yolov8n.pt')
            
            USER_PROMPTS = [
                "a person on a horse",
                "a person with a bicycle",
                "a group of people"
            ]
            
            semantic_provider = ClipSemanticProvider(
                clip_model=clip_model,
                categories=[m.label for m in CocoCategory], # Get all 80 labels
                user_prompts=USER_PROMPTS
            )
            
            print("\nCreating pipeline: YOLO -> Clusterer -> Viewer")
            viewer = ImageViewer(window_name="Object Clusterer Pipeline")
            
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
            detector3 = DetectionViewer(source=detector2, viewer=viewer)
            
            try:
                # --- Enable auditing ---
                detector3.start(audit=True)
                
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