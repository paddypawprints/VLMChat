"""Clusterer task for filter-based object association.

Performs hierarchical clustering of detections based on filters.
Clusters objects that belong to the same search filter using:
- Spatial proximity (packing efficiency)
- Size compatibility  
- Category matching (from filter)
- Soft attribute matching (for person detections)
"""

import logging
import math
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass
from camera_framework import BaseTask, CocoCategory
from .config import ClustererConfig


logger = logging.getLogger(__name__)


@dataclass
class ClusterNode:
    """Node in clustering hierarchy.
    
    Attributes:
        detection: Detection object
        pixel_area: Total pixel area of detections in this cluster
        categories: Set of category IDs in this cluster
        category_areas: Dict mapping category ID to list of bbox areas
        creation_cost: Similarity score when this cluster was created
    """
    detection: any  # Detection object
    pixel_area: float
    categories: Set[int]
    category_areas: Dict[int, List[float]]
    creation_cost: float = 1.0


class Clusterer(BaseTask):
    """
    Hierarchical clustering for per-filter object association.
    
    Runs clustering independently for each filter that has 2+ categories enabled.
    Uses spatial proximity, size compatibility, and category/attribute matching.
    
    Example:
        from .detection_filter import DetectionFilter
        
        filter_config = DetectionFilter()
        clusterer = Clusterer(filter_config=filter_config)
        clusterer.add_input("detections", color_filtered_buffer)
        clusterer.add_output("clustered", mqtt_buffer)
    """
    
    def __init__(
        self,
        config: ClustererConfig,
        name: str = "clusterer",
        filter_config: Optional['DetectionFilter'] = None,
    ):
        """
        Initialize clusterer.
        
        Args:
            config: ClustererConfig instance with clustering settings
            name: Task name
            filter_config: Shared DetectionFilter instance
        
        Raises:
            ValueError: If config is None
        """
        super().__init__(name=name)
        
        if config is None:
            raise ValueError("ClustererConfig is required for Clusterer")
        
        self.config = config
        self.filter_config = filter_config
        self.max_clusters = config.max_clusters
        self.merge_threshold = config.merge_threshold
        self.proximity_weight = config.weights.proximity
        self.size_weight = config.weights.size
        self.category_weight = config.weights.category
        self.attribute_weight = config.weights.attribute
        
        logger.info(f"Clusterer initialized - max_clusters={config.max_clusters}, "
                   f"merge_threshold={config.merge_threshold}")
        logger.debug(f"Weights - proximity={config.weights.proximity}, size={config.weights.size}, "
                    f"category={config.weights.category}, attribute={config.weights.attribute}")
    
    def process(self) -> None:
        """Run clustering for each filter."""
        if not self.inputs or not self.filter_config:
            return
        
        input_buffer = list(self.inputs.values())[0]
        message = input_buffer.get()
        if not message:
            return
        
        detections = message.get("detections", [])
        if not detections:
            return
        
        # Get filters from config
        filters = self.filter_config._filters
        if not filters:
            # No filters - pass through
            if self.outputs:
                output_buffer = list(self.outputs.values())[0]
                message = {"detections": detections}
                output_buffer.put(message)
            return
        
        # Run clustering for each filter with 2+ categories
        all_clustered = []
        processed_ids = set()  # Track which detections we've clustered
        
        for search_filter in filters:
            # Count enabled categories in this filter
            enabled_categories = set([i for i, enabled in enumerate(search_filter.category_mask) if enabled])
            
            if len(enabled_categories) < 2:
                # Single category - no clustering needed
                continue
            
            # Get detections for this filter (ONLY from enabled categories)
            filter_detections = []
            for det in detections:
                det_id = id(det)
                # Skip if already clustered in another filter
                if det_id in processed_ids:
                    continue
                
                # ONLY include detections whose category is enabled in this filter
                if det.category.id in enabled_categories:
                    filter_detections.append(det)
            
            if len(filter_detections) < 2:
                # Not enough detections to cluster
                continue
            
            # Run clustering for this filter
            logger.debug(f"[Clusterer] Filter '{search_filter.name}': {len(filter_detections)} detections from categories {sorted(enabled_categories)}")
            
            clustered = self._cluster_for_filter(
                filter_detections,
                search_filter,
                enabled_categories
            )
            
            # Mark these detections as processed
            for det in filter_detections:
                processed_ids.add(id(det))
            
            all_clustered.extend(clustered)
        
        # Add unclustered detections (single-category filters or not in any filter)
        for det in detections:
            if id(det) not in processed_ids:
                all_clustered.append(det)
        
        # Write to output
        if all_clustered and self.outputs:
            output_buffer = list(self.outputs.values())[0]
            message = {"detections": all_clustered}
            output_buffer.put(message)
    
    def _cluster_for_filter(
        self,
        detections: List,
        search_filter,
        enabled_categories: Set[int]
    ) -> List:
        """
        Run hierarchical clustering for detections in a filter.
        
        NOTE: All detections passed in are guaranteed to have categories
        in enabled_categories (pre-filtered in process()).
        
        CLUSTERING STRATEGY:
        - Only merges DIFFERENT categories (e.g., person + bicycle)
        - Does NOT merge same categories (e.g., person + person)
        - This associates objects across categories (person WITH bicycle)
        
        Args:
            detections: List of Detection objects (already filtered by category)
            search_filter: SearchFilter with category/attribute masks
            enabled_categories: Set of category IDs enabled in this filter
            
        Returns:
            List of clustered Detection objects (top-level only, children removed)
        """
        # Initialize cluster nodes
        nodes = []
        for det in detections:
            bbox = det.bbox
            area = self._box_area(bbox)
            
            node = ClusterNode(
                detection=det,
                pixel_area=area,
                categories={det.category.id},
                category_areas={det.category.id: [area]},
                creation_cost=1.0
            )
            nodes.append(node)
        
        # Hierarchical clustering
        max_iterations = 100
        for iteration in range(max_iterations):
            if len(nodes) <= 1:
                break
            
            # Find best merge candidate
            best_sim = -1.0
            best_i, best_j = -1, -1
            
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    # Skip same-category pairs (only merge different categories)
                    if nodes[i].categories.intersection(nodes[j].categories):
                        continue
                    
                    sim, _ = self._calculate_similarity(
                        nodes[i],
                        nodes[j],
                        search_filter,
                        enabled_categories
                    )
                    
                    if sim > best_sim:
                        best_sim = sim
                        best_i, best_j = i, j
            
            # Check if best candidate meets threshold
            if best_sim < self.merge_threshold:
                logger.debug(f"[Clusterer] Stopping: best similarity {best_sim:.3f} < threshold {self.merge_threshold}")
                break
            
            # Merge best candidate
            logger.debug(f"[Clusterer] Iteration {iteration}: Merged {nodes[best_i].detection.category.label} + {nodes[best_j].detection.category.label}, similarity={best_sim:.3f}")
            
            merged = self._merge_nodes(
                nodes[best_i],
                nodes[best_j],
                best_sim
            )
            
            # Remove old nodes and add merged
            nodes = [n for idx, n in enumerate(nodes) if idx not in (best_i, best_j)]
            nodes.append(merged)
        
        # Limit to max clusters
        if len(nodes) > self.max_clusters:
            logger.info(f"[Clusterer] Limiting {len(nodes)} clusters to {self.max_clusters}")
            nodes.sort(key=lambda n: n.creation_cost, reverse=True)
            nodes = nodes[:self.max_clusters]
        
        # Extract top-level detections
        result = []
        for node in nodes:
            result.append(node.detection)
        
        return result
    
    def _calculate_similarity(
        self,
        a: ClusterNode,
        b: ClusterNode,
        search_filter,
        enabled_categories: Set[int]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate similarity between two cluster nodes.
        
        NOTE: Both nodes are guaranteed to have categories in enabled_categories
        (pre-filtered), so category matching is implicitly 1.0.
        
        Args:
            a: First cluster node
            b: Second cluster node
            search_filter: SearchFilter with category/attribute requirements
            enabled_categories: Set of category IDs enabled in this filter
            search_filter: SearchFilter with category/attribute requirements
            
        Returns:
            (total_similarity, component_dict)
        """
        # 1. Proximity similarity (packing efficiency)
        a_box = a.detection.bbox
        b_box = b.detection.bbox
        
        merged_box = self._union_box(a_box, b_box)
        merged_box_area = self._box_area(merged_box)
        total_pixel_area = a.pixel_area + b.pixel_area
        
        sim_prox = min(1.0, total_pixel_area / merged_box_area) if merged_box_area > 0 else 0.0
        
        # 2. Size similarity
        sim_size = 1.0
        shared_categories = a.categories.intersection(b.categories)
        
        if shared_categories:
            # Same category - should not happen since we skip same-category pairs
            # But if it does (e.g., after merging), compare sizes
            ratios = []
            for cat_id in shared_categories:
                areas_a = a.category_areas.get(cat_id, [])
                areas_b = b.category_areas.get(cat_id, [])
                
                if areas_a and areas_b:
                    avg_a = sum(areas_a) / len(areas_a)
                    avg_b = sum(areas_b) / len(areas_b)
                    
                    if avg_a > 0 and avg_b > 0:
                        ratio = min(avg_a, avg_b) / max(avg_a, avg_b)
                        ratios.append(ratio)
            
            if ratios:
                sim_size = sum(ratios) / len(ratios)
        else:
            # Different categories (expected case: person + bicycle)
            # Size compatibility: smaller object should be reasonable relative to larger
            min_area = min(a.pixel_area, b.pixel_area)
            max_area = max(a.pixel_area, b.pixel_area)
            sim_size = min_area / max_area if max_area > 0 else 0.0
        
        # 3. Category match
        # Both detections are guaranteed to be from enabled_categories (pre-filtered)
        # so category matching is always 1.0 within a filter's clustering
        sim_cat = 1.0
        
        # 4. Attribute match (for person detections)
        sim_attr = 0.5  # Neutral default
        
        # Check if either detection is a person - compute filter attribute match
        a_is_person = a.detection.category == CocoCategory.PERSON
        b_is_person = b.detection.category == CocoCategory.PERSON
        
        if a_is_person or b_is_person:
            # For person detections, check how well they match the filter's attribute requirements
            # Use dot product of detection attributes vs filter attribute mask
            
            attr_scores = []
            
            if a_is_person:
                a_attrs = a.detection.metadata.get('attributes', {})
                a_score = self._attribute_filter_match(a_attrs, search_filter.attribute_mask)
                attr_scores.append(a_score)
            
            if b_is_person:
                b_attrs = b.detection.metadata.get('attributes', {})
                b_score = self._attribute_filter_match(b_attrs, search_filter.attribute_mask)
                attr_scores.append(b_score)
            
            # Use average of person attribute matches
            if attr_scores:
                sim_attr = sum(attr_scores) / len(attr_scores)
        
        # Weighted average
        total_weight = self.proximity_weight + self.size_weight + self.category_weight + self.attribute_weight
        
        if total_weight == 0:
            return 0.0, {'prox': sim_prox, 'size': sim_size, 'cat': sim_cat, 'attr': sim_attr}
        
        weighted_sum = (
            sim_prox * self.proximity_weight +
            sim_size * self.size_weight +
            sim_cat * self.category_weight +
            sim_attr * self.attribute_weight
        )
        
        total_sim = weighted_sum / total_weight
        
        return total_sim, {'prox': sim_prox, 'size': sim_size, 'cat': sim_cat, 'attr': sim_attr}
    
    def _attribute_filter_match(self, detection_attrs: Dict, filter_mask: List[bool]) -> float:
        """
        Compute dot product between detection attribute vector and filter attribute mask.
        
        This measures how well a person detection matches the filter's attribute requirements.
        
        Args:
            detection_attrs: Detection attributes dict {attr_name: {value: bool, confidence: float}}
            filter_mask: Boolean mask for 26 PA-100K attributes
            
        Returns:
            Normalized score 0-1 (1.0 = perfect match)
        """
        # PA-100K attribute names in order
        PA100K_ATTRIBUTES = [
            "Female", "AgeOver60", "Age18-60", "AgeLess18",
            "Front", "Side", "Back",
            "Hat", "Glasses",
            "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront",
            "ShortSleeve", "LongSleeve",
            "UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice",
            "LowerStripe", "LowerPattern", "LongCoat",
            "Trousers", "Shorts", "Skirt&Dress",
            "boots"
        ]
        
        # Count how many filter attributes are enabled
        num_filter_attrs = sum(filter_mask)
        
        if num_filter_attrs == 0:
            # No filter requirements - perfect match
            return 1.0
        
        # Build detection attribute vector (0/1 for each attribute)
        det_vector = []
        for attr_name in PA100K_ATTRIBUTES:
            if attr_name in detection_attrs:
                attr_data = detection_attrs[attr_name]
                value = attr_data.get('value', False) if isinstance(attr_data, dict) else attr_data
                det_vector.append(1.0 if value else 0.0)
            else:
                det_vector.append(0.0)
        
        # Compute dot product with filter mask
        dot_product = sum(d * f for d, f in zip(det_vector, filter_mask))
        
        # Normalize by number of filter requirements
        # Score = (number of matched requirements) / (total requirements)
        normalized_score = dot_product / num_filter_attrs if num_filter_attrs > 0 else 1.0
        
        return normalized_score
    
    def _merge_nodes(
        self,
        a: ClusterNode,
        b: ClusterNode,
        similarity: float
    ) -> ClusterNode:
        """
        Merge two cluster nodes.
        
        Args:
            a: First cluster node
            b: Second cluster node
            similarity: Merge similarity score
            
        Returns:
            New merged cluster node
        """
        from camera_framework import Detection
        
        # Merge spatial and category info
        new_box = self._union_box(a.detection.bbox, b.detection.bbox)
        new_pixel_area = a.pixel_area + b.pixel_area
        new_categories = a.categories.union(b.categories)
        
        new_category_areas = a.category_areas.copy()
        for cat_id, areas in b.category_areas.items():
            new_category_areas.setdefault(cat_id, []).extend(areas)
        
        # Choose category from larger detection
        new_category = a.detection.category if a.pixel_area > b.pixel_area else b.detection.category
        new_conf = max(a.detection.confidence, b.detection.confidence)
        
        # Get base image
        base_image = a.detection.source_image
        
        # Create new Detection with children
        merged_detection = Detection(
            bbox=new_box,
            confidence=new_conf,
            category=new_category,
            source_image=base_image
        )
        
        # Merge metadata
        merged_detection.metadata = a.detection.metadata.copy()
        merged_detection.metadata.update(b.detection.metadata)
        
        # Set children
        merged_detection.children = [a.detection, b.detection]
        
        return ClusterNode(
            detection=merged_detection,
            pixel_area=new_pixel_area,
            categories=new_categories,
            category_areas=new_category_areas,
            creation_cost=similarity
        )
    
    def _box_area(self, bbox: Tuple[float, float, float, float]) -> float:
        """Calculate bounding box area."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def _union_box(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Calculate union of two bounding boxes."""
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        return (x1, y1, x2, y2)
