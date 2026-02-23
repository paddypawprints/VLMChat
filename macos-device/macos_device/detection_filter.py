"""
Thread-safe detection filter with multiple search term filters.

The DetectionFilter manages a list of SearchFilters and implements:
- OR logic for categories (detection matches if ANY filter enables the category)
- AND logic for attributes within each filter (detection must have ALL attributes)
- OR logic across filters (detection matches if it passes ANY filter)

All methods are protected by a lock for safe concurrent access.
"""

import threading
from typing import List, Dict, Optional
from camera_framework import CocoCategory
from .models.pa100k_onnx import PA100KOnnx as PA100KInference
from .search_filter import SearchFilter


class DetectionFilter:
    """Unified detection filter with boolean vectors for categories and attributes.
    
    Uses fixed-size boolean vectors for efficient filtering:
    - category_mask: 80 bools (one per COCO category, indexed by category.id)
    - attribute_mask: 26 bools (one per PA100K attribute, same order as PA100K.ATTRIBUTES)
    
    Thread-safe for MQTT updates. Shared by multiple filter tasks.
    
    Example:
        # Create shared filter
        filter_config = DetectionFilter()
        
        # Enable only persons
        filter_config.set_category_mask([i == CocoCategory.PERSON.id for i in range(80)])
        
        # Enable specific attributes (e.g., Female with Hat)
        attr_mask = [False] * 26
        attr_mask[PA100KInference.ATTRIBUTES.index('Female')] = True
        attr_mask[PA100KInference.ATTRIBUTES.index('Hat')] = True
        filter_config.set_attribute_mask(attr_mask)
        
        # Share filter between tasks
        yolo_router = YoloCategoryRouter(filter_config=filter_config)
        attribute_filter = AttributeFilter(filter_config=filter_config)
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        
        # List of search filters
        self._filters: List[SearchFilter] = []
        
        # Cached merged category mask (OR of all filter category masks)
        # Recomputed when filters change
        self._merged_category_mask: List[bool] = [False] * 80
    
    def set_filters(self, filters: List[SearchFilter]) -> None:
        """Replace entire filter list.
        
        Args:
            filters: List of SearchFilter objects
        """
        with self._lock:
            self._filters = filters.copy()
            self._recompute_merged_mask()
    
    def get_filters(self) -> List[SearchFilter]:
        """Get copy of current filter list.
        
        Returns:
            List of SearchFilter objects
        """
        with self._lock:
            return self._filters.copy()
    
    def _recompute_merged_mask(self) -> None:
        """Recompute merged category mask (OR of all filter category masks)."""
        # Start with all false
        merged = [False] * 80
        
        # OR all filter category masks
        for filter in self._filters:
            for i in range(80):
                if filter.category_mask[i]:
                    merged[i] = True
        
        self._merged_category_mask = merged
    
    def get_merged_category_mask(self) -> List[bool]:
        """Get the merged category mask (OR of all filters)."""
        with self._lock:
            return self._merged_category_mask.copy()
    
    def matches_category(self, category: CocoCategory) -> bool:
        """Check if a category is enabled in ANY filter (OR logic).
        
        Args:
            category: COCO category to check
            
        Returns:
            True if category is enabled in at least one filter
        """
        with self._lock:
            return self._merged_category_mask[category.id]
    
    def matches_attributes(self, attributes: Dict[str, Dict[str, any]]) -> bool:
        """Check if detection attributes match ANY filter.
        
        For each filter:
        - Detection must have ALL attributes enabled in that filter (AND logic)
        
        Across filters:
        - Detection passes if it matches ANY filter (OR logic)
        
        Args:
            attributes: Dict from detection.metadata['attributes']
                       Format: {attr: {'value': bool, 'confidence': float}}
        
        Returns:
            True if detection matches at least one filter
        """
        with self._lock:
            # If no filters, allow all detections
            if not self._filters:
                return True
            
            # Convert attributes dict to simple dict for SearchFilter
            simple_attrs = {}
            for attr_name, attr_data in attributes.items():
                if isinstance(attr_data, dict):
                    simple_attrs[attr_name] = attr_data.get('value', False)
                else:
                    simple_attrs[attr_name] = attr_data
            
            # Check if detection matches ANY filter
            for filter in self._filters:
                if filter.matches_attributes(simple_attrs):
                    return True
            
            return False
    
    def get_color_requirements(self, category_id: int) -> List[str]:
        """Get color requirements for a category across all filters.
        
        Returns unique list of color names required for this category.
        
        Args:
            category_id: COCO category ID
            
        Returns:
            List of color names (e.g., ['red', 'blue'])
        """
        with self._lock:
            colors = set()
            
            for filter in self._filters:
                # Only check filters that enable this category
                if filter.has_category(category_id):
                    color_list = filter.color_requirements.get(category_id, [])
                    colors.update(color_list)
            
            return list(colors)
