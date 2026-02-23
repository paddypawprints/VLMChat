"""
Search filter data structure for multi-filter support.

A SearchFilter represents a single search term with category and attribute masks.
The DetectionFilter stores a list of SearchFilters and implements OR logic for
categories and AND logic for attributes within each filter.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class SearchFilter:
    """
    Individual search term filter with category and attribute masks.
    
    Attributes:
        id: Unique filter identifier (search term ID from database)
        name: Human-readable filter name
        category_mask: Boolean mask for 80 COCO categories
        category_colors: Hex color strings for category visualization
        attribute_mask: Boolean mask for 26 PA-100K attributes
        attribute_colors: Hex color strings for attribute visualization
        color_requirements: Color requirements per category with region mapping 
                          {category_id: {region_name: [[r,g,b], ...]}}
                          e.g., {0: {"middle-top": [[255,255,255]]}} for person with white upper body
        vlm_required: Whether this filter needs VLM verification (from Groq)
        vlm_reasoning: Explanation of why VLM is needed (from Groq)
    """
    id: str
    name: str
    category_mask: List[bool]
    category_colors: List[str]
    attribute_mask: List[bool]
    attribute_colors: List[str]
    color_requirements: Dict[int, Dict[str, List[List[int]]]] = field(default_factory=dict)
    vlm_required: bool = False
    vlm_reasoning: str = ""
    
    def __post_init__(self):
        """Validate mask sizes."""
        if len(self.category_mask) != 80:
            raise ValueError(f"category_mask must have 80 items, got {len(self.category_mask)}")
        if len(self.category_colors) != 80:
            raise ValueError(f"category_colors must have 80 items, got {len(self.category_colors)}")
        if len(self.attribute_mask) != 26:
            raise ValueError(f"attribute_mask must have 26 items, got {len(self.attribute_mask)}")
        if len(self.attribute_colors) != 26:
            raise ValueError(f"attribute_colors must have 26 items, got {len(self.attribute_colors)}")
    
    def get_attribute_color_requirements(self) -> Dict[str, List[str]]:
        """Get color requirements per attribute for persons.
        
        Returns:
            Dict mapping attribute name to list of required color names
        """
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
        
        result = {}
        for idx, color_hex in enumerate(self.attribute_colors):
            if color_hex and idx < len(PA100K_ATTRIBUTES):
                attr_name = PA100K_ATTRIBUTES[idx]
                # Convert hex to color name
                from .color_utils import hex_to_color_name
                color_name = hex_to_color_name(color_hex)
                if color_name and color_name != 'unknown':
                    result[attr_name] = [color_name]
        
        return result
    
    def has_category(self, category_id: int) -> bool:
        """Check if this filter enables the given category."""
        if 0 <= category_id < 80:
            return self.category_mask[category_id]
        return False
    
    def matches_attributes(self, attributes: dict) -> bool:
        """
        Check if detection attributes match this filter.
        
        Uses AND logic - detection must have ALL attributes enabled in this filter.
        
        Args:
            attributes: Dict with PA-100K attribute names as keys (booleans or floats)
        
        Returns:
            True if detection has all required attributes
        """
        # Attribute names in PA-100K order
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
        
        # Check each attribute enabled in the mask
        for idx, enabled in enumerate(self.attribute_mask):
            if enabled:
                attr_name = PA100K_ATTRIBUTES[idx]
                
                # Check if detection has this attribute
                if attr_name not in attributes:
                    return False
                
                # Check if attribute value is truthy (>0.5 for floats, True for bools)
                value = attributes[attr_name]
                if isinstance(value, bool):
                    if not value:
                        return False
                elif isinstance(value, (int, float)):
                    if value <= 0.5:
                        return False
                else:
                    return False
        
        return True
