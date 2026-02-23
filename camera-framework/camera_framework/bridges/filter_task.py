"""Filter configuration update handler via MQTT."""

import logging
from typing import Dict, Any, List
from ..task import BaseTask

logger = logging.getLogger(__name__)


class FilterConfigTask(BaseTask):
    """One-shot task to update DetectionFilter from MQTT command.
    
    Receives complete filter list from MQTT and updates the shared DetectionFilter.
    
    Usage:
        # In MQTT callback
        config_task = FilterConfigTask(payload, detection_filter)
        runner.queue_task(config_task)
    """
    
    def __init__(self, config: Dict[str, Any], detection_filter):
        """Initialize filter update task.
        
        Args:
            config: Filter list config dict with key 'filters': List[Dict]
                   Each filter dict has: id, name, category_mask, category_colors,
                                        attribute_mask, attribute_colors, color_requirements
            detection_filter: DetectionFilter instance to update
        """
        super().__init__("filter_config_update")
        self.config = config
        self.detection_filter = detection_filter
        self._executed = False
    
    def process(self) -> None:
        """Update detection filter with new filter list."""
        if self._executed:
            return
        
        try:
            # Import SearchFilter here to avoid circular dependency
            from macos_device.search_filter import SearchFilter
            
            # Extract filter list
            filter_dicts = self.config.get('filters', [])
            
            if not isinstance(filter_dicts, list):
                logger.error(f"[FilterConfig] Invalid filters type: {type(filter_dicts)} (expected list)")
                return
            
            # Parse filters
            filters: List[SearchFilter] = []
            for filter_dict in filter_dicts:
                try:
                    # Get color_requirements, default to empty dict if not present
                    color_reqs = filter_dict.get('color_requirements', {})
                    
                    # Convert string keys to int (JSON keys are always strings)
                    color_requirements = {}
                    if color_reqs:
                        for key, value in color_reqs.items():
                            color_requirements[int(key)] = value
                    
                    filter = SearchFilter(
                        id=filter_dict['id'],
                        name=filter_dict['name'],
                        category_mask=filter_dict['category_mask'],
                        category_colors=filter_dict['category_colors'],
                        attribute_mask=filter_dict['attribute_mask'],
                        attribute_colors=filter_dict['attribute_colors'],
                        color_requirements=color_requirements,
                        vlm_required=filter_dict.get('vlm_required', False),
                        vlm_reasoning=filter_dict.get('vlm_reasoning', ''),
                    )
                    filters.append(filter)
                except (KeyError, ValueError) as e:
                    logger.error(f"[FilterConfig] Invalid filter dict: {e}")
                    continue
            
            # Update detection filter (thread-safe)
            self.detection_filter.set_filters(filters)
            
            logger.info(f"[FilterConfig] ✓ Applied {len(filters)} filters")
            for f in filters:
                logger.info(f"[FilterConfig]   - Filter '{f.name}': {sum(f.category_mask)}/80 cats, {sum(f.attribute_mask)}/26 attrs, color_reqs={f.color_requirements}")
            
            self._executed = True
            
        except Exception as e:
            logger.error(f"[FilterConfig] Failed to update filter: {e}")
            raise  # Propagate exception - filter updates are critical
    
    def poll(self) -> bool:
        """Always ready to execute."""
        return not self._executed
