"""
Connector subclasses for different split/merge strategies.

Provides specialized connectors that override default split/merge behavior:
- FirstCompleteConnector: Takes first branch to complete, ignores others
- OrderedMergeConnector: Reorders branch results before merging
"""

import logging
from typing import List, Dict, Optional
from .task_base import Connector, Context, ContextDataType

logger = logging.getLogger(__name__)


class FirstCompleteConnector(Connector):
    """
    Connector that takes the first branch to complete and ignores others.
    
    Useful for race conditions where multiple approaches are tried in parallel
    and the fastest valid result is desired.
    
    Example DSL:
        [clone_split(): fast_model, accurate_model :first_complete()]
    """
    
    def __init__(self, task_id: str, time_budget_ms: Optional[int] = None):
        super().__init__(task_id, time_budget_ms)
        logger.info(f"Created FirstCompleteConnector: {task_id}")
    
    def merge_strategy(self, contexts: List[Context]) -> Context:
        """
        Return the first context (first branch to complete).
        
        Args:
            contexts: List of contexts from parallel branches
            
        Returns:
            First context in the list
        """
        if not contexts:
            logger.warning(f"FirstCompleteConnector {self.task_id}: no contexts to merge")
            return Context()
        
        logger.info(f"FirstCompleteConnector {self.task_id}: taking first of {len(contexts)} contexts")
        return contexts[0]


class OrderedMergeConnector(Connector):
    """
    Connector that reorders branch results before merging.
    
    Allows specifying priority order for branches. Useful when certain
    branches should be processed before others in the merge.
    
    Example DSL:
        [clone_split(): high_priority, low_priority :ordered_merge(order=2,1)]
        # Would process low_priority results first, then high_priority
    
    Configuration:
        order: Comma-separated list of 1-based indices (e.g., "2,1,3")
    """
    
    def __init__(self, task_id: str, time_budget_ms: Optional[int] = None):
        super().__init__(task_id, time_budget_ms)
        self.order: Optional[List[int]] = None
        logger.info(f"Created OrderedMergeConnector: {task_id}")
    
    def configure(self, params: Dict[str, str]) -> None:
        """
        Configure merge order from DSL parameters.
        
        Args:
            params: Dictionary with "order" key containing comma-separated indices
            
        Example:
            params = {"order": "2,1,3"}
            # Converts to [1, 0, 2] (0-indexed)
        """
        if "order" in params:
            try:
                # Parse "1,2,3" to [0,1,2] (convert to 0-indexed)
                self.order = [int(x.strip()) - 1 for x in params["order"].split(",")]
                logger.info(f"OrderedMergeConnector {self.task_id}: configured with order {self.order}")
            except (ValueError, IndexError) as e:
                logger.error(f"OrderedMergeConnector {self.task_id}: invalid order parameter '{params['order']}': {e}")
                raise ValueError(f"Invalid order parameter: {params['order']}")
    
    def merge_strategy(self, contexts: List[Context]) -> Context:
        """
        Reorder contexts according to configured order, then merge.
        
        Args:
            contexts: List of contexts from parallel branches
            
        Returns:
            Merged context with reordered inputs
        """
        if not contexts:
            logger.warning(f"OrderedMergeConnector {self.task_id}: no contexts to merge")
            return Context()
        
        # Reorder if order is specified
        if self.order:
            if len(self.order) != len(contexts):
                logger.warning(
                    f"OrderedMergeConnector {self.task_id}: order length ({len(self.order)}) "
                    f"doesn't match contexts ({len(contexts)}), using default order"
                )
            else:
                try:
                    contexts = [contexts[i] for i in self.order]
                    logger.info(f"OrderedMergeConnector {self.task_id}: reordered contexts using {self.order}")
                except IndexError as e:
                    logger.error(f"OrderedMergeConnector {self.task_id}: invalid order indices: {e}")
                    # Continue with original order
        
        # Use default merge strategy (concatenate mutable, first immutable)
        return super().merge_strategy(contexts)
