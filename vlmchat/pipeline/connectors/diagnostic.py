"""
Diagnostic connector for testing split/merge behavior.

Provides configurable merge strategies:
- Default: Concatenate all data
- First: Take only first context
- Last: Take only last context
- Custom: Use provided merge function
"""

import logging
import threading
from typing import List, Optional, Callable, Dict, Any
from ..core.task_base import Connector, Context, ContextDataType

logger = logging.getLogger(__name__)


class DiagnosticConnector(Connector):
    """
    Connector for testing parallel execution and merge strategies.
    
    Merge strategies:
    - 'concat': Default, concatenate all data (standard behavior)
    - 'first': Use only first context
    - 'last': Use only last context
    - 'custom': Use provided merge_fn
    
    Split strategies:
    - 'default': Standard deep copy for mutable, share immutable
    - 'custom': Use provided split_fn
    """
    
    def __init__(
        self,
        merge_mode: str = 'concat',
        merge_fn: Optional[Callable[[List[Context]], Context]] = None,
        split_fn: Optional[Callable[[Context, int], List[Context]]] = None,
        log_events: bool = True,
        task_id: Optional[str] = None
    ):
        super().__init__(task_id=task_id or "diagnostic_connector")
        self.merge_mode = merge_mode
        self.merge_fn = merge_fn
        self.split_fn = split_fn
        self.log_events = log_events
        self.merge_count = 0
        self.split_count = 0
    
    def configure(self, **kwargs):
        """Configure connector from DSL parameters."""
        if 'merge_mode' in kwargs:
            self.merge_mode = kwargs['merge_mode']
        if 'log_events' in kwargs:
            self.log_events = kwargs['log_events'].lower() in ('true', '1', 'yes')
    
    def merge_strategy(self, contexts: List[Context]) -> Context:
        """Execute configured merge strategy with logging."""
        thread_id = threading.current_thread().name
        thread_ident = threading.get_ident()
        self.merge_count += 1
        
        if self.log_events:
            context_summary = []
            for i, ctx in enumerate(contexts):
                data_counts = {dt.type_name: len(items) for dt, items in ctx.data.items()}
                context_summary.append(f"ctx{i}={data_counts}")
            
            logger.info(
                f"[{thread_id}#{thread_ident}] DiagnosticConnector.merge: "
                f"mode={self.merge_mode}, inputs={len(contexts)}, {', '.join(context_summary)}"
            )
        
        # Execute merge based on mode
        if self.merge_mode == 'first':
            merged = contexts[0] if contexts else Context()
            if self.log_events:
                logger.info(
                    f"[{thread_id}#{thread_ident}] DiagnosticConnector.merge: "
                    f"using first context"
                )
        
        elif self.merge_mode == 'last':
            merged = contexts[-1] if contexts else Context()
            if self.log_events:
                logger.info(
                    f"[{thread_id}#{thread_ident}] DiagnosticConnector.merge: "
                    f"using last context"
                )
        
        elif self.merge_mode == 'custom':
            if self.merge_fn is None:
                raise ValueError("merge_mode='custom' requires merge_fn parameter")
            merged = self.merge_fn(contexts)
            if self.log_events:
                logger.info(
                    f"[{thread_id}#{thread_ident}] DiagnosticConnector.merge: "
                    f"using custom function"
                )
        
        else:  # 'concat' or default
            merged = super().merge_strategy(contexts)
            if self.log_events:
                data_counts = {dt.type_name: len(items) for dt, items in merged.data.items()}
                logger.info(
                    f"[{thread_id}#{thread_ident}] DiagnosticConnector.merge: "
                    f"concatenated to {data_counts}"
                )
        
        return merged
    
    def split_strategy(self, context: Context, num_branches: int) -> List[Context]:
        """Execute configured split strategy with logging."""
        thread_id = threading.current_thread().name
        thread_ident = threading.get_ident()
        self.split_count += 1
        
        if self.log_events:
            data_counts = {dt.type_name: len(items) for dt, items in context.data.items()}
            logger.info(
                f"[{thread_id}#{thread_ident}] DiagnosticConnector.split: "
                f"branches={num_branches}, input={data_counts}"
            )
        
        # Execute split based on mode
        if self.split_fn is not None:
            splits = self.split_fn(context, num_branches)
            if self.log_events:
                logger.info(
                    f"[{thread_id}#{thread_ident}] DiagnosticConnector.split: "
                    f"using custom function"
                )
        else:
            splits = super().split_strategy(context, num_branches)
            if self.log_events:
                logger.info(
                    f"[{thread_id}#{thread_ident}] DiagnosticConnector.split: "
                    f"using default strategy"
                )
        
        return splits


if __name__ == "__main__":
    """Test diagnostic connector behavior."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("="*70)
    print("DiagnosticConnector Tests")
    print("="*70)
    
    from vlmchat.pipeline.diagnostic_task import DiagnosticTask
    
    # Test 1: Concat merge (default)
    print("\n1. Concat merge (default):")
    connector1 = DiagnosticConnector(merge_mode='concat')
    
    ctx1a = Context()
    ctx1a.data[ContextDataType.DETECTIONS] = ['det1', 'det2']  # Mutable type
    ctx1b = Context()
    ctx1b.data[ContextDataType.DETECTIONS] = ['det3', 'det4']
    
    merged1 = connector1.merge_strategy([ctx1a, ctx1b])
    assert len(merged1.data[ContextDataType.DETECTIONS]) == 4
    print(f"✅ Concat merge: {len(merged1.data[ContextDataType.DETECTIONS])} detections")
    
    # Test 2: First merge
    print("\n2. First merge:")
    connector2 = DiagnosticConnector(merge_mode='first')
    
    ctx2a = Context()
    ctx2a.data[ContextDataType.IMAGE] = ['first1', 'first2']
    ctx2b = Context()
    ctx2b.data[ContextDataType.IMAGE] = ['second1', 'second2']
    
    merged2 = connector2.merge_strategy([ctx2a, ctx2b])
    assert merged2.data[ContextDataType.IMAGE] == ['first1', 'first2']
    print(f"✅ First merge: {merged2.data[ContextDataType.IMAGE]}")
    
    # Test 3: Last merge
    print("\n3. Last merge:")
    connector3 = DiagnosticConnector(merge_mode='last')
    
    merged3 = connector3.merge_strategy([ctx2a, ctx2b])
    assert merged3.data[ContextDataType.IMAGE] == ['second1', 'second2']
    print(f"✅ Last merge: {merged3.data[ContextDataType.IMAGE]}")
    
    # Test 4: Custom merge
    print("\n4. Custom merge (take longest list):")
    
    def merge_longest(contexts: List[Context]) -> Context:
        merged = Context()
        for data_type in [ContextDataType.IMAGE, ContextDataType.DETECTIONS]:
            longest = []
            for ctx in contexts:
                if data_type in ctx.data and len(ctx.data[data_type]) > len(longest):
                    longest = ctx.data[data_type]
            if longest:
                merged.data[data_type] = longest
        return merged
    
    connector4 = DiagnosticConnector(merge_mode='custom', merge_fn=merge_longest)
    
    ctx4a = Context()
    ctx4a.data[ContextDataType.IMAGE] = ['a']
    ctx4b = Context()
    ctx4b.data[ContextDataType.IMAGE] = ['b1', 'b2', 'b3']
    
    merged4 = connector4.merge_strategy([ctx4a, ctx4b])
    assert len(merged4.data[ContextDataType.IMAGE]) == 3
    print(f"✅ Custom merge: {len(merged4.data[ContextDataType.IMAGE])} images (longest)")
    
    # Test 5: Split strategy
    print("\n5. Default split:")
    connector5 = DiagnosticConnector()
    
    ctx5 = Context()
    ctx5.data[ContextDataType.IMAGE] = ['img1', 'img2']
    ctx5.data[ContextDataType.DETECTIONS] = [{'box': [1,2,3,4]}]
    
    splits = connector5.split_strategy(ctx5, 3)
    assert len(splits) == 3
    assert all(ContextDataType.IMAGE in split.data for split in splits)
    print(f"✅ Split into {len(splits)} branches")
    
    # Test 6: Event counting
    print("\n6. Event counting:")
    connector6 = DiagnosticConnector()
    
    connector6.merge_strategy([Context(), Context()])
    connector6.split_strategy(Context(), 2)
    connector6.merge_strategy([Context()])
    
    assert connector6.merge_count == 2
    assert connector6.split_count == 1
    print(f"✅ Event counting: {connector6.merge_count} merges, {connector6.split_count} splits")
    
    print("\n" + "="*70)
    print("All DiagnosticConnector tests passed!")
    print("="*70)
