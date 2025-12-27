"""
Pipeline class - A BaseTask that executes named sub-pipelines.

This allows pipeline composition where pipelines can invoke other pipelines,
enabling modular and reusable pipeline components.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .task_base import BaseTask, Context, register_task

logger = logging.getLogger(__name__)


@register_task('pipeline')
class Pipeline(BaseTask):
    """
    A pipeline that can be executed as a task within another pipeline.
    
    This enables recursive pipeline composition where complex pipelines
    can be built from simpler, named pipeline components.
    
    Example DSL usage:
        camera() -> pipeline(name="detection") -> display()
        
        pipeline(name="semantic_search", prompts="person,car", threshold=0.15)
    
    Attributes:
        name: Name of the pipeline to execute (looks up in registry)
        dsl: Optional inline DSL string (alternative to name lookup)
        ast: Parsed AST of the pipeline
        registry: Task registry for parsing
        pipeline_registry: Registry of named pipelines (name -> DSL/file)
        overrides: Parameter overrides to apply to sub-pipeline tasks
    """
    
    def __init__(self, 
                 name: Optional[str] = None,
                 dsl: Optional[str] = None,
                 task_registry: Optional[Dict[str, Any]] = None,
                 pipeline_registry: Optional[Dict[str, Any]] = None,
                 task_id: str = "pipeline"):
        """
        Initialize pipeline task.
        
        Args:
            name: Name of pipeline to execute (from pipeline_registry)
            dsl: Inline DSL string (alternative to name)
            task_registry: Registry of task types for parsing
            pipeline_registry: Registry of named pipelines (name -> DSL/file)
            task_id: Unique task identifier
        """
        super().__init__(task_id)
        self.name = name
        self.dsl = dsl
        self.ast = None
        self.task_registry = task_registry
        self.pipeline_registry = pipeline_registry or {}
        self.overrides: Dict[str, Any] = {}
        
        # Define contracts - pipeline tasks are flexible
        self.input_contract = {}
        self.output_contract = {}
        
        logger.info(f"Pipeline '{task_id}' initialized (name='{name}')")
    
    def configure(self, **kwargs) -> None:
        """
        Configure the pipeline task and capture parameter overrides.
        
        Args:
            **kwargs: Configuration parameters:
                - name: Name of pipeline to execute
                - dsl: Inline DSL string
                - file: Path to .dsl file to load
                - Any other params become overrides for sub-pipeline tasks
        """
        if 'name' in kwargs:
            self.name = kwargs['name']
            logger.info(f"Pipeline '{self.task_id}': Set name='{self.name}'")
        
        if 'dsl' in kwargs:
            self.dsl = kwargs['dsl']
            logger.info(f"Pipeline '{self.task_id}': Set inline DSL")
        
        if 'file' in kwargs:
            # Load DSL from file
            import pathlib
            file_path = pathlib.Path(kwargs['file'])
            if not file_path.exists():
                raise ValueError(f"Pipeline '{self.task_id}': DSL file not found: {file_path}")
            self.dsl = file_path.read_text()
            logger.info(f"Pipeline '{self.task_id}': Loaded DSL from {file_path}")
        
        # All other parameters are overrides for sub-pipeline tasks
        excluded_keys = {'name', 'dsl', 'file'}
        for key, value in kwargs.items():
            if key not in excluded_keys:
                self.overrides[key] = value
                logger.info(f"Pipeline '{self.task_id}': Override {key}={value}")
    
    def _load_pipeline_dsl(self) -> str:
        """
        Load the pipeline DSL from name, inline DSL, or file.
        
        Returns:
            str: The DSL text to parse
            
        Raises:
            ValueError: If pipeline cannot be found
        """
        # If we have inline DSL, use it
        if self.dsl:
            logger.info(f"Pipeline '{self.task_id}': Using inline DSL")
            return self.dsl
        
        # Otherwise look up by name in pipeline registry
        if not self.name:
            raise ValueError(f"Pipeline '{self.task_id}': No name or DSL provided")
        
        if self.name not in self.pipeline_registry:
            raise ValueError(
                f"Pipeline '{self.task_id}': Pipeline '{self.name}' not found in registry. "
                f"Available: {list(self.pipeline_registry.keys())}"
            )
        
        pipeline_def = self.pipeline_registry[self.name]
        
        # If it's a string, treat as DSL
        if isinstance(pipeline_def, str):
            logger.info(f"Pipeline '{self.task_id}': Loaded '{self.name}' as DSL string")
            return pipeline_def
        
        # If it's a Path, load from file
        if isinstance(pipeline_def, (Path, str)):
            file_path = Path(pipeline_def)
            if file_path.exists():
                logger.info(f"Pipeline '{self.task_id}': Loading '{self.name}' from {file_path}")
                with open(file_path, 'r') as f:
                    return f.read()
            else:
                raise ValueError(f"Pipeline '{self.task_id}': File not found: {file_path}")
        
        raise ValueError(
            f"Pipeline '{self.task_id}': Invalid pipeline definition type: {type(pipeline_def)}"
        )
    
    def _apply_overrides_to_tasks(self, root_node) -> None:
        """
        Apply parameter overrides to tasks in the parsed pipeline.
        
        Recursively walks the AST and calls configure() on tasks with override values.
        
        Args:
            root_node: Root node of the parsed AST
        """
        if not self.overrides:
            return
        
        logger.info(f"Pipeline '{self.task_id}': Applying {len(self.overrides)} overrides")
        
        from .dsl_parser import TaskNode, SequenceNode, ParallelNode, LoopNode
        from .task_base import Connector
        
        def apply_to_node(node):
            """Recursively apply overrides to all tasks in the AST."""
            if isinstance(node, TaskNode):
                # This is a task node - but we need to apply to the actual task instance
                # This will be done after building, so we store overrides on the node
                if not hasattr(node, 'param_overrides'):
                    node.param_overrides = {}
                node.param_overrides.update(self.overrides)
                
            elif isinstance(node, SequenceNode):
                for task in node.tasks:
                    apply_to_node(task)
                    
            elif isinstance(node, ParallelNode):
                for task in node.tasks:
                    apply_to_node(task)
                    
            elif isinstance(node, LoopNode):
                apply_to_node(node.body)
        
        apply_to_node(root_node)
    
    def run(self, context: Context) -> Context:
        """
        Execute the sub-pipeline.
        
        Args:
            context: Input context from upstream tasks
            
        Returns:
            Context: Output context from the sub-pipeline
            
        Raises:
            Exception: If pipeline execution fails
        """
        logger.info(f"Pipeline '{self.task_id}': Starting sub-pipeline '{self.name}'")
        
        # Load and parse the pipeline if not already done
        if self.ast is None:
            dsl_text = self._load_pipeline_dsl()
            
            from .dsl_parser import DSLParser, create_task_registry
            
            # Use provided task registry or create default
            if self.task_registry is None:
                self.task_registry = create_task_registry()
            
            parser = DSLParser(self.task_registry)
            
            logger.info(f"Pipeline '{self.task_id}': Parsing DSL")
            self.ast = parser.parse(dsl_text)
            
            # Apply overrides to the parsed AST
            self._apply_overrides_to_tasks(self.ast)
        
        # Execute the sub-pipeline
        from .pipeline_runner import PipelineRunner
        from .task_base import Connector
        
        # Ensure ast is properly wrapped for PipelineRunner
        # If it's a single task that's not a Connector, wrap it
        execution_root = self.ast
        if not isinstance(self.ast, (Connector, list)):
            # Single task - wrap in a Connector
            wrapper = Connector(task_id=f"{self.task_id}_wrapper")
            wrapper.internal_tasks = [self.ast]
            execution_root = wrapper
        
        # Use fewer workers for sub-pipelines to avoid resource exhaustion
        max_workers = 2
        runner = PipelineRunner(
            execution_root,
            max_workers=max_workers,
            enable_trace=False  # Disable tracing for sub-pipelines
        )
        
        try:
            logger.info(f"Pipeline '{self.task_id}': Executing sub-pipeline")
            result_context = runner.run(context)
            logger.info(f"Pipeline '{self.task_id}': Sub-pipeline completed successfully")
            return result_context
            
        except Exception as e:
            logger.error(f"Pipeline '{self.task_id}': Sub-pipeline failed: {e}", exc_info=True)
            raise
        
        finally:
            runner.shutdown()
