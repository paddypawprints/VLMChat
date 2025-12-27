"""
Pipeline DSL Parser

Implements lexer, parser, and builder for the VLMChat Pipeline DSL.
Converts DSL text to executable pipeline objects.

Grammar:
    pipeline          ::= task_sequence | loop_pipeline | parallel_pipeline
    task_sequence     ::= task ( "->" pipeline )*
    loop_pipeline     ::= "{" loop_sequence "}"
    loop_sequence     ::= loop_item ( "->" loop_item )*
    loop_item         ::= operator | pipeline
    operator          ::= ":" task ":"
    task              ::= identifier "(" params? ")"
    parallel_pipeline ::= "[" operator? pipeline_list operator? "]"
    pipeline_list     ::= pipeline ( "," pipeline )*
    params            ::= param ( "," param )*
    param             ::= identifier "=" value
    value             ::= string | number | boolean
    
Key features:
    - Fully recursive: any pipeline can contain any other pipeline type
    - Operators always use :task: syntax (no ambiguity)
    - Control operators only valid in loop_sequence
    - Split/merge operators only valid in parallel_pipeline
    
Examples:
    # Task sequence
    task_a() -> task_b() -> task_c()
    
    # Loop with operator
    {input() -> :break_on(code=1): -> process()}
    
    # Parallel with merge operator
    [task_a(), task_b() :ordered_merge(order="0,1"):]
    
    # Nested structures
    {task() -> [parallel_a(), parallel_b()] -> task()}
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TokenType(Enum):
    """Token types for lexical analysis."""
    # Operators
    ARROW = "->"
    COMMA = ","
    COLON = ":"
    EQUALS = "="
    TILDE = "~"
    GREATER_EQUAL = ">="
    
    # Brackets
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    LBRACE = "{"
    RBRACE = "}"
    
    # Literals
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    
    # Special
    EOF = "EOF"
    COMMENT = "COMMENT"


@dataclass
class Token:
    """A lexical token."""
    type: TokenType
    value: Any
    line: int
    column: int


class Lexer:
    """Tokenizes DSL text."""
    
    # Token patterns (order matters)
    PATTERNS = [
        (TokenType.COMMENT, r'#[^\n]*'),
        (TokenType.GREATER_EQUAL, r'>='),  # Must come before individual symbols
        (TokenType.ARROW, r'->'),
        (TokenType.LPAREN, r'\('),
        (TokenType.RPAREN, r'\)'),
        (TokenType.LBRACKET, r'\['),
        (TokenType.RBRACKET, r'\]'),
        (TokenType.LBRACE, r'\{'),
        (TokenType.RBRACE, r'\}'),
        (TokenType.COMMA, r','),
        (TokenType.COLON, r':'),
        (TokenType.EQUALS, r'='),
        (TokenType.TILDE, r'~'),
        (TokenType.STRING, r'"[^"]*"'),
        (TokenType.BOOLEAN, r'\b(true|false)\b'),
        (TokenType.NUMBER, r'\d+\.?\d*'),
        (TokenType.IDENTIFIER, r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ]
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        
    def tokenize(self) -> List[Token]:
        """Tokenize the input text."""
        while self.pos < len(self.text):
            # Skip whitespace
            if self.text[self.pos].isspace():
                if self.text[self.pos] == '\n':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.pos += 1
                continue
            
            # Try to match a token
            matched = False
            for token_type, pattern in self.PATTERNS:
                regex = re.compile(pattern)
                match = regex.match(self.text, self.pos)
                if match:
                    value = match.group(0)
                    
                    # Skip comments
                    if token_type == TokenType.COMMENT:
                        self.pos = match.end()
                        self.column += len(value)
                        matched = True
                        break
                    
                    # Parse value
                    if token_type == TokenType.STRING:
                        value = value[1:-1]  # Remove quotes
                    elif token_type == TokenType.NUMBER:
                        value = float(value) if '.' in value else int(value)
                    elif token_type == TokenType.BOOLEAN:
                        value = value == 'true'
                    
                    token = Token(token_type, value, self.line, self.column)
                    self.tokens.append(token)
                    
                    self.pos = match.end()
                    self.column += len(match.group(0))
                    matched = True
                    break
            
            if not matched:
                raise SyntaxError(
                    f"Unexpected character '{self.text[self.pos]}' "
                    f"at line {self.line}, column {self.column}"
                )
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens


@dataclass
class ASTNode:
    """Base class for AST nodes."""
    
    def accept(self, visitor: 'Visitor') -> Any:
        """Accept a visitor (Visitor pattern)."""
        raise NotImplementedError(f"{type(self).__name__} must implement accept()")


@dataclass
class TaskNode(ASTNode):
    """A task invocation."""
    name: str
    params: Dict[str, Any]
    is_control: bool = False
    advisory_time_ms: Optional[int] = None  # ~ timing (soft constraint)
    enforced_min_ms: Optional[int] = None   # >= timing (hard constraint)
    line: int = 0
    column: int = 0
    task: Optional[Any] = None  # Filled by builder: the actual task instance
    
    def accept(self, visitor: 'Visitor') -> Any:
        return visitor.visit_task_node(self)


@dataclass
class SequenceNode(ASTNode):
    """A sequence of tasks."""
    tasks: List[ASTNode]
    
    def accept(self, visitor: 'Visitor') -> Any:
        return visitor.visit_sequence_node(self)


@dataclass
class ParallelNode(ASTNode):
    """Parallel execution."""
    tasks: List[ASTNode]
    split_strategy: Optional[str] = None
    merge_strategy: Optional[str] = None
    merge_params: Optional[Dict[str, str]] = None
    fork_task: Optional[Any] = None   # Filled by builder: ForkConnector
    merge_task: Optional[Any] = None  # Filled by builder: MergeConnector
    
    def accept(self, visitor: 'Visitor') -> Any:
        return visitor.visit_parallel_node(self)


@dataclass
class LoopNode(ASTNode):
    """A loop."""
    body: ASTNode
    advisory_time_ms: Optional[int] = None  # ~ timing per iteration
    enforced_min_ms: Optional[int] = None   # >= timing per iteration
    loop_task: Optional[Any] = None  # Filled by builder: LoopConnector
    
    def accept(self, visitor: 'Visitor') -> Any:
        return visitor.visit_loop_node(self)


class Visitor:
    """Base visitor for traversing AST nodes."""
    
    def visit_task_node(self, node: TaskNode) -> Any:
        """Visit a task node."""
        raise NotImplementedError()
    
    def visit_sequence_node(self, node: SequenceNode) -> Any:
        """Visit a sequence node."""
        raise NotImplementedError()
    
    def visit_parallel_node(self, node: ParallelNode) -> Any:
        """Visit a parallel node."""
        raise NotImplementedError()
    
    def visit_loop_node(self, node: LoopNode) -> Any:
        """Visit a loop node."""
        raise NotImplementedError()


class Parser:
    """Parses tokens into an AST."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    def current(self) -> Token:
        """Get current token."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else self.tokens[-1]
    
    def peek(self, offset: int = 1) -> Token:
        """Peek ahead."""
        pos = self.pos + offset
        return self.tokens[pos] if pos < len(self.tokens) else self.tokens[-1]
    
    def consume(self, expected_type: Optional[TokenType] = None) -> Token:
        """Consume and return current token."""
        token = self.current()
        if expected_type and token.type != expected_type:
            raise SyntaxError(
                f"Expected {expected_type.value}, got {token.type.value} "
                f"at line {token.line}, column {token.column}"
            )
        self.pos += 1
        return token
    
    def parse(self) -> ASTNode:
        """Parse the token stream into an AST."""
        return self.parse_pipeline()
    
    def parse_pipeline(self) -> ASTNode:
        """
        Parse top-level pipeline.
        
        Grammar:
            pipeline ::= task_sequence | loop_pipeline | parallel_pipeline | task
            task_sequence ::= (task | loop_pipeline | parallel_pipeline) ( "->" pipeline )+
        """
        # Parse first element
        token = self.current()
        
        if token.type == TokenType.LBRACE:
            first = self.parse_loop_pipeline()
        elif token.type == TokenType.LBRACKET:
            first = self.parse_parallel_pipeline()
        else:
            first = self.parse_task()
        
        # Check if this is a sequence (has arrows after first element)
        if self.current().type == TokenType.ARROW:
            elements = [first]
            while self.current().type == TokenType.ARROW:
                self.consume(TokenType.ARROW)
                # Recursively parse next element
                elements.append(self.parse_pipeline())
            return SequenceNode(elements)
        
        # Single element (not a sequence)
        return first
    
    def parse_task_sequence(self) -> ASTNode:
        """
        Legacy method - now handled by parse_pipeline.
        Kept for compatibility.
        """
        return self.parse_pipeline()
    
    def parse_task(self) -> ASTNode:
        """
        Parse a single task (used in contexts where loops/parallels not allowed).
        For general pipeline parsing, use parse_pipeline() instead.
        """
        token = self.current()
        
        # This should only be called for regular tasks now
        # Loops and parallels are handled by parse_pipeline
        if token.type == TokenType.LBRACE:
            raise SyntaxError(
                f"Unexpected loop at line {token.line} - loops must be parsed at pipeline level"
            )
        
        if token.type == TokenType.LBRACKET:
            raise SyntaxError(
                f"Unexpected parallel block at line {token.line} - parallel blocks must be parsed at pipeline level"
            )
        
        if token.type == TokenType.COLON:
            raise SyntaxError(
                f"Unexpected operator at line {token.line} - operators only allowed in loop context"
            )
        
        task_node = self.parse_regular_task()
        
        # Parse optional timing suffix
        advisory_ms, enforced_min_ms = self.parse_timing()
        task_node.advisory_time_ms = advisory_ms
        task_node.enforced_min_ms = enforced_min_ms
        
        return task_node
    
    def parse_operator(self) -> TaskNode:
        """
        Parse an operator (control task with both colons).
        
        Grammar:
            operator ::= ":" task ":"
        """
        self.consume(TokenType.COLON)
        task_node = self.parse_regular_task()
        self.consume(TokenType.COLON)
        
        # Mark as control task
        task_node.is_control = True
        
        # Parse optional timing suffix
        advisory_ms, enforced_min_ms = self.parse_timing()
        task_node.advisory_time_ms = advisory_ms
        task_node.enforced_min_ms = enforced_min_ms
        
        return task_node
    
    def parse_regular_task(self) -> TaskNode:
        """Parse a regular task with optional parameters."""
        name_token = self.consume(TokenType.IDENTIFIER)
        self.consume(TokenType.LPAREN)
        
        params = {}
        if self.current().type != TokenType.RPAREN:
            params = self.parse_params()
        
        self.consume(TokenType.RPAREN)
        
        return TaskNode(
            name=name_token.value,
            params=params,
            line=name_token.line,
            column=name_token.column
        )
    
    def parse_params(self) -> Dict[str, Any]:
        """Parse parameter list."""
        params = {}
        
        while True:
            name = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.EQUALS)
            value = self.parse_value()
            params[name] = value
            
            if self.current().type != TokenType.COMMA:
                break
            self.consume(TokenType.COMMA)
        
        return params
    
    def parse_value(self) -> Any:
        """Parse a parameter value."""
        token = self.current()
        
        if token.type == TokenType.STRING:
            return self.consume(TokenType.STRING).value
        elif token.type == TokenType.NUMBER:
            return self.consume(TokenType.NUMBER).value
        elif token.type == TokenType.BOOLEAN:
            return self.consume(TokenType.BOOLEAN).value
        else:
            raise SyntaxError(
                f"Expected value, got {token.type.value} "
                f"at line {token.line}, column {token.column}"
            )
    
    def parse_timing(self) -> tuple[Optional[int], Optional[int]]:
        """Parse optional timing suffix: ~100 or >=100."""
        advisory_ms = None
        enforced_min_ms = None
        
        if self.current().type == TokenType.TILDE:
            self.consume(TokenType.TILDE)
            time_value = self.consume(TokenType.NUMBER).value
            advisory_ms = int(time_value)
        elif self.current().type == TokenType.GREATER_EQUAL:
            self.consume(TokenType.GREATER_EQUAL)
            time_value = self.consume(TokenType.NUMBER).value
            enforced_min_ms = int(time_value)
        
        return advisory_ms, enforced_min_ms
    
    def parse_parallel_pipeline(self) -> ParallelNode:
        """
        Parse parallel execution block.
        
        Grammar:
            parallel_pipeline ::= "[" operator? pipeline_list operator? "]"
            pipeline_list ::= pipeline ( "," pipeline )*
            operator ::= ":" task ":"
        """
        self.consume(TokenType.LBRACKET)
        
        split_strategy = None
        merge_strategy = None
        merge_params = None
        
        # Parse optional leading operator (split_op): :identifier():
        if self.current().type == TokenType.COLON and \
           self.peek(1).type == TokenType.IDENTIFIER:
            self.consume(TokenType.COLON)
            split_strategy = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.LPAREN)
            self.consume(TokenType.RPAREN)
            self.consume(TokenType.COLON)
        
        # Parse pipeline_list: pipeline ( "," pipeline )*
        branches = [self.parse_pipeline()]
        while self.current().type == TokenType.COMMA:
            self.consume(TokenType.COMMA)
            branches.append(self.parse_pipeline())
        
        # Parse optional trailing operator (merge_op): :identifier(params?):
        if self.current().type == TokenType.COLON:
            self.consume(TokenType.COLON)
            merge_strategy = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.LPAREN)
            if self.current().type != TokenType.RPAREN:
                merge_params = self.parse_params()
            self.consume(TokenType.RPAREN)
            self.consume(TokenType.COLON)
        
        self.consume(TokenType.RBRACKET)
        
        return ParallelNode(
            tasks=branches,
            split_strategy=split_strategy,
            merge_strategy=merge_strategy,
            merge_params=merge_params
        )
    
    
    def parse_loop_pipeline(self) -> LoopNode:
        """
        Parse loop pipeline.
        
        Grammar:
            loop_pipeline ::= "{" loop_sequence "}"
        """
        self.consume(TokenType.LBRACE)
        
        if self.current().type == TokenType.RBRACE:
            raise SyntaxError(
                f"Empty loop body not allowed at line {self.current().line}"
            )
        
        body = self.parse_loop_sequence()
        self.consume(TokenType.RBRACE)
        
        # Parse optional timing suffix for loop
        advisory_ms, enforced_min_ms = self.parse_timing()
        
        return LoopNode(
            body=body,
            advisory_time_ms=advisory_ms,
            enforced_min_ms=enforced_min_ms
        )
    
    def parse_loop_sequence(self) -> ASTNode:
        """
        Parse loop sequence (allows operators and pipelines).
        
        Grammar:
            loop_sequence ::= loop_item ( "->" loop_item )*
            loop_item ::= operator | pipeline
            
        Note: pipeline here means task_sequence, loop_pipeline, or parallel_pipeline,
        but NOT a task_sequence that consumes arrows (we handle arrows at this level).
        """
        items = []
        
        # Parse first item
        if self.current().type == TokenType.COLON:
            items.append(self.parse_operator())
        else:
            # Parse a "pipeline" but not one that consumes arrows
            # We need to parse a single unit: task, loop, or parallel
            items.append(self.parse_loop_item_pipeline())
        
        # Parse remaining items
        while self.current().type == TokenType.ARROW:
            self.consume(TokenType.ARROW)
            if self.current().type == TokenType.COLON:
                items.append(self.parse_operator())
            else:
                items.append(self.parse_loop_item_pipeline())
        
        return SequenceNode(items) if len(items) > 1 else items[0]
    
    def parse_loop_item_pipeline(self) -> ASTNode:
        """
        Parse a pipeline item within a loop (no arrow consumption).
        Can be: task, loop_pipeline, or parallel_pipeline.
        """
        token = self.current()
        
        if token.type == TokenType.LBRACE:
            return self.parse_loop_pipeline()
        elif token.type == TokenType.LBRACKET:
            return self.parse_parallel_pipeline()
        else:
            # Just parse a single task (no sequence, no arrow consumption)
            return self.parse_task()


class PipelineBuilder:
    """Builds pipeline objects from AST."""
    
    def __init__(self, task_registry: Dict[str, type], pipeline_dirs: Optional[List[str]] = None):
        """
        Initialize builder.
        
        Args:
            task_registry: Dict mapping task names to task classes
            pipeline_dirs: Optional list of directories to search for .dsl files
        """
        self.task_registry = task_registry
        self.pipeline_dirs = pipeline_dirs or []
        self.task_counter = {}  # Track instances per task type for unique IDs
        self.task_instances = {}  # Cache task instances by explicit ID
        self.in_loop = False
        self._lazy_load_attempted = set()  # Track lazy load attempts to avoid repeated failures
    
    def _lazy_load_task(self, task_name: str) -> Optional[type]:
        """
        Attempt to lazy-load a task not in the initial registry.
        
        Searches for task modules matching naming conventions:
        - {task_name}_task.py in pipeline root
        - {task_name}_task.py in tasks/ subdirectory
        
        Args:
            task_name: Name of the task to load
            
        Returns:
            Task class if found and loaded, None otherwise
        """
        # Avoid repeated attempts for the same task
        if task_name in self._lazy_load_attempted:
            return None
        
        self._lazy_load_attempted.add(task_name)
        
        import importlib
        from pathlib import Path
        
        # Get pipeline directory
        pipeline_dir = Path(__file__).parent
        
        # Search patterns: module names to try
        search_patterns = [
            # Pattern 1: {task_name}_task in pipeline root
            ('vlmchat.pipeline', f'{task_name}_task'),
            # Pattern 2: {task_name}_task in tasks/ subdirectory
            ('vlmchat.pipeline_tasks', f'{task_name}_task'),
            # Pattern 3: Exact match (for special cases like 'pipeline')
            ('vlmchat.pipeline', task_name),
        ]
        
        for package, module_name in search_patterns:
            try:
                # Check if file exists before attempting import
                if package == 'vlmchat.pipeline':
                    module_file = pipeline_dir / f'{module_name}.py'
                else:  # vlmchat.pipeline_tasks
                    module_file = pipeline_dir / 'tasks' / f'{module_name}.py'
                
                if module_file.exists():
                    # Import the module
                    importlib.import_module(f'.{module_name}', package=package)
                    
                    # Check if task was registered during import
                    from .task_base import get_task_registry
                    registry = get_task_registry()
                    
                    if task_name in registry:
                        task_class = registry[task_name]
                        # Add to our registry for future lookups
                        self.task_registry[task_name] = task_class
                        logger.debug(f"Lazy-loaded task '{task_name}' from {module_file}")
                        return task_class
                        
            except ImportError as e:
                logger.debug(f"Failed to lazy-load '{task_name}' from {package}.{module_name}: {e}")
                continue
        
        return None
    
    def build(self, ast: ASTNode) -> ASTNode:
        """Build pipeline from AST by decorating it with task instances.
        
        Returns the decorated AST root node.
        """
        # First pass: decorate nodes with tasks and wire upstream_tasks
        self._decorate_node(ast)
        
        # Second pass: wire downstream_tasks (inverse of upstream_tasks)
        self._wire_downstream_tasks(ast)
        
        return ast
    
    def _wire_downstream_tasks(self, node: ASTNode) -> None:
        """Wire downstream_tasks for all tasks in the AST (second pass)."""
        # Collect all tasks
        all_tasks = self._collect_all_tasks_from_node(node)
        
        # For each task, populate downstream_tasks based on upstream_tasks
        for task in all_tasks:
            for upstream in task.upstream_tasks:
                if task not in upstream.downstream_tasks:
                    upstream.downstream_tasks.append(task)
    
    def _decorate_node(self, node: ASTNode) -> None:
        """Decorate an AST node with task instances and wire them together."""
        if isinstance(node, TaskNode):
            self._decorate_task(node)
        elif isinstance(node, SequenceNode):
            self._decorate_sequence(node)
        elif isinstance(node, ParallelNode):
            self._decorate_parallel(node)
        elif isinstance(node, LoopNode):
            self._decorate_loop(node)
        else:
            raise ValueError(f"Unknown node type: {type(node)}")
    
    def _decorate_task(self, node: TaskNode) -> None:
        """Create task instance and attach it to the node."""
        # Validate control tasks only in loops
        if node.is_control and not self.in_loop:
            raise SyntaxError(
                f"Control task ':{node.name}()' must be inside loop {{}} "
                f"at line {node.line}, column {node.column}"
            )
        
        # Look up task class or .dsl file
        task_class = None
        dsl_file_path = None
        
        if node.name in self.task_registry:
            task_class = self.task_registry[node.name]
        else:
            # Lazy load: try to find and import the task
            task_class = self._lazy_load_task(node.name)
        
        if not task_class and self.pipeline_dirs:
            # Search for {task_name}.dsl in pipeline directories
            import os
            for directory in self.pipeline_dirs:
                candidate_path = os.path.join(directory, f"{node.name}.dsl")
                if os.path.isfile(candidate_path):
                    dsl_file_path = candidate_path
                    # Use Pipeline task class
                    task_class = self.task_registry.get('pipeline')
                    if not task_class:
                        raise ValueError(
                            f"Pipeline task not registered but needed for '{node.name}.dsl' "
                            f"at line {node.line}, column {node.column}"
                        )
                    break
        
        if not task_class:
            raise ValueError(
                f"Unknown task '{node.name}' at line {node.line}, column {node.column}. "
                f"Not found in registry{' or pipeline directories: ' + ', '.join(self.pipeline_dirs) if self.pipeline_dirs else ''}"
            )
        
        # Check if explicit ID provided in DSL params
        if 'id' in node.params:
            unique_id = node.params['id']
            # Remove 'id' from params so it's not passed to configure()
            params_for_configure = {k: v for k, v in node.params.items() if k != 'id'}
            
            # Check if we already created this task instance
            if unique_id in self.task_instances:
                # Reuse existing instance - reconfigure with new params
                node.task = self.task_instances[unique_id]
                if node.task and hasattr(node.task, 'configure') and params_for_configure:
                    node.task.configure(**params_for_configure)
                return
        else:
            # Generate unique task ID using counter
            if node.name not in self.task_counter:
                self.task_counter[node.name] = 0
            self.task_counter[node.name] += 1
            unique_id = f"{node.name}_{self.task_counter[node.name]}"
            params_for_configure = node.params
        
        # Instantiate task
        try:
            # Try to instantiate with task_id parameter
            try:
                task = task_class(task_id=unique_id)
            except TypeError:
                # Fallback: create without task_id, then set it
                task = task_class()
                task.task_id = unique_id
            
            # If this is a Pipeline task loaded from .dsl file, configure it with the file path
            if dsl_file_path:
                # Pipeline task needs the DSL file path
                params_for_configure['file'] = dsl_file_path
            
            # Set timing constraints
            if node.advisory_time_ms:
                task.time_budget_ms = node.advisory_time_ms
            if node.enforced_min_ms:
                task._enforced_min_ms = node.enforced_min_ms
            
            # Configure with parameters if supported (excluding 'id')
            if hasattr(task, 'configure') and params_for_configure:
                task.configure(**params_for_configure)
            
            # Cache task instance if explicit ID was provided
            if 'id' in node.params:
                self.task_instances[unique_id] = task
                
                # Validate that all parameters have descriptions
                if hasattr(task, 'describe_parameters'):
                    param_descriptions = task.describe_parameters()
                    for param_name in params_for_configure.keys():
                        if param_name not in param_descriptions:
                            logger.warning(
                                f"Parameter '{param_name}' used in task '{node.name}' at line {node.line} "
                                f"has no description. Add it to describe_parameters() method."
                            )
                        elif not param_descriptions[param_name]:
                            # Empty string or empty dict
                            logger.warning(
                                f"Parameter '{param_name}' in task '{node.name}' at line {node.line} "
                                f"has empty description. Please provide documentation."
                            )
            
            # Attach task to node
            node.task = task
            
        except TypeError as e:
            # Task requires constructor arguments - try common patterns
            if 'timeout_seconds' in str(e) and 'seconds' in node.params:
                # TimeoutCondition needs timeout_seconds parameter
                task = task_class(timeout_seconds=node.params['seconds'])
                node.task = task
            else:
                raise ValueError(
                    f"Error creating task '{node.name}' at line {node.line}: {e}. "
                    f"Task may need constructor parameters that aren't provided."
                )
        except Exception as e:
            raise ValueError(
                f"Error creating task '{node.name}' at line {node.line}: {e}"
            )
    
    def _decorate_sequence(self, node: SequenceNode) -> None:
        """Decorate sequence node: create tasks and wire them sequentially."""
        # Decorate all child nodes first
        for child in node.tasks:
            self._decorate_node(child)
        
        # Wire tasks sequentially using upstream_tasks
        for i in range(1, len(node.tasks)):
            prev_node = node.tasks[i-1]
            curr_node = node.tasks[i]
            
            # Get the task to wire from (could be in TaskNode.task or ParallelNode.merge_task)
            prev_task = self._get_exit_task(prev_node)
            curr_task = self._get_entry_task(curr_node)
            
            if prev_task and curr_task:
                curr_task.upstream_tasks.append(prev_task)
    
    def _get_entry_task(self, node: ASTNode) -> Any:
        """Get the entry (first) task from a node."""
        if isinstance(node, TaskNode):
            return node.task
        elif isinstance(node, ParallelNode):
            return node.fork_task
        elif isinstance(node, LoopNode):
            return node.loop_task
        elif isinstance(node, SequenceNode):
            return self._get_entry_task(node.tasks[0]) if node.tasks else None
        return None
    
    def _get_exit_task(self, node: ASTNode) -> Any:
        """Get the exit (last) task from a node."""
        if isinstance(node, TaskNode):
            return node.task
        elif isinstance(node, ParallelNode):
            return node.merge_task
        elif isinstance(node, LoopNode):
            return node.loop_task
        elif isinstance(node, SequenceNode):
            return self._get_exit_task(node.tasks[-1]) if node.tasks else None
        return None
    
    def _decorate_parallel(self, node: ParallelNode) -> None:
        """Decorate parallel node: create fork/merge and wire branches."""
        from vlmchat.pipeline.fork_connector import ForkConnector
        from vlmchat.pipeline.task_base import Connector
        from vlmchat.pipeline.connectors import OrderedMergeConnector
        
        # Decorate all branch nodes first
        for branch_node in node.tasks:
            self._decorate_node(branch_node)
        
        # Get entry and exit tasks for each branch
        branch_entries = [self._get_entry_task(n) for n in node.tasks]
        branch_exits = [self._get_exit_task(n) for n in node.tasks]
        
        # Create fork connector
        fork = ForkConnector(task_id="fork", num_outputs=len(branch_entries))
        fork.output_tasks = branch_entries
        node.fork_task = fork
        
        # Wire fork to each branch entry
        for entry_task in branch_entries:
            entry_task.upstream_tasks.append(fork)
        
        # Create merge connector based on strategy
        if node.merge_strategy == "ordered_merge":
            merge = OrderedMergeConnector(task_id="merge")
            if node.merge_params:
                merge.configure(node.merge_params)  # type: ignore[call-arg]
        else:
            merge = Connector(task_id="merge")
        
        node.merge_task = merge
        
        # Wire branch exits to merge
        merge.upstream_tasks = branch_exits.copy()
        merge.input_tasks = branch_exits.copy()
    
    def _decorate_loop(self, node: LoopNode) -> None:
        """Decorate loop node: create LoopConnector with body tasks."""
        from vlmchat.pipeline.loop_connector import LoopConnector
        
        # Set loop context
        was_in_loop = self.in_loop
        self.in_loop = True
        
        try:
            # Decorate loop body
            self._decorate_node(node.body)
            
            # Collect all tasks from body
            body_tasks = self._collect_all_tasks_from_node(node.body)
            
            # Create loop connector with body tasks
            loop = LoopConnector(body_tasks=body_tasks)
            
            # Set timing constraints for loop
            if node.advisory_time_ms:
                loop.time_budget_ms = node.advisory_time_ms
            if node.enforced_min_ms:
                loop._enforced_min_ms = node.enforced_min_ms  # type: ignore[attr-defined]
            
            node.loop_task = loop
        finally:
            self.in_loop = was_in_loop
    
    def _collect_all_tasks_from_node(self, node: ASTNode) -> List[Any]:
        """Collect all task instances from a node tree."""
        tasks = []
        if isinstance(node, TaskNode):
            if node.task:
                tasks.append(node.task)
        elif isinstance(node, SequenceNode):
            for child in node.tasks:
                tasks.extend(self._collect_all_tasks_from_node(child))
        elif isinstance(node, ParallelNode):
            if node.fork_task:
                tasks.append(node.fork_task)
            for child in node.tasks:
                tasks.extend(self._collect_all_tasks_from_node(child))
            if node.merge_task:
                tasks.append(node.merge_task)
        elif isinstance(node, LoopNode):
            if node.loop_task:
                tasks.append(node.loop_task)
        return tasks


class DSLParser:
    """
    Main DSL parser interface.
    
    Usage:
        parser = DSLParser(task_registry)
        sources, pipeline = parser.parse(dsl_text)
        
        # With automatic pipeline loading:
        parser = DSLParser(task_registry, pipeline_dirs=['./pipelines', './dsl'])
        sources, pipeline = parser.parse('a->my_pipeline()->b')  # Looks for my_pipeline.dsl
        
        # With source declarations:
        dsl = '''
        @camera: camera_source(device=0, fps=30)
        @prompts: mqtt_source(topic="commands")
        
        wait(camera) -> latest(camera) -> yolo() -> display()
        '''
        sources, pipeline = parser.parse(dsl)
    """
    
    def __init__(self, task_registry: Dict[str, type], pipeline_dirs: Optional[List[str]] = None):
        """
        Initialize parser.
        
        Args:
            task_registry: Dict mapping task names to task classes
            pipeline_dirs: Optional list of directories to search for .dsl files
                          when a task name is not found in the registry
        """
        self.task_registry = task_registry
        self.pipeline_dirs = pipeline_dirs or []
        self.sources: Dict[str, Any] = {}  # name -> source instance
    
    def parse(self, text: str) -> Tuple[Dict[str, Any], Any]:
        """
        Parse DSL text into sources and pipeline.
        
        Args:
            text: DSL source code (may include @source: declarations)
            
        Returns:
            Tuple of (sources_dict, pipeline_root_task)
            
        Raises:
            SyntaxError: If DSL syntax is invalid
            ValueError: If task names or parameters are invalid
        """
        # Parse source declarations first
        pipeline_text, sources = self._parse_source_declarations(text)
        self.sources = sources
        
        # Tokenize remaining pipeline DSL
        lexer = Lexer(pipeline_text)
        tokens = lexer.tokenize()
        
        # Parse to AST
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Build pipeline
        builder = PipelineBuilder(self.task_registry, self.pipeline_dirs)
        pipeline = builder.build(ast)
        
        # Validate pipeline structure
        self._validate_pipeline(pipeline)
        
        return sources, pipeline
    
    def _parse_source_declarations(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse @name: source(params) declarations.
        
        Args:
            text: Full DSL text
            
        Returns:
            Tuple of (remaining_text, sources_dict)
        """
        import re
        
        sources = {}
        lines = text.split('\n')
        pipeline_lines = []
        
        # Pattern: @name: source_type(param=value, ...)
        source_pattern = re.compile(r'^\s*@(\w+)\s*:\s*(\w+)\s*\((.*)\)\s*$')
        
        for line in lines:
            match = source_pattern.match(line)
            if match:
                name = match.group(1)
                source_type = match.group(2)
                params_str = match.group(3)
                
                # Parse parameters
                params = self._parse_source_params(params_str)
                
                # Create source instance
                source = self._create_source(name, source_type, params)
                sources[name] = source
                
                logger.info(f"Parsed source declaration: @{name}: {source_type}({params})")
            else:
                # Not a source declaration, keep for pipeline parsing
                pipeline_lines.append(line)
        
        remaining_text = '\n'.join(pipeline_lines)
        return remaining_text, sources
    
    def _parse_source_params(self, params_str: str) -> Dict[str, Any]:
        """
        Parse parameter string from source declaration.
        
        Args:
            params_str: String like 'device=0, fps=30, buffer_size=300'
            
        Returns:
            Dict of parsed parameters
        """
        import re
        
        params = {}
        if not params_str.strip():
            return params
        
        # Split by commas not inside strings
        param_pattern = re.compile(r'(\w+)\s*=\s*([^,]+)')
        
        for match in param_pattern.finditer(params_str):
            key = match.group(1)
            value_str = match.group(2).strip()
            
            # Parse value
            if value_str.startswith('"') and value_str.endswith('"'):
                value = value_str[1:-1]  # String
            elif value_str.lower() in ('true', 'false'):
                value = value_str.lower() == 'true'  # Boolean
            elif '.' in value_str:
                try:
                    value = float(value_str)  # Float
                except ValueError:
                    value = value_str  # String fallback
            else:
                try:
                    value = int(value_str)  # Int
                except ValueError:
                    value = value_str  # String fallback
            
            params[key] = value
        
        return params
    
    def _create_source(self, name: str, source_type: str, params: Dict[str, Any]) -> Any:
        """
        Create a source instance.
        
        Args:
            name: Source name
            source_type: Source type (e.g., 'jetson_camera_source')
            params: Source parameters
            
        Returns:
            Source instance
        """
        # Map source types to (module, class) tuples
        source_registry = {
            'jetson_camera_source': ('vlmchat.pipeline.sources.jetson_camera', 'JetsonCameraSource'),
            'jetson_camera': ('vlmchat.pipeline.sources.jetson_camera', 'JetsonCameraSource'),  # Alias
        }
        
        if source_type not in source_registry:
            raise ValueError(f"Unknown source type: {source_type}. Available: {list(source_registry.keys())}")
        
        # Dynamic import - only import what we need
        module_path, class_name = source_registry[source_type]
        import importlib
        module = importlib.import_module(module_path)
        source_class = getattr(module, class_name)
        
        # Create source with name as first parameter
        return source_class(name=name, **params)
    
    def _validate_pipeline(self, pipeline: Any) -> None:
        """
        Validate pipeline structure for stream sources.
        
        Checks:
        - If pipeline uses wait(), it must be the first task
        - Only one wait() task per pipeline
        
        Args:
            pipeline: Pipeline root task
            
        Raises:
            ValueError: If validation fails
        """
        from .stream_tasks import WaitTask
        
        # Collect all tasks
        all_tasks = self._collect_all_tasks(pipeline)
        
        # Find wait tasks
        wait_tasks = [t for t in all_tasks if isinstance(t, WaitTask)]
        
        if not wait_tasks:
            # No wait tasks, no validation needed
            return
        
        if len(wait_tasks) > 1:
            raise ValueError(f"Only one wait() task allowed per pipeline, found {len(wait_tasks)}")
        
        # Check if wait is first task
        first_task = self._get_first_task(pipeline)
        if not isinstance(first_task, WaitTask):
            raise ValueError(f"wait() must be the first task in pipeline, found {type(first_task).__name__} instead")
    
    def _collect_all_tasks(self, root: Any) -> List[Any]:
        """Recursively collect all tasks from pipeline."""
        tasks = []
        
        if hasattr(root, 'task') and root.task:
            tasks.append(root.task)
        
        # Recurse into children
        if hasattr(root, 'tasks'):
            for child in root.tasks:
                tasks.extend(self._collect_all_tasks(child))
        
        if hasattr(root, 'body'):
            tasks.extend(self._collect_all_tasks(root.body))
        
        return tasks
    
    def _get_first_task(self, root: Any) -> Optional[Any]:
        """Get the first task in the pipeline."""
        if hasattr(root, 'task') and root.task:
            return root.task
        
        if hasattr(root, 'tasks') and root.tasks:
            return self._get_first_task(root.tasks[0])
        
        return None


def create_task_registry() -> Dict[str, type]:
    """
    Create initial registry with core tasks.
    
    Uses hybrid approach:
    - Core tasks (diagnostic, conditions, I/O) loaded upfront for fast access
    - Other tasks lazy-loaded on-demand by PipelineBuilder
    
    Tasks self-register using the @register_task decorator.
    
    Returns:
        Dict mapping task names to task classes
    """
    import importlib
    
    # Core tasks that should always be available
    core_modules = [
        # Essential infrastructure
        'diagnostic_task',
        'start_task',
        'pass_task',
        
        # Loop control
        'exit_code_condition',
        'diagnostic_condition',
        'timeout_task',
        
        # Error handling
        'tasks.error_handler_tasks',
        
        # Basic I/O (commonly used)
        'tasks.console_input_task',
        'tasks.console_output_task',
        'tasks.test_input_task',
        
        # Pipeline support
        'pipeline',
    ]
    
    # Import core modules
    for module_name in core_modules:
        try:
            importlib.import_module(f'.{module_name}', package='vlmchat.pipeline')
        except ImportError as e:
            logger.debug(f"Could not import core module {module_name}: {e}")
    
    # Get the registry populated by @register_task decorators
    from .task_base import get_task_registry
    registry = get_task_registry()
    
    # Stream tasks (manually register since they don't use @register_task decorator)
    try:
        from .stream_tasks import WaitTask, LatestTask, NextTask
        registry['wait'] = WaitTask
        registry['latest'] = LatestTask
        registry['next'] = NextTask
    except ImportError as e:
        logger.debug(f"Could not import stream tasks: {e}")
    
    return registry


# Example usage and testing
if __name__ == "__main__":
    # Simple lexer/parser tests without full task imports
    print("DSL Parser Test\n" + "="*50)
    
    test_cases = [
        # Simple sequence
        ("camera() -> detect() -> console_output()", "Simple sequence"),
        
        # With parameters
        ('detect(model="yolov8n", confidence=0.5) -> console_output()', "With parameters"),
        
        # Parallel
        ("[camera(), prompt_embeddings()] -> detect()", "Parallel tasks"),
        
        # Loop
        ("{camera() -> detect() ->:timeout(seconds=60)}", "Loop with timeout"),
        
        # Complex
        ("""{
            [camera(), prompt_embeddings()] ->
            detect(confidence=0.5) ->
            clip_vision() ->
            clip_compare() ->
            console_output() ->
            :timeout(seconds=300)
        }""", "Complex pipeline"),
    ]
    
    for i, (dsl, description) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description}")
        print(f"DSL: {dsl[:60]}{'...' if len(dsl) > 60 else ''}")
        try:
            # Test lexer
            lexer = Lexer(dsl)
            tokens = lexer.tokenize()
            print(f"  Tokens: {len(tokens)} (including EOF)")
            
            # Test parser
            parser = Parser(tokens)
            ast = parser.parse()
            print(f"  ✅ Parsed to AST: {type(ast).__name__}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "="*50)
    print("✅ All DSL parsing tests completed")
    print("\nTo use with actual tasks:")
    print("  from vlmchat.pipeline.dsl_parser import DSLParser, create_task_registry")
    print("  parser = DSLParser(create_task_registry())")
    print('  pipeline = parser.parse("camera() -> detect()")')
