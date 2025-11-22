"""
Pipeline DSL Parser

Implements lexer, parser, and builder for the VLMChat Pipeline DSL.
Converts DSL text to executable pipeline objects.

Grammar:
    pipeline         ::= task_sequence | loop
    task_sequence    ::= task ( "->" task )*
    task             ::= regular_task | control_task | parallel | loop
    regular_task     ::= identifier "(" params? ")"
    control_task     ::= ":"? identifier "(" params? ")" ":"?
    parallel         ::= "[" parallel_body "]"
    parallel_body    ::= split_op? task_list merge_op?
    split_op         ::= identifier "():"
    merge_op         ::= ":" identifier "()"
    task_list        ::= task ( "," task )*
    loop             ::= "{" task_sequence "}"
    params           ::= param ( "," param )*
    param            ::= identifier "=" value
    value            ::= string | number | boolean
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


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
    pass


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


@dataclass
class SequenceNode(ASTNode):
    """A sequence of tasks."""
    tasks: List[ASTNode]


@dataclass
class ParallelNode(ASTNode):
    """Parallel execution."""
    tasks: List[ASTNode]
    split_strategy: Optional[str] = None
    merge_strategy: Optional[str] = None


@dataclass
class LoopNode(ASTNode):
    """A loop."""
    body: ASTNode
    advisory_time_ms: Optional[int] = None  # ~ timing per iteration
    enforced_min_ms: Optional[int] = None   # >= timing per iteration


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
        """Parse top-level pipeline."""
        return self.parse_task_sequence()
    
    def parse_task_sequence(self) -> ASTNode:
        """Parse a sequence of tasks connected by ->."""
        tasks = [self.parse_task()]
        
        while self.current().type == TokenType.ARROW:
            self.consume(TokenType.ARROW)
            # Skip redundant arrows adjacent to colons
            if self.current().type != TokenType.ARROW:
                tasks.append(self.parse_task())
        
        return SequenceNode(tasks) if len(tasks) > 1 else tasks[0]
    
    def parse_task(self) -> ASTNode:
        """Parse a single task (regular, control, parallel, or loop)."""
        token = self.current()
        
        # Loop
        if token.type == TokenType.LBRACE:
            return self.parse_loop()
        
        # Parallel
        if token.type == TokenType.LBRACKET:
            return self.parse_parallel()
        
        # Control or regular task
        is_control_prefix = token.type == TokenType.COLON
        if is_control_prefix:
            self.consume(TokenType.COLON)
        
        task_node = self.parse_regular_task()
        
        # Check for trailing colon
        is_control_suffix = self.current().type == TokenType.COLON
        if is_control_suffix:
            self.consume(TokenType.COLON)
        
        task_node.is_control = is_control_prefix or is_control_suffix
        
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
    
    def parse_parallel(self) -> ParallelNode:
        """Parse parallel execution block."""
        self.consume(TokenType.LBRACKET)
        
        split_strategy = None
        merge_strategy = None
        
        # Check for split strategy: identifier():
        if self.current().type == TokenType.IDENTIFIER and \
           self.peek(1).type == TokenType.LPAREN and \
           self.peek(2).type == TokenType.RPAREN and \
           self.peek(3).type == TokenType.COLON:
            split_strategy = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.LPAREN)
            self.consume(TokenType.RPAREN)
            self.consume(TokenType.COLON)
        
        # Parse task list
        tasks = [self.parse_task()]
        while self.current().type == TokenType.COMMA:
            self.consume(TokenType.COMMA)
            
            # Check for merge strategy: :identifier()
            if self.current().type == TokenType.COLON and \
               self.peek(1).type == TokenType.IDENTIFIER and \
               self.peek(2).type == TokenType.LPAREN:
                self.consume(TokenType.COLON)
                merge_strategy = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.LPAREN)
                self.consume(TokenType.RPAREN)
                break
            
            tasks.append(self.parse_task())
        
        # Check for merge strategy at end: :identifier()
        if merge_strategy is None and self.current().type == TokenType.COLON:
            self.consume(TokenType.COLON)
            merge_strategy = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.LPAREN)
            self.consume(TokenType.RPAREN)
        
        self.consume(TokenType.RBRACKET)
        
        return ParallelNode(
            tasks=tasks,
            split_strategy=split_strategy,
            merge_strategy=merge_strategy
        )
    
    def parse_loop(self) -> LoopNode:
        """Parse loop block."""
        self.consume(TokenType.LBRACE)
        
        if self.current().type == TokenType.RBRACE:
            raise SyntaxError(
                f"Empty loop body not allowed at line {self.current().line}"
            )
        
        body = self.parse_task_sequence()
        self.consume(TokenType.RBRACE)
        
        # Parse optional timing suffix for loop
        advisory_ms, enforced_min_ms = self.parse_timing()
        
        return LoopNode(
            body=body,
            advisory_time_ms=advisory_ms,
            enforced_min_ms=enforced_min_ms
        )


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
        self.in_loop = False
    
    def build(self, ast: ASTNode) -> Any:
        """Build pipeline from AST."""
        return self._build_node(ast)
    
    def _build_node(self, node: ASTNode) -> Any:
        """Build a node into pipeline objects."""
        if isinstance(node, TaskNode):
            return self._build_task(node)
        elif isinstance(node, SequenceNode):
            return self._build_sequence(node)
        elif isinstance(node, ParallelNode):
            return self._build_parallel(node)
        elif isinstance(node, LoopNode):
            return self._build_loop(node)
        else:
            raise ValueError(f"Unknown node type: {type(node)}")
    
    def _build_task(self, node: TaskNode) -> Any:
        """Build a task instance."""
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
        elif self.pipeline_dirs:
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
            return task
        except TypeError as e:
            # Task requires constructor arguments - try common patterns
            if 'timeout_seconds' in str(e) and 'seconds' in node.params:
                # TimeoutCondition needs timeout_seconds parameter
                task = task_class(timeout_seconds=node.params['seconds'])
                return task
            else:
                raise ValueError(
                    f"Error creating task '{node.name}' at line {node.line}: {e}. "
                    f"Task may need constructor parameters that aren't provided."
                )
        except Exception as e:
            raise ValueError(
                f"Error creating task '{node.name}' at line {node.line}: {e}"
            )
    
    def _build_sequence(self, node: SequenceNode) -> Any:
        """Build a sequence of tasks with proper wiring."""
        from src.pipeline.task_base import BaseTask, Connector
        
        tasks = []
        for task_node in node.tasks:
            task = self._build_node(task_node)
            if isinstance(task, list):
                tasks.extend(task)
            else:
                tasks.append(task)
        
        # Wire tasks sequentially by setting upstream_tasks
        for i in range(1, len(tasks)):
            tasks[i].upstream_tasks.append(tasks[i-1])
        
        # Return natural structure: single task or list of tasks
        return tasks[0] if len(tasks) == 1 else tasks
    
    def _build_parallel(self, node: ParallelNode) -> Any:
        """Build parallel execution with proper wiring."""
        from src.pipeline.fork_connector import ForkConnector
        from src.pipeline.task_base import Connector
        
        # Build parallel tasks
        parallel_tasks = [self._build_node(task) for task in node.tasks]
        
        # Create fork connector with number of branches
        fork = ForkConnector(task_id="fork", num_outputs=len(parallel_tasks))
        
        # Wire parallel tasks: each has fork as upstream
        for task in parallel_tasks:
            task.upstream_tasks.append(fork)
        
        # Set fork's output tasks
        fork.output_tasks = parallel_tasks
        
        # Create merge connector
        merge = Connector(task_id="merge")
        
        # Wire merge: all parallel tasks as upstreams
        merge.upstream_tasks = parallel_tasks.copy()
        
        # Set merge's input tasks
        merge.input_tasks = parallel_tasks.copy()
        
        # Return natural structure: list of [fork, tasks..., merge]
        return [fork, *parallel_tasks, merge]
    
    def _build_loop(self, node: LoopNode) -> Any:
        """Build a loop."""
        from src.pipeline.loop_connector import LoopConnector
        
        # Set loop context
        was_in_loop = self.in_loop
        self.in_loop = True
        
        try:
            # Build loop body
            body = self._build_node(node.body)
            
            # Ensure body is a list of tasks
            if not isinstance(body, list):
                body_tasks = [body]
            else:
                body_tasks = body
            
            # Create loop connector with body tasks
            loop = LoopConnector(body_tasks=body_tasks)
            
            # Set timing constraints for loop
            if node.advisory_time_ms:
                loop.time_budget_ms = node.advisory_time_ms
            if node.enforced_min_ms:
                loop._enforced_min_ms = node.enforced_min_ms
            
            return loop
        finally:
            self.in_loop = was_in_loop


class DSLParser:
    """
    Main DSL parser interface.
    
    Usage:
        parser = DSLParser(task_registry)
        pipeline = parser.parse(dsl_text)
        
        # With automatic pipeline loading:
        parser = DSLParser(task_registry, pipeline_dirs=['./pipelines', './dsl'])
        pipeline = parser.parse('a->my_pipeline()->b')  # Looks for my_pipeline.dsl
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
    
    def parse(self, text: str) -> Any:
        """
        Parse DSL text into a pipeline.
        
        Args:
            text: DSL source code
            
        Returns:
            Pipeline root task
            
        Raises:
            SyntaxError: If DSL syntax is invalid
            ValueError: If task names or parameters are invalid
        """
        # Tokenize
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        
        # Parse to AST
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Build pipeline
        builder = PipelineBuilder(self.task_registry, self.pipeline_dirs)
        pipeline = builder.build(ast)
        
        return pipeline


def create_task_registry() -> Dict[str, type]:
    """
    Create a registry of all available tasks.
    
    Tasks self-register using the @register_task decorator.
    This function triggers imports to ensure all tasks are registered.
    
    Returns:
        Dict mapping task names to task classes
    """
    # Import all task modules to trigger registration
    # Tasks in pipeline root
    try:
        from . import diagnostic_task
        from . import diagnostic_condition
        from . import start_task
        from . import pass_task
        from . import context_cleanup_task
        from . import detector_task
        from . import smolvlm_task
        from . import timeout_task
        from . import pipeline
    except ImportError as e:
        # Some pipeline tasks may not be available
        pass
    
    # Tasks in tasks/ subdirectory - import individually to handle failures
    try:
        from .tasks import camera_task
    except ImportError:
        pass
    try:
        from .tasks import clip_compare_task
    except ImportError:
        pass
    try:
        from .tasks import clip_text_encoder_task
    except ImportError:
        pass
    try:
        from .tasks import clip_vision_task
    except ImportError:
        pass
    try:
        from .tasks import color_enhance_task
    except ImportError:
        pass
    try:
        from .tasks import color_swap_task
    except ImportError:
        pass
    try:
        from .tasks import console_input_task
    except ImportError:
        pass
    try:
        from .tasks import console_output_task
    except ImportError:
        pass
    try:
        from .tasks import detection_expander_task
    except ImportError:
        pass
    try:
        from .tasks import fashion_clip_text_encoder_task
    except ImportError:
        pass
    try:
        from .tasks import fashion_clip_vision_task
    except ImportError:
        pass
    try:
        from .tasks import history_update_task
    except ImportError:
        pass
    try:
        from .tasks import prompt_embedding_source_task
    except ImportError:
        pass
    try:
        from .tasks import prompt_similarity_compare_task
    except ImportError:
        pass
    
    # Get the registry populated by @register_task decorators
    from .task_base import get_task_registry
    return get_task_registry()


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
    print("  from src.pipeline.dsl_parser import DSLParser, create_task_registry")
    print("  parser = DSLParser(create_task_registry())")
    print('  pipeline = parser.parse("camera() -> detect()")')
