"""
Task help formatter for generating documentation in various formats.

This module provides a standalone formatter that takes BaseTask instances
and generates formatted help documentation in console, markdown, HTML, or JSON.
"""

from typing import Any, Dict, Union
import json


class TaskHelpFormatter:
    """
    Standalone formatter for task documentation.
    
    Takes BaseTask instances and formats their describe*() output
    into various formats (console, markdown, HTML, JSON).
    """
    
    def format_console(self, task: 'BaseTask') -> str:
        """
        Format task help for console display.
        
        Args:
            task: BaseTask instance to format
            
        Returns:
            Formatted help text for console
        """
        lines = []
        lines.append(f"Task: {task.__class__.__name__}")
        lines.append(f"ID: {task.task_id}")
        lines.append("")
        
        # Description
        lines.append("Description:")
        lines.append(f"  {task.describe()}")
        lines.append("")
        
        # Contracts
        contracts = task.describe_contracts()
        
        if contracts['inputs']:
            lines.append("Inputs:")
            for data_type, expected_type in contracts['inputs'].items():
                type_name = self._format_type(expected_type)
                lines.append(f"  • {data_type.type_name}: {type_name}")
        else:
            lines.append("Inputs: None (source task)")
        lines.append("")
        
        if contracts['outputs']:
            lines.append("Outputs:")
            for data_type, expected_type in contracts['outputs'].items():
                type_name = self._format_type(expected_type)
                lines.append(f"  • {data_type.type_name}: {type_name}")
        else:
            lines.append("Outputs: None (sink task)")
        lines.append("")
        
        # Parameters
        params = task.describe_parameters()
        if params:
            lines.append("Parameters:")
            for param_name, param_info in params.items():
                lines.append(self._format_parameter_console(param_name, param_info))
        else:
            lines.append("Parameters: None")
        
        return "\n".join(lines)
    
    def format_markdown(self, task: 'BaseTask') -> str:
        """
        Format task help as Markdown.
        
        Args:
            task: BaseTask instance to format
            
        Returns:
            Formatted help text in Markdown
        """
        lines = []
        lines.append(f"# {task.__class__.__name__}")
        lines.append(f"**Task ID:** `{task.task_id}`")
        lines.append("")
        
        # Description
        lines.append("## Description")
        lines.append(task.describe())
        lines.append("")
        
        # Contracts
        contracts = task.describe_contracts()
        
        lines.append("## Inputs")
        if contracts['inputs']:
            for data_type, expected_type in contracts['inputs'].items():
                type_name = self._format_type(expected_type)
                lines.append(f"- **{data_type.type_name}**: `{type_name}`")
        else:
            lines.append("*None (source task)*")
        lines.append("")
        
        lines.append("## Outputs")
        if contracts['outputs']:
            for data_type, expected_type in contracts['outputs'].items():
                type_name = self._format_type(expected_type)
                lines.append(f"- **{data_type.type_name}**: `{type_name}`")
        else:
            lines.append("*None (sink task)*")
        lines.append("")
        
        # Parameters
        params = task.describe_parameters()
        if params:
            lines.append("## Parameters")
            for param_name, param_info in params.items():
                lines.append(self._format_parameter_markdown(param_name, param_info))
        else:
            lines.append("## Parameters")
            lines.append("*None*")
        
        return "\n".join(lines)
    
    def format_html(self, task: 'BaseTask') -> str:
        """
        Format task help as HTML.
        
        Args:
            task: BaseTask instance to format
            
        Returns:
            Formatted help text in HTML
        """
        lines = []
        lines.append("<div class='task-help'>")
        lines.append(f"<h2>{task.__class__.__name__}</h2>")
        lines.append(f"<p><strong>Task ID:</strong> <code>{task.task_id}</code></p>")
        
        # Description
        lines.append("<h3>Description</h3>")
        lines.append(f"<p>{task.describe()}</p>")
        
        # Contracts
        contracts = task.describe_contracts()
        
        lines.append("<h3>Inputs</h3>")
        if contracts['inputs']:
            lines.append("<ul>")
            for data_type, expected_type in contracts['inputs'].items():
                type_name = self._format_type(expected_type)
                lines.append(f"<li><strong>{data_type.type_name}</strong>: <code>{type_name}</code></li>")
            lines.append("</ul>")
        else:
            lines.append("<p><em>None (source task)</em></p>")
        
        lines.append("<h3>Outputs</h3>")
        if contracts['outputs']:
            lines.append("<ul>")
            for data_type, expected_type in contracts['outputs'].items():
                type_name = self._format_type(expected_type)
                lines.append(f"<li><strong>{data_type.type_name}</strong>: <code>{type_name}</code></li>")
            lines.append("</ul>")
        else:
            lines.append("<p><em>None (sink task)</em></p>")
        
        # Parameters
        params = task.describe_parameters()
        lines.append("<h3>Parameters</h3>")
        if params:
            lines.append("<ul>")
            for param_name, param_info in params.items():
                lines.append(self._format_parameter_html(param_name, param_info))
            lines.append("</ul>")
        else:
            lines.append("<p><em>None</em></p>")
        
        lines.append("</div>")
        return "\n".join(lines)
    
    def format_json(self, task: 'BaseTask') -> str:
        """
        Format task help as JSON string.
        
        Args:
            task: BaseTask instance to format
            
        Returns:
            Formatted help text as JSON string
        """
        contracts = task.describe_contracts()
        
        data = {
            "task_class": task.__class__.__name__,
            "task_id": task.task_id,
            "description": task.describe(),
            "inputs": {
                dt.type_name: self._format_type(typ)
                for dt, typ in contracts['inputs'].items()
            },
            "outputs": {
                dt.type_name: self._format_type(typ)
                for dt, typ in contracts['outputs'].items()
            },
            "parameters": task.describe_parameters()
        }
        
        return json.dumps(data, indent=2)
    
    def _format_type(self, typ: Any) -> str:
        """Format a type for display."""
        if hasattr(typ, '__name__'):
            return typ.__name__
        return str(typ)
    
    def _format_parameter_console(self, name: str, info: Union[str, Dict[str, Any]]) -> str:
        """Format a parameter for console display."""
        if isinstance(info, str):
            # Legacy simple string
            return f"  • {name}: {info}"
        
        # Structured dict
        parts = [f"  • {name}"]
        
        if 'description' in info:
            parts.append(f": {info['description']}")
        
        details = []
        if 'type' in info:
            details.append(f"type: {info['type']}")
        
        if 'choices' in info and info['choices']:
            choices_str = ', '.join(str(c) for c in info['choices'])
            details.append(f"choices: {choices_str}")
        
        if 'default' in info and info['default'] is not None:
            details.append(f"default: {info['default']}")
        
        if 'example' in info:
            details.append(f"example: {info['example']}")
        
        if 'format' in info:
            details.append(f"format: {info['format']}")
        
        if info.get('required'):
            details.append("REQUIRED")
        
        if details:
            parts.append(f" ({', '.join(details)})")
        
        return ''.join(parts)
    
    def _format_parameter_markdown(self, name: str, info: Union[str, Dict[str, Any]]) -> str:
        """Format a parameter for markdown display."""
        if isinstance(info, str):
            return f"- **`{name}`**: {info}"
        
        parts = [f"- **`{name}`**"]
        
        if 'description' in info:
            parts.append(f": {info['description']}")
        
        details = []
        if 'type' in info:
            details.append(f"*Type:* `{info['type']}`")
        
        if 'choices' in info and info['choices']:
            choices_str = ', '.join(f"`{c}`" for c in info['choices'])
            details.append(f"*Choices:* {choices_str}")
        
        if 'default' in info and info['default'] is not None:
            details.append(f"*Default:* `{info['default']}`")
        
        if 'example' in info:
            details.append(f"*Example:* `{info['example']}`")
        
        if 'format' in info:
            details.append(f"*Format:* `{info['format']}`")
        
        if info.get('required'):
            details.append("**REQUIRED**")
        
        if details:
            parts.append(f"  \n  {' | '.join(details)}")
        
        return ''.join(parts)
    
    def _format_parameter_html(self, name: str, info: Union[str, Dict[str, Any]]) -> str:
        """Format a parameter for HTML display."""
        if isinstance(info, str):
            return f"<li><strong><code>{name}</code></strong>: {info}</li>"
        
        parts = [f"<li><strong><code>{name}</code></strong>"]
        
        if 'description' in info:
            parts.append(f": {info['description']}")
        
        details = []
        if 'type' in info:
            details.append(f"<em>Type:</em> <code>{info['type']}</code>")
        
        if 'choices' in info and info['choices']:
            choices_html = ', '.join(f"<code>{c}</code>" for c in info['choices'])
            details.append(f"<em>Choices:</em> {choices_html}")
        
        if 'default' in info and info['default'] is not None:
            details.append(f"<em>Default:</em> <code>{info['default']}</code>")
        
        if 'example' in info:
            details.append(f"<em>Example:</em> <code>{info['example']}</code>")
        
        if 'format' in info:
            details.append(f"<em>Format:</em> <code>{info['format']}</code>")
        
        if info.get('required'):
            details.append("<strong>REQUIRED</strong>")
        
        if details:
            parts.append(f" ({' | '.join(details)})")
        
        parts.append("</li>")
        return ''.join(parts)
