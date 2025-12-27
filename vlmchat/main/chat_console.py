"""
Console I/O wrapper used by the interactive chat loop.

All direct calls to input() and print() should go through this module so
the UI layer can be mocked or replaced for tests or alternate frontends.
"""
from typing import Iterable, Any
import json
import sys
from pathlib import Path
import time
import threading
import readline
import os
import atexit

from metrics.metrics_collector import Collector, Session
from utils.config import VLMChatConfig

# Import instruments so they register themselves in Instrument._registry
from metrics import instruments  # noqa: F401

# Configure persistent command history
histfile = os.path.join(os.path.expanduser("~"), ".vlmchathistory")
try:
    readline.read_history_file(histfile)
    readline.set_history_length(1000)
except FileNotFoundError:
    pass

# Save history on exit
atexit.register(readline.write_history_file, histfile)

# Try relative imports when used as a package (normal case). If the module is
# executed directly (python src/main/console_io.py) the package context may not
# exist, so fall back to adding the repository's `src` directory to sys.path
# and import the package-style modules.

from .service_response import ServiceResponse as SR
from .chat_services import VLMChatServices


def input_text(prompt: str = "") -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""

def print_text(msg: str = "") -> None:
    print(msg)

def print_lines(lines: Iterable[str]) -> None:
    for ln in lines:
        print(ln)

def print_json(obj: Any) -> None:
    try:
        print(json.dumps(obj, indent=2))
    except Exception:
        print(str(obj))


def print_help_message() -> None:
    """Print the interactive help menu using plain print()."""
    print("Available Commands:")
    print("  /pipeline <file.dsl>  - Load pipeline from .dsl file")
    print("  /run [key=value...]   - Execute loaded pipeline with optional overrides")
    print("  /stop                 - Stop running pipeline")
    print("  /status               - Check pipeline execution status")
    print("  /log <level>          - Set logging level (DEBUG, INFO, WARNING, ERROR)")
    print("  /trace                - Show execution trace of last pipeline run")
    print("  /describe <task>      - Show help for a task type")
    print("  /clear_env            - Clear all environment data")
    print("  /show_env             - Show all environment keys and types")
    print("  /metrics              - Show metrics collector status")
    print("  /wait <seconds>       - Wait the given number of seconds before accepting input")
    print("  /echo <text>          - Echo the given text back to the console")
    print("  /help                 - Show this help message")
    print("  /quit                 - Exit the application")
    print("")
    print("Note: Multi-line DSL must be in a .dsl file (e.g., /pipeline pipelines/smolvlm_chat.dsl)")
    print("      Use the pipelines/ directory for your DSL files.")


def process_command(app: VLMChatServices, user_input: str) -> bool:
    """Dispatch a slash command against the provided application instance.

    Returns True to continue the interactive loop, False to exit.
    """
    # Ignore comment lines starting with '#'. This allows users to paste or
    # include comment lines in scripted input; they are treated as no-ops.
    if user_input.strip().startswith('#'):
        return True
    # Quit handled as special case
    if user_input.startswith('/quit'):
        print_text("Goodbye!")
        return False

    if user_input.startswith('/help'):
        print_help_message()
        return True

    if user_input.startswith('/clear_env'):
        resp = app._service_clear_environment()
        if resp.message:
            print_text(resp.message)
        return resp.code.value == SR.Code.OK.value

    if user_input.startswith('/show_env'):
        resp = app._service_show_environment()
        if resp.message:
            print_text(resp.message)
        return resp.code.value == SR.Code.OK.value

    if user_input.startswith('/metrics'):
        # Metrics command is handled in run_interactive_chat with session access
        return True

    if user_input.startswith('/wait'):
        parts = user_input.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            print_text("Usage: /wait <seconds>")
            return True
        arg = parts[1].strip()
        try:
            secs = float(arg)
            if secs < 0:
                raise ValueError("negative")
        except Exception:
            print_text("Invalid number of seconds. Provide a non-negative number, e.g. /wait 2.5")
            return True
        # Inform the user and sleep; this blocks the console until done.
        print_text(f"Waiting for {secs} seconds...")
        try:
            time.sleep(secs)
        except KeyboardInterrupt:
            print_text("Wait interrupted.")
        return True

    if user_input.startswith('/echo'):
        # Echo the remainder of the line (allow multiple words)
        parts = user_input.split(maxsplit=1)
        text = parts[1] if len(parts) > 1 else ''
        print_text(text)
        return True

    if user_input.startswith('/pipeline'):
        parts = user_input.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            print_text("Usage: /pipeline <dsl_or_file>")
            return True
        
        dsl_or_file = parts[1].strip()
        
        resp = app._service_pipeline(dsl_or_file)
        if resp.message:
            print_text(resp.message)
        return resp.code.value == SR.Code.OK.value

    if user_input.startswith('/run'):
        parts = user_input.split(maxsplit=1)
        overrides = parts[1] if len(parts) > 1 else ''
        resp = app._service_run(overrides.strip())
        if resp.message:
            print_text(resp.message)
        # Give pipeline thread a moment to start and set is_running()
        time.sleep(0.2)
        return resp.code.value == SR.Code.OK.value

    if user_input.startswith('/stop'):
        resp = app._service_stop()
        if resp.message:
            print_text(resp.message)
        return resp.code.value == SR.Code.OK.value

    if user_input.startswith('/status'):
        resp = app._service_status()
        if resp.message:
            print_text(resp.message)
        return resp.code.value == SR.Code.OK.value
    
    if user_input.startswith('/describe'):
        parts = user_input.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            print_text("Usage: /describe <task_name>")
            return True
        task_name = parts[1].strip()
        resp = app._service_describe(task_name)
        if resp.message:
            print_text(resp.message)
        return resp.code.value == SR.Code.OK.value
    
    if user_input.startswith('/trace'):
        resp = app._service_trace()
        if resp.message:
            print_text(resp.message)
        return resp.code.value == SR.Code.OK.value
    
    if user_input.startswith('/log'):
        parts = user_input.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            print_text("Usage: /log <level>")
            print_text("Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL")
            return True
        level_name = parts[1].strip().upper()
        import logging
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        if level_name not in level_map:
            print_text(f"Unknown log level: {level_name}")
            print_text("Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL")
            return True
        logging.getLogger().setLevel(level_map[level_name])
        print_text(f"Log level set to {level_name}")
        return True

    print_text("Unknown command. Type /help for a list of commands.")
    return True


def display_metrics(collector: Collector, session: Session) -> None:
    """Display metrics session information with instrument values."""
    lines = ["=== Metrics Collector Status ==="]
    lines.append(f"Collector name: {collector.name}")
    lines.append(f"Timeseries count: {len(collector._ts_map)}")
    
    # List registered timeseries
    if collector._ts_map:
        lines.append("\nRegistered Timeseries:")
        for name, ts in collector._ts_map.items():
            point_count = len(ts._points)
            lines.append(f"  - {name} ({point_count} points)")
    
    # Session info
    lines.append(f"\nActive sessions: {len(collector._sessions)}")
    
    # Display session details and instruments
    lines.append(f"\n--- Session ---")
    lines.append(f"Start time: {session.start_time:.2f}")
    if session.end_time:
        duration = session.end_time - session.start_time
        lines.append(f"End time: {session.end_time:.2f} (duration: {duration:.2f}s)")
    else:
        lines.append("Status: Active")
    
    # Display instruments and their current values
    if session._instruments:
        lines.append(f"\nInstruments ({len(session._instruments)}):")
        for ts_name, instrument in session._instruments:
            lines.append(f"\n  Timeseries: {ts_name}")
            lines.append(f"  Instrument: {instrument.__class__.__name__} - {instrument.name}")
            
            # Export instrument state to show current values
            try:
                state = instrument.export()
                # Remove metadata fields to focus on values
                display_fields = {k: v for k, v in state.items() 
                                if k not in ['type', 'name', 'binding_keys']}
                if display_fields:
                    for key, value in display_fields.items():
                        if isinstance(value, float):
                            lines.append(f"    {key}: {value:.4f}")
                        elif isinstance(value, dict):
                            lines.append(f"    {key}:")
                            for sub_key, sub_val in value.items():
                                if isinstance(sub_val, dict):
                                    # For nested dicts (like buckets)
                                    lines.append(f"      {sub_key}:")
                                    for k, v in sub_val.items():
                                        if isinstance(v, float):
                                            lines.append(f"        {k}: {v:.4f}")
                                        else:
                                            lines.append(f"        {k}: {v}")
                                else:
                                    lines.append(f"      {sub_key}: {sub_val}")
                        else:
                            lines.append(f"    {key}: {value}")
            except Exception as e:
                lines.append(f"    Error exporting instrument state: {e}")
    else:
        lines.append("\nNo instruments configured")
    
    print_text("\n".join(lines))


def run_interactive_chat(config: VLMChatConfig, collector: Collector) -> None:
    """Interactive pipeline execution loop with queue-based I/O coordination.

    Uses a background thread to drain output queue in real-time while the
    main thread handles user input. When pipeline is running, non-command
    input is forwarded to the pipeline's input queue.
    """

    # Create metrics session - it will observe the collector
    session = Session(collector)
    
    # Load instruments from metrics.json if it exists
    metrics_file = Path(__file__).parent.parent.parent / "metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            print_text(f"Loading {len(metrics_data)} metric configurations from {metrics_file.name}...")
            session.load_instruments(metrics_data)
            print_text(f"Successfully loaded instruments. Total: {len(session._instruments)}")
        except Exception as e:
            import traceback
            print_text(f"Error loading metrics from {metrics_file}: {e}")
            print_text(traceback.format_exc())
    else:
        print_text(f"Metrics file not found: {metrics_file}")
    
    session.start()

    try:
        app = VLMChatServices(config, collector)
        console_state = {'is_running': False}  # Use dict so thread can modify
        output_thread = None
        stop_output_thread = threading.Event()

        def output_drainer():
            """Background thread that continuously drains output queue."""
            runner = getattr(app, '_pipeline_runner', None)
            if not runner:
                return
            
            while not stop_output_thread.is_set():
                output = runner.get_output(timeout=0.1)
                if output is not None:
                    print_text(output)
                elif not runner.is_running():
                    # Pipeline stopped, drain any remaining output
                    while True:
                        output = runner.get_output(timeout=0.1)
                        if output is None:
                            break
                        print_text(output)
                    # Signal that pipeline finished
                    console_state['is_running'] = False
                    break

        print_text("=== VLMChat Pipeline Executor ===")
        print_help_message()
        print_text("")
        
        import colorama
        from colorama import Fore, Style

        while True:
            try:
                # Show appropriate prompt based on console state
                # Flashing prompt for running state: \033[5m makes it blink, \033[0m resets
                if console_state['is_running']:
                    prompt = f"\n{Fore.RED}>{Style.RESET_ALL} "
                else:
                    prompt = "\n> "
                user_input = input_text(prompt).strip()

                if not user_input:
                    # Empty input - if running, forward it (triggers break_on)
                    if console_state['is_running']:
                        runner = getattr(app, '_pipeline_runner', None)
                        if runner:
                            runner.send_input(user_input)
                    continue

                # Allow comment lines beginning with '#' to be ignored.
                if user_input.startswith('#'):
                    continue

                if user_input.startswith('/'):
                    # Handle /metrics specially - needs session access
                    if user_input.startswith('/metrics'):
                        display_metrics(collector, session)
                        continue
                    
                    # Handle /run specially - starts console running state
                    if user_input.startswith('/run'):
                        cont = process_command(app, user_input)
                        if not cont:
                            break
                        
                        # Check if pipeline actually started
                        runner = getattr(app, '_pipeline_runner', None)
                        if runner and runner.is_running():
                            console_state['is_running'] = True
                            # Start output drainer thread
                            stop_output_thread.clear()
                            output_thread = threading.Thread(target=output_drainer, daemon=True)
                            output_thread.start()
                        continue
                    
                    # Handle /stop - exits running state
                    elif user_input.startswith('/stop'):
                        cont = process_command(app, user_input)
                        if not cont:
                            break
                        console_state['is_running'] = False
                        if output_thread:
                            stop_output_thread.set()
                            output_thread.join(timeout=1.0)
                        continue
                    
                    # All other commands work normally
                    else:
                        cont = process_command(app, user_input)
                        if not cont:
                            break
                        continue

                # Non-command input
                if console_state['is_running']:
                    # Forward to pipeline
                    runner = getattr(app, '_pipeline_runner', None)
                    if runner:
                        runner.send_input(user_input)
                else:
                    # No pipeline running
                    print_text("Unknown input. Commands start with '/'. Type /help for available commands.")

            except KeyboardInterrupt:
                print_text("\nGoodbye!")
                if output_thread:
                    stop_output_thread.set()
                    output_thread.join(timeout=1.0)
                break
    finally:
        # Stop the metrics session
        session.stop()
