"""
Console I/O wrapper used by the interactive chat loop.

All direct calls to input() and print() should go through this module so
the UI layer can be mocked or replaced for tests or alternate frontends.
"""
from typing import Iterable, Any
import json
import sys
from pathlib import Path

# Try relative imports when used as a package (normal case). If the module is
# executed directly (python src/main/console_io.py) the package context may not
# exist, so fall back to adding the repository's `src` directory to sys.path
# and import the package-style modules.
try:
    from .service_response import ServiceResponse as SR
    from .chat_application import SmolVLMChatApplication
except Exception:
    # Compute src/ directory (two levels up from this file: src/main/console_io.py)
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from main.service_response import ServiceResponse as SR
    from main.chat_application import SmolVLMChatApplication

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
    print("  /load_url <url>     - Load image from URL for conversation")
    print("  /load_file <path>   - Load image from local file path")
    print("  /clear_context      - Clear conversation history")
    print("  /show_context       - Display current conversation history")
    print("  /context_stats      - Show context buffer statistics")
    print("  /format <format>    - Change history format (xml|minimal)")
    print("  /camera             - Capture image from camera")
    print("  /help               - Show this help message")
    print("  /quit               - Exit the application")


def process_command(app, user_input: str) -> bool:
    """Dispatch a slash command against the provided application instance.

    Returns True to continue the interactive loop, False to exit.
    """
    # Quit handled as special case
    if user_input.startswith('/quit'):
        print_text("Goodbye!")
        return False

    if user_input.startswith('/help'):
        # Help is a UI operation; call app's print helper or fallback to
        # the console helper.
        try:
            app._print_help_message()
        except Exception:
            print_help_message()
        return True

    if user_input.startswith('/load_url '):
        url = user_input[len('/load_url '):].strip()
        resp = app._service_load_url(url)
        if resp.message:
            print_text(resp.message)
        return resp.code == SR.Code.OK

    if user_input.startswith('/load_file '):
        path = user_input[len('/load_file '):].strip()
        resp = app._service_load_file(path)
        if resp.message:
            print_text(resp.message)
        return resp.code == SR.Code.OK

    if user_input.startswith('/clear_context'):
        resp = app._service_clear_context()
        if resp.message:
            print_text(resp.message)
        return resp.code == SR.Code.OK

    if user_input.startswith('/show_context'):
        resp = app._service_show_context()
        if resp.message:
            print_text(resp.message)
        return resp.code == SR.Code.OK

    if user_input.startswith('/context_stats'):
        resp = app._service_context_stats()
        if resp.message:
            print_text(resp.message)
        return resp.code == SR.Code.OK

    if user_input.startswith('/format'):
        parts = user_input.split(maxsplit=1)
        arg = parts[1] if len(parts) > 1 else ''
        resp = app._service_format(arg)
        if resp.message:
            print_text(resp.message)
        return resp.code == SR.Code.OK

    if user_input.lower() == '/camera':
        resp = app._service_camera()
        if resp.message:
            print_text(resp.message)
        return resp.code == SR.Code.OK

    if user_input.startswith('/metrics'):
        resp = app._service_metrics()
        if resp.message:
            print_text(resp.message)
        return resp.code == SR.Code.OK

    if user_input.startswith('/backend'):
        parts = user_input.split()
        resp = app._service_backend(parts)
        if resp.message:
            print_text(resp.message)
        return resp.code == SR.Code.OK

    print_text("Unknown command. Type /help for a list of commands.")
    return True


def run_interactive_chat() -> None:
    """Interactive chat loop that drives the provided application instance.

    This function replaces the previous method on the application class and
    centralizes all I/O to the console_io module so it can be mocked in tests.
    """

    app = SmolVLMChatApplication()

    print_text("=== SmolVLM Interactive Chat ===")
    print_help_message()
    print_text("")

    while True:
        try:
            user_input = input_text("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.startswith('/'):
                cont = process_command(app, user_input)
                if not cont:
                    break
                continue

            # Regular query
            response = app.process_query(user_input)
            print_text(f"\nSmolVLM: {response}")

        except KeyboardInterrupt:
            print_text("\nGoodbye!")
            break
