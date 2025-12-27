"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
import os

# Add src to path if needed
if 'vlmchat' not in sys.modules:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vlmchat.pipeline.dsl.parser import DSLParser
from vlmchat.pipeline.core.factory import PipelineFactory
from vlmchat.pipeline.core.runner import PipelineRunner
from vlmchat.pipeline.core.task_base import Context
from vlmchat.utils.platform_detect import detect_platform, Platform


# ============================================================================
# Platform-aware test markers and skipping
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "jetson: Tests that require Jetson hardware"
    )
    config.addinivalue_line(
        "markers", "rpi: Tests that require Raspberry Pi hardware"
    )
    config.addinivalue_line(
        "markers", "mac: Tests that require macOS"
    )
    config.addinivalue_line(
        "markers", "gstreamer: Tests that require GStreamer"
    )
    config.addinivalue_line(
        "markers", "opencv: Tests that require OpenCV"
    )
    config.addinivalue_line(
        "markers", "cuda: Tests that require CUDA"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take >30 seconds"
    )
    config.addinivalue_line(
        "markers", "hardware: Tests that require real hardware (camera, etc.)"
    )
    config.addinivalue_line(
        "markers", "all_platforms: Tests that run on all platforms"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests based on platform.
    
    Default behavior:
    - Run tests marked 'all_platforms' or with no platform marker
    - Run tests marked for detected platform
    - Skip tests marked for other platforms
    """
    current_platform = detect_platform()
    
    for item in items:
        # Skip platform-specific tests on wrong platform
        if "jetson" in item.keywords and current_platform != Platform.JETSON:
            item.add_marker(pytest.mark.skip(reason="Requires Jetson hardware"))
        
        if "rpi" in item.keywords and current_platform != Platform.RPI:
            item.add_marker(pytest.mark.skip(reason="Requires Raspberry Pi hardware"))
        
        if "mac" in item.keywords and current_platform != Platform.MAC:
            item.add_marker(pytest.mark.skip(reason="Requires macOS"))
        
        # Skip tests requiring unavailable dependencies
        if "gstreamer" in item.keywords and not _has_gstreamer():
            item.add_marker(pytest.mark.skip(reason="GStreamer not available"))
        
        if "opencv" in item.keywords and not _has_opencv():
            item.add_marker(pytest.mark.skip(reason="OpenCV not available"))
        
        if "cuda" in item.keywords and not _has_cuda():
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))


def _has_gstreamer() -> bool:
    """Check if GStreamer is available."""
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        return True
    except:
        return False


def _has_opencv() -> bool:
    """Check if OpenCV is available."""
    try:
        import cv2
        return True
    except:
        return False


def _has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


# ============================================================================
# Platform fixtures
# ============================================================================

@pytest.fixture
def platform():
    """Get current platform."""
    return detect_platform()


# ============================================================================
# Buffer pool fixtures (for Jetson camera tests)
# ============================================================================

@pytest.fixture
def mock_buffer_pool():
    """Create a small buffer pool for testing."""
    from vlmchat.pipeline.sources.jetson_camera import BufferPool
    return BufferPool(num_buffers=5, width=100, height=100)


@pytest.fixture
def mock_pooled_buffer(mock_buffer_pool):
    """Get a single pooled buffer."""
    buf = mock_buffer_pool.acquire()
    yield buf
    if buf:
        mock_buffer_pool.release(buf)


# ============================================================================
# Pipeline fixtures
# ============================================================================


@pytest.fixture
def task_registry():
    """Create task registry with core tasks loaded."""
    from vlmchat.pipeline.core.task_base import get_task_registry
    return get_task_registry()


@pytest.fixture
def parser(task_registry):
    """Create DSL parser with task registry."""
    return DSLParser(task_registry)


@pytest.fixture
def fresh_context():
    """Create a fresh context for each test."""
    return Context()


@pytest.fixture
def run_pipeline(parser):
    """
    Helper fixture to parse DSL and run pipeline.
    
    Usage:
        result, runner = run_pipeline(dsl_string)
    """
    def _run(dsl: str, context: Context = None):
        factory = PipelineFactory(task_registry=parser.task_registry)
        sources, pipeline = parser.parse(dsl)
        runner = PipelineRunner(pipeline, enable_trace=True)
        for name, source in sources.items():
            runner.register_source(name, source)
        
        ctx = context or Context()
        result = runner.run(ctx)
        return result, runner
    
    return _run
