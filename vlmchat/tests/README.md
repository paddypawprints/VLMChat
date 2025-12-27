# Tests Module

This module contains unit tests and test utilities for the VLMChat application components.

## Purpose

The tests module provides comprehensive testing coverage for the VLMChat application, ensuring:
- Component functionality and reliability
- Integration between modules
- Error handling and edge cases
- Performance and memory usage
- Compatibility across different environments

## Test Structure

The tests are organized to mirror the main source code structure:
- **Model Tests**: Testing SmolVLM model wrapper and ONNX integration
- **Prompt Tests**: Testing conversation history and formatting
- **Service Tests**: Testing RAG service and metadata retrieval
- **Utility Tests**: Testing image processing and camera interface
- **Integration Tests**: Testing component interactions

## Test Categories

### Unit Tests
Individual component testing in isolation:
- Model configuration and initialization
- History management and formatting
- Image loading and processing
- Camera capture functionality

### Integration Tests
Testing component interactions:
- Chat application workflow
- Model and prompt integration
- Image loading and model inference
- Camera capture and image processing

### Performance Tests
Testing efficiency and resource usage:
- Model inference speed
- Memory consumption
- ONNX vs transformers performance
- Image processing benchmarks

### Error Handling Tests
Testing robustness and error recovery:
- Invalid image URLs and files
- Model loading failures
- Network connectivity issues
- Hardware unavailability

## Running Tests

### Basic Test Execution
```bash
# Run all tests
python -m pytest src/tests/

# Run specific test module
python -m pytest src/tests/test_model.py

# Run with verbose output
python -m pytest -v src/tests/
```

### Test Categories
```bash
# Run only unit tests
python -m pytest -m unit src/tests/

# Run only integration tests
python -m pytest -m integration src/tests/

# Run performance tests
python -m pytest -m performance src/tests/
```

### Coverage Analysis
```bash
# Generate coverage report
python -m pytest --cov=src src/tests/

# Generate HTML coverage report
python -m pytest --cov=src --cov-report=html src/tests/
```

## Test Configuration

### Environment Setup
Tests require specific environment configuration:
- Mock objects for hardware dependencies
- Test data for image processing
- Temporary directories for file operations
- Network mocking for URL testing

### Test Data
The tests module includes:
- Sample images for vision model testing
- Mock conversation histories
- Test configuration files
- Hardware simulation data

### Fixtures and Mocks
Common test utilities:
- Model configuration fixtures
- Image loading mocks
- Camera interface simulation
- Network request mocking

## Test Examples

### Model Testing
```python
def test_model_initialization():
    """Test SmolVLM model initialization."""
    config = ModelConfig(model_path="test_model")
    model = SmolVLMModel(config, use_onnx=False)
    assert model.config.model_path == "test_model"

def test_onnx_fallback():
    """Test automatic fallback to transformers."""
    # Test ONNX failure graceful handling
    pass
```

### History Testing
```python
def test_history_management():
    """Test conversation history limits."""
    history = History(max_pairs=2)
    history.add_conversation_pair("Q1", "A1")
    history.add_conversation_pair("Q2", "A2")
    history.add_conversation_pair("Q3", "A3")  # Should evict Q1/A1
    assert len(history._pairs) == 2

def test_format_switching():
    """Test runtime format changes."""
    history = History(history_format=HistoryFormat.XML)
    xml_output = history.get_formatted_history()

    history.set_format(HistoryFormat.MINIMAL)
    minimal_output = history.get_formatted_history()

    assert xml_output != minimal_output
```

### Image Processing Testing
```python
def test_image_url_loading():
    """Test loading images from URLs."""
    with mock_requests():
        image = load_image_from_url("http://test.com/image.jpg")
        assert image is not None
        assert image.mode == "RGB"

def test_invalid_url_handling():
    """Test graceful handling of invalid URLs."""
    image = load_image_from_url("invalid_url")
    assert image is None
```

## Continuous Integration

### GitHub Actions
Automated testing on:
- Multiple Python versions
- Different operating systems
- Various dependency versions

### Test Pipeline
1. **Linting**: Code quality checks
2. **Unit Tests**: Component testing
3. **Integration Tests**: System testing
4. **Performance Tests**: Benchmark validation
5. **Coverage**: Test coverage analysis

### Quality Gates
Tests must pass before code integration:
- All unit tests passing
- Integration tests successful
- Minimum coverage threshold met
- No regression in performance

## Dependencies

### Testing Framework
- **pytest**: Primary testing framework
- **pytest-cov**: Coverage analysis
- **pytest-mock**: Mocking utilities
- **pytest-benchmark**: Performance testing

### Mock Libraries
- **unittest.mock**: Standard mocking
- **responses**: HTTP request mocking
- **PIL**: Image processing for tests

### Test Data
- Sample images in various formats
- Mock neural network outputs
- Test configuration files
- Hardware simulation data

This testing module ensures the reliability and maintainability of the VLMChat application across different deployment scenarios.