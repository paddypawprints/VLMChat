# Prompt Module Test Suite

This directory contains a comprehensive test suite for the VLMChat prompt module components, including unit tests, integration tests, performance tests, and edge case tests.

## Overview

The test suite provides thorough coverage of all prompt module components:
- **History**: Conversation history management
- **Prompt**: Facade for prompt operations
- **Formatters**: XML and minimal formatting strategies
- **Factory**: Factory pattern for formatter creation
- **Integration**: Component interaction testing
- **Performance**: Scalability and efficiency testing

## Test Structure

```
test_prompt/
├── conftest.py                    # Pytest fixtures and configuration
├── test_history.py               # History class unit tests
├── test_prompt.py                # Prompt facade unit tests
├── test_formatters.py            # Formatter classes unit tests
├── test_factory.py               # Factory pattern unit tests
├── test_integration.py           # Integration tests
├── test_performance_edge_cases.py # Performance and edge case tests
└── README.md                     # This file
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
Test individual components in isolation:
- **History Management**: Conversation storage, limits, statistics
- **Image Handling**: Image setting, clearing, validation
- **Format Management**: Format switching, configuration
- **Formatter Logic**: XML/minimal formatting, truncation
- **Factory Pattern**: Formatter creation, error handling
- **Facade Operations**: Prompt interface, delegation

### Integration Tests (`@pytest.mark.integration`)
Test component interactions:
- **History-Formatter Integration**: Format switching with data
- **Prompt-History Integration**: Facade delegation and consistency
- **Factory-History Integration**: Formatter creation in context
- **End-to-End Scenarios**: Complete conversation workflows

### Performance Tests (`@pytest.mark.performance`)
Test efficiency and scalability:
- **Large Conversations**: Performance with many conversation pairs
- **Memory Usage**: Memory efficiency and cleanup
- **Format Switching**: Overhead of changing formatters
- **Scalability**: Linear scaling characteristics

### Edge Case Tests (`@pytest.mark.edge_case`)
Test boundary conditions and error scenarios:
- **Empty/None Values**: Handling of missing data
- **Unicode/Special Characters**: International text support
- **Extreme Values**: Very large texts, limits
- **Error Recovery**: Graceful handling of invalid input

## Running Tests

### Quick Start
```bash
# Run all tests
python -m pytest src/tests/test_prompt/

# Run with the provided test runner
python run_prompt_tests.py
```

### Test Categories
```bash
# Unit tests only
python run_prompt_tests.py --unit

# Integration tests
python run_prompt_tests.py --integration

# Performance tests
python run_prompt_tests.py --performance

# Edge case tests
python run_prompt_tests.py --edge-case
```

### Specific Components
```bash
# Test specific file
python run_prompt_tests.py --test-file test_history.py

# Test with coverage
python run_prompt_tests.py --coverage

# Quick test suite
python run_prompt_tests.py --quick

# Full test suite with coverage
python run_prompt_tests.py --full
```

### Advanced Options
```bash
# Parallel execution (requires pytest-xdist)
python run_prompt_tests.py --parallel 4

# With timeout (requires pytest-timeout)
python run_prompt_tests.py --timeout 300

# Verbose output
python run_prompt_tests.py --verbose

# Custom pytest arguments
python run_prompt_tests.py --pytest-args "-k test_conversation"
```

## Test Fixtures

The `conftest.py` file provides comprehensive fixtures:

### Data Fixtures
- `sample_image`: Test PIL image
- `sample_conversation_pairs`: Standard conversation data
- `long_conversation_pairs`: Data exceeding typical limits
- `text_with_special_chars`: Unicode and special character text

### Component Fixtures
- `history_default`: Default History instance
- `history_xml/minimal`: Pre-configured History instances
- `prompt_default`: Default Prompt instance
- `xml_formatter/minimal_formatter`: Formatter instances

### Utility Fixtures
- `TestUtilities`: Helper functions for testing
- Custom assertion methods for XML/minimal format validation

## Test Coverage

The test suite aims for comprehensive coverage:

### History Class
- ✅ Initialization and configuration
- ✅ Conversation pair management
- ✅ Image handling
- ✅ Format switching
- ✅ Statistics and debugging
- ✅ Edge cases and error conditions

### Prompt Facade
- ✅ Initialization and properties
- ✅ Conversation management delegation
- ✅ Statistics access
- ✅ Image coordination
- ✅ Facade pattern implementation

### Formatters
- ✅ XML formatter structure and content
- ✅ Minimal formatter with truncation
- ✅ Format comparison and efficiency
- ✅ Special character handling
- ✅ Performance characteristics

### Factory Pattern
- ✅ Formatter creation for all types
- ✅ Configuration passing
- ✅ Error handling for invalid types
- ✅ Extensibility support

### Integration Scenarios
- ✅ End-to-end conversation workflows
- ✅ Format switching during conversations
- ✅ Conversation limits with multiple components
- ✅ Error recovery scenarios

## Performance Benchmarks

The test suite includes performance benchmarks:

### Expected Performance
- **1000 conversations**: < 1 second to add
- **Format switching**: < 1 second for 100 switches
- **Large text formatting**: < 0.1 second per operation
- **Memory usage**: Bounded by conversation limits

### Scalability Tests
- Linear scaling with conversation count
- Efficient memory usage with limits
- Fast formatter creation
- Consistent performance across operations

## Requirements

### Core Requirements
- Python 3.8+
- pytest >= 6.0
- PIL (Pillow) for image handling

### Optional Requirements
- `pytest-cov`: Coverage reporting
- `pytest-xdist`: Parallel test execution
- `pytest-timeout`: Test timeouts
- `pytest-benchmark`: Performance benchmarking

### Installation
```bash
# Core requirements
pip install pytest pillow

# Optional requirements
pip install pytest-cov pytest-xdist pytest-timeout pytest-benchmark
```

## Development Guidelines

### Writing New Tests
1. **Follow naming conventions**: `test_*.py` files, `test_*` functions
2. **Use appropriate markers**: `@pytest.mark.unit`, `@pytest.mark.integration`
3. **Use existing fixtures**: Leverage shared fixtures from `conftest.py`
4. **Test edge cases**: Include boundary conditions and error cases
5. **Document complex tests**: Add docstrings explaining test purpose

### Test Organization
- **One test class per major feature**
- **Group related tests** within classes
- **Use descriptive test names** that explain what is being tested
- **Include both positive and negative test cases**

### Performance Testing
- **Use time measurements** for performance-critical operations
- **Test with realistic data sizes**
- **Include memory usage considerations**
- **Compare different approaches** when applicable

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure you're running from project root
cd /path/to/VLMChat
python -m pytest src/tests/test_prompt/
```

#### Missing Dependencies
```bash
# Install required packages
pip install pytest pillow

# For coverage reporting
pip install pytest-cov
```

#### Path Issues
```bash
# Add src to Python path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Test Failures
1. **Check test output** for specific failure details
2. **Run individual tests** to isolate issues
3. **Use verbose mode** (`-v` or `--verbose`)
4. **Check fixtures** are properly configured

## Contributing

When contributing to the test suite:

1. **Run existing tests** to ensure no regressions
2. **Add tests for new functionality**
3. **Update fixtures** as needed for new test cases
4. **Follow existing patterns** for consistency
5. **Update documentation** for new test categories

## Test Results

Expected test results for a complete run:

```
======================== test session starts ========================
collected 150+ items

src/tests/test_prompt/test_history.py .................... [ 25%]
src/tests/test_prompt/test_formatters.py ................ [ 45%]
src/tests/test_prompt/test_factory.py ................... [ 60%]
src/tests/test_prompt/test_prompt.py .................... [ 75%]
src/tests/test_prompt/test_integration.py ............... [ 90%]
src/tests/test_prompt/test_performance_edge_cases.py .... [100%]

==================== 150+ passed in X.XXs =====================
```

This comprehensive test suite ensures the reliability, performance, and maintainability of the VLMChat prompt module.