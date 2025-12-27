# Regression Tests

Full system tests for stability and performance validation.

## Test Organization

- Long-running tests (10+ minutes)
- Hardware-dependent tests
- End-to-end pipeline tests
- Memory leak detection
- Performance benchmarks

## Running

```bash
python tests/run_tests.py --regression
```

## Platform-Specific Tests

- `test_jetson_*.py` - Jetson-specific tests
- `test_rpi_*.py` - Raspberry Pi-specific tests
- `test_*.py` - All platforms
