# Integration Tests

Integration tests verify interactions between components.

## Test Organization

- Tests run on all platforms unless marked with platform-specific markers
- Use `@pytest.mark.jetson`, `@pytest.mark.rpi`, etc. for platform-specific tests
- Integration tests should complete in < 5 minutes

## Running

```bash
python tests/run_tests.py --integration
```
