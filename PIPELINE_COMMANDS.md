# Pipeline Commands Quick Reference

## New Commands

### `/pipeline <dsl_or_file>`
Load a pipeline from DSL text or a .dsl file.

**Examples:**
```bash
# Load from file in pipeline_dirs
/pipeline detection.dsl

# Load from absolute path
/pipeline /path/to/my/pipeline.dsl

# Load inline DSL
/pipeline NoneCamera() -> Grayscale() -> SaveImage(path="output.jpg")
```

**Notes:**
- Searches `~/pipelines` and `./pipelines` by default
- Supports both relative and absolute file paths
- Can use inline DSL for quick testing

---

### `/run [key=value ...]`
Execute the loaded pipeline with optional runtime overrides.

**Examples:**
```bash
# Run without overrides
/run

# Run with overrides
/run threshold=0.7 max_items=10

# Run with multiple overrides
/run confidence=0.85 output_dir=/tmp
```

**Notes:**
- Pipeline must be loaded first with `/pipeline`
- Only one pipeline can run at a time
- Overrides are space-separated `key=value` pairs
- Numeric values are auto-detected (int/float)
- Runs asynchronously in background thread

---

### `/stop`
Stop the currently running pipeline.

**Examples:**
```bash
/stop
```

**Notes:**
- Sets a stop flag for graceful cancellation
- Pipeline may take a moment to fully stop
- Use `/status` to verify pipeline has stopped

---

### `/status`
Check pipeline execution status.

**Examples:**
```bash
/status
```

**Output shows:**
- Whether a pipeline is currently running
- Whether a pipeline is loaded and ready to run

---

## Typical Workflow

1. **Load an image** (makes it available to pipeline camera tasks):
   ```bash
   /load_file path/to/image.jpg
   ```

2. **Load a pipeline**:
   ```bash
   /pipeline my_pipeline.dsl
   ```

3. **Check status** (optional):
   ```bash
   /status
   ```

4. **Run the pipeline**:
   ```bash
   /run
   ```

5. **Monitor or stop** (if needed):
   ```bash
   /status
   /stop
   ```

---

## Pipeline Directory Configuration

Default search locations (configured in `config.json`):
- `~/pipelines/` - User-specific pipelines
- `./pipelines/` - Project-local pipelines

To add more directories, edit `config.json`:
```json
{
  "paths": {
    "pipeline_dirs": [
      "~/pipelines",
      "./pipelines",
      "/custom/path/to/pipelines"
    ]
  }
}
```

---

## Integration with Environment

The **Environment singleton** provides shared state:
- `current_image`: Image loaded via `/load_file` or `/load_url`
- `history`: Conversation history (for future use)

**NoneCamera** integration:
- `NoneCamera()` task automatically uses `Environment.current_image`
- Falls back to test image if no image loaded
- Enables camera-based pipelines to work with loaded images

---

## Example Pipelines

### Simple Grayscale Conversion
```bash
/load_file photo.jpg
/pipeline NoneCamera() -> Grayscale() -> SaveImage(path="gray.jpg")
/run
```

### Detection with Threshold Override
```bash
/load_file scene.jpg
/pipeline detection_pipeline.dsl
/run confidence=0.8
```

### Stopping Long Pipeline
```bash
/pipeline complex_pipeline.dsl
/run
# ... wait a bit ...
/stop
```

---

## Tips

- Use `/status` frequently to check pipeline state
- Only one pipeline runs at a time (prevents conflicts)
- Pipeline execution is async (doesn't block chat)
- Override values are type-detected automatically
- Check `~/pipelines/` for shareable pipeline files
