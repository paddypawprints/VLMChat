# VLMChat Annotation Tool - Implementation Summary

## Overview

A GUI-based annotation tool for testing VLMChat pipeline configurations with scenario files. The tool helps visualize detection results, edit prompts, and manage test scenarios.

## Current Status

### ✅ Phase 1: UI Design - COMPLETE

The UI is fully implemented with mock behavior:

**Components:**
- Image display panel with detection overlays
- Hierarchical detection tree (configurable depth 1-10)
- Prompt editor with add/edit/delete operations
- Scenario file loader/saver
- Control panel with status display

**Features:**
- Load and display images
- Show detections with color-coded bounding boxes
- Edit prompts and track changes
- Load/save YAML scenario files
- Mock data for testing without dependencies

### 🔨 Phase 2: Pipeline Integration - IN PROGRESS

Integration infrastructure is in place:

**Completed:**
- `ScenarioParser` - Full YAML parsing with tests
- `PipelineAdapter` - Structure for pipeline integration
- Scenario → Prompt extraction
- Prompt → Scenario updates
- Graceful fallback to mock mode

**Remaining Work:**
1. Connect `ScenarioParser` to actual pipeline configuration
2. Implement scenario → pipeline DSL translation
3. Run pipeline and capture Detection objects
4. Convert Detection objects to UI tree
5. Re-run pipeline when prompts change

## File Structure

```
testapp/
├── __init__.py                  # Package initialization
├── annotation_tool.py           # Main GUI (530+ lines)
├── mock_data.py                 # Mock data for testing
├── pipeline_integration.py      # Pipeline adapter (150 lines)
├── scenario_parser.py           # YAML parser (204 lines)
├── test_components.py           # Component tests
├── test_scenario_parser.py      # Parser tests
├── example_scenarios.yaml       # Sample scenario file
└── README.md                    # Documentation
```

## Running the Tool

### Without Display (Testing Only)
```bash
# Test core components
python testapp/test_components.py

# Test scenario parser
python testapp/test_scenario_parser.py
```

### With Display (Full GUI)
```bash
# Requires tkinter and display
python -m testapp.annotation_tool
```

## Usage Workflow

1. **Launch** - Tool starts with mock data loaded
2. **Load Scenario** - File → Load Scenario → Select YAML file
3. **View Prompts** - Prompts extracted and shown in right panel
4. **Edit Prompts** - Select prompt, edit text, click "Update Prompt"
5. **Load Image** - File → Load Image (optional, for visualization)
6. **Run Pipeline** - Click "Run Pipeline" (currently mock mode)
7. **View Detections** - Detection tree shows results with adjustable depth
8. **Save Scenario** - File → Save Scenario (saves updated prompts)

## Technical Details

### Scenario Parser

Parses YAML files with structure:
```yaml
image_settings:
  width: 1920
  height: 1080

scenarios:
  - id: scenario_name
    prompt: "scene description"
    entities:
      - id: entity_id
        label: category
        description: "details"
    expected_outcome:
      clusters:
        - id: cluster_id
          members: [entity_ids]
      matches:
        - query: "search query"
          target_id: cluster_id
```

**Capabilities:**
- Extract prompts from all parts (scene, entities, matches)
- Update prompts in place
- Preserve structure when saving
- Handle multiple scenarios per file

### Pipeline Integration

The `PipelineAdapter` provides:
- `create_simple_detection_pipeline()` - Set up pipeline
- `run_pipeline()` - Execute pipeline with context
- `get_detections()` - Extract Detection objects
- `get_prompts()` / `update_prompts()` - Prompt management
- `load_from_scenario()` - Configure from scenario dict

### Detection Tree

Shows hierarchical structure:
```
Detection 1 (cluster)
  ├─ Detection 2 (person)
  └─ Detection 3 (bicycle)
Detection 4 (cluster)
  ├─ Detection 5 (person)
  └─ Detection 6 (dog)
```

- Configurable depth (1-10 levels)
- Shows ID, category, confidence, bounding box
- Color-coded by depth level

## Testing

All core functionality is tested:

```bash
# Component tests (8 assertions)
python testapp/test_components.py

# Scenario parser tests (9 test cases)
python testapp/test_scenario_parser.py
```

Tests cover:
- Mock data generation
- Scenario file parsing
- Prompt extraction/update
- Cluster/entity extraction
- Detection hierarchy
- Round-trip data integrity

## Next Steps for Full Integration

### 1. Scenario → Pipeline Configuration

Map scenario elements to pipeline tasks:
```python
# From scenario:
entities = [
    {"id": "obj_person", "label": "person", ...},
    {"id": "obj_bike", "label": "bicycle", ...}
]

# To pipeline:
pipeline = """
camera(id=input) ->
detector(id=yolo, confidence=0.5) ->
clusterer(id=cluster, max_clusters=3) ->
clip_compare(id=matcher, prompts=["person riding bicycle"])
"""
```

### 2. Image Generation/Loading

Options:
- Load test images matching scenario descriptions
- Generate synthetic images from scenarios
- Use existing image library with tagging

### 3. Detection Conversion

Convert VLMChat Detection → UI format:
```python
def convert_detection(det: Detection) -> MockDetection:
    ui_det = MockDetection(det.id, det.box, det.object_category, det.conf)
    for child in det.children:
        ui_det.add_child(convert_detection(child))
    return ui_det
```

### 4. Re-run Pipeline on Edit

When prompt edited:
1. Update pipeline configuration
2. Re-run pipeline with new prompts
3. Extract new detections
4. Update UI tree
5. Redraw image with new overlays

### 5. Interactive Positioning

Add drag-and-drop for expected results:
- Click and drag cluster positions
- Update scenario coordinates
- Visual comparison: expected vs actual

## Dependencies

- Python 3.10+
- tkinter (GUI)
- PyYAML (scenario files)
- Pillow (image handling)
- VLMChat pipeline (src/pipeline/*)
- VLMChat detection (src/object_detector/*)

## Known Limitations

1. **GUI requires display** - Headless systems can only run tests
2. **Mock mode fallback** - Gracefully handles missing pipeline
3. **Single scenario display** - Multi-scenario files load first only
4. **No image generation** - Requires manual image loading
5. **Detection highlighting** - Selection tracking implemented but not visually highlighted yet

## Design Decisions

### Why tkinter?
- Built into Python (no extra dependencies)
- Cross-platform
- Good enough for internal tools
- Can be replaced later if needed

### Why mock data?
- Allows UI development without pipeline
- Tests UI logic independently
- Enables headless testing
- Provides fallback for errors

### Why separate parser?
- Clean separation of concerns
- Testable independently
- Reusable in other tools
- Clear API for scenario access

### Why gradual integration?
- Verify UI design first (Phase 1)
- Get user feedback early
- Reduce risk of rework
- Easier to test incrementally

## Future Enhancements

- Multi-scenario selector dialog
- Batch processing of scenarios
- Export detection results to JSON
- Comparison mode (before/after)
- Keyboard shortcuts
- Detection confidence threshold slider
- Image zoom/pan controls
- Undo/redo for prompt edits
- Recent files menu
- Configuration file for tool settings

## Questions for User

Before completing full integration:

1. **Image Source**: Should we generate images from scenarios, or use a test image library?
2. **Pipeline Config**: What prompts should control clustering? Just match queries, or entity descriptions too?
3. **Detection Display**: Any specific attributes to show beyond ID, category, confidence, box?
4. **Expected Results**: How should we handle positioning - manual drag, or automatic from scenario coordinates?
5. **Workflow Priority**: What's more important - quick iteration (edit/run), or detailed result analysis?

## Summary

**What's Working:**
- Complete UI with all panels
- Scenario loading and parsing
- Prompt extraction and editing
- Mock data visualization
- Comprehensive test coverage

**What's Left:**
- Connect scenario parser to pipeline factory
- Run actual VLMChat pipelines
- Display real Detection objects
- Re-run on prompt edit
- Interactive result positioning

**Estimated Effort:**
- ~2-3 hours for pipeline integration
- ~1-2 hours for detection display
- ~1-2 hours for re-run workflow
- ~2-3 hours for interactive positioning

Total remaining: ~6-10 hours for full Phase 2 completion
