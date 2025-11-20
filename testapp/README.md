# VLMChat Annotation Tool

A GUI tool for annotating images and testing VLMChat pipeline configurations with scenario files.

## Overview

This tool helps you:
- Visualize detection results from VLMChat pipelines
- View and edit prompts used in clustering
- Load and save scenario configuration files
- Test pipeline configurations interactively

## Features

### Phase 1: UI Design (Current - Mock Behavior) ✅
- Image display panel with detection overlays
- Hierarchical detection tree view with configurable depth
- Prompt editor with list of all prompts
- Scenario file loading and saving
- Mock data for UI testing

### Phase 2: Integration (Planned)
- Real VLMChat pipeline integration
- Live detection from pipeline execution
- Edit prompts and re-run pipeline automatically
- Update scenario files with current results
- Move expected results on the screen

## Installation

The annotation tool requires:
- Python 3.10+
- tkinter (Python's standard GUI library)
- PyYAML
- Pillow (PIL)

### Installing Dependencies

Most systems include tkinter with Python. If not, install it:

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-tk
```

**macOS:**
```bash
# Usually included with Python
# If not, install Python from python.org
```

**Windows:**
```bash
# Usually included with Python
```

**Python packages:**
```bash
pip install pillow pyyaml
# Or use the project requirements:
pip install -r requirements.txt
```

## Usage

### Running the Tool

From the VLMChat root directory:

```bash
python -m testapp.annotation_tool
```

Or directly:

```bash
python testapp/annotation_tool.py
```

### Loading Mock Data

Click "Load Mock Data" button to populate the UI with sample detections and prompts.

### Loading a Scenario

1. Click `File > Load Scenario`
2. Select a YAML scenario file
3. The scenario configuration will be loaded (full integration pending)

### Loading an Image

1. Click `File > Load Image`
2. Select an image file (PNG, JPG, etc.)
3. The image will be displayed with detection overlays

### Editing Prompts

1. Select a prompt from the list in the Prompts panel
2. Edit the text in the editor below
3. Click "Update Prompt" to save changes
4. Click "Run Pipeline" to re-run with new prompts (mock mode currently)

### Viewing Detections

- The detection tree shows all detected objects
- Adjust the "Display Depth" spinner to show more/fewer levels of children
- Click on a detection to select it (highlighting coming in Phase 2)

### Saving Scenarios

1. Click `File > Save Scenario`
2. Choose a location and filename
3. The current scenario configuration with prompts will be saved

## Scenario File Format

Example scenario file structure:

```yaml
image_settings:
  width: 1920
  height: 1080

scenarios:
  - id: std_ride_bike_crowded
    type: standard
    environment:
      time: "day"
      weather: "sunny"
      lighting: "high contrast shadows"
    prompt: "Daytime CCTV close-up view..."
    entities:
      - id: obj_person
        label: person
        description: "person riding a bicycle..."
        size: large
        position: center
    expected_outcome:
      clusters:
        - id: cluster_main
          location: center
          members: ["obj_person", "obj_bike"]
      matches:
        - query: "person riding bicycle"
          type: relationship
          target_id: "cluster_main"
```

## Architecture

### Modules

- `annotation_tool.py` - Main GUI application
- `mock_data.py` - Mock data structures for testing
- `__init__.py` - Package initialization

### UI Components

1. **Image Display Panel** - Shows the current image with detection overlays
2. **Detection Tree Panel** - Hierarchical view of all detections with configurable depth
3. **Prompt Editor Panel** - List and editor for all prompts
4. **Control Panel** - Buttons for running pipeline and updating scenarios

## Development

### Phase 1: UI Design (Complete)
- ✅ Basic UI layout with three panels
- ✅ Mock data structures
- ✅ Image display with detection overlays
- ✅ Detection tree view with depth control
- ✅ Prompt list and editor
- ✅ Scenario load/save (basic)

### Phase 2: Integration (Next Steps)
- [ ] Import VLMChat pipeline components
- [ ] Load real Detection objects from pipeline
- [ ] Parse scenario files into pipeline configuration
- [ ] Run pipeline on button click
- [ ] Update detection display from pipeline results
- [ ] Save updated prompts back to scenario
- [ ] Interactive detection result positioning

## Testing

### Component Tests (No GUI Required)

Test the core logic without running the GUI:

```bash
python testapp/test_components.py
```

This validates:
- Mock detection generation
- Mock prompt generation
- Mock scenario generation
- Scenario file loading
- Detection hierarchy structure

### GUI Testing (Requires Display)

Currently uses mock data to test the UI:

```python
from testapp.mock_data import get_mock_detections, get_mock_prompts, get_mock_scenario

# Get mock detections
detections = get_mock_detections()

# Get mock prompts
prompts = get_mock_prompts()

# Get mock scenario
scenario = get_mock_scenario()
```

Run the full GUI (requires tkinter and a display):

```bash
python -m testapp.annotation_tool
```

## Known Limitations

- Phase 1 only: Uses mock data
- Pipeline integration not yet implemented
- Detection highlighting on selection not yet implemented
- Interactive positioning of expected results not yet implemented

## Future Enhancements

- Real-time pipeline execution with progress indicator
- Multiple scenario support in one file
- Export detection results to various formats
- Comparison mode for before/after pipeline runs
- Batch processing of multiple scenarios
- Keyboard shortcuts for common operations
