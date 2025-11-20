"""
Pipeline integration for the annotation tool.

This module provides integration between the annotation tool UI
and the VLMChat pipeline system.
"""

import sys
import os
from typing import List, Dict, Any, Optional

# Add parent directory to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline.task_base import Context, ContextDataType
from src.pipeline.pipeline_runner import PipelineRunner
from src.pipeline.pipeline_factory import create_default_factory
from src.object_detector.detection_base import Detection


class PipelineAdapter:
    """
    Adapter to run VLMChat pipelines and extract results for the annotation tool.
    """
    
    def __init__(self):
        """Initialize the pipeline adapter."""
        self.factory = create_default_factory()
        self.current_pipeline = None
        self.current_context = None
        self.last_result = None
    
    def create_simple_detection_pipeline(self, image_path: Optional[str] = None,
                                         detector_type: str = "yolo_cpu") -> None:
        """
        Create a simple pipeline for object detection.
        
        Args:
            image_path: Optional path to image file
            detector_type: Type of detector to use (yolo_cpu, yolo, etc.)
        """
        # For now, create a minimal pipeline structure
        # This will be expanded in later iterations
        from src.pipeline.task_base import Connector
        
        pipeline = Connector("detection_pipeline")
        
        # TODO: Add actual tasks based on requirements
        # For now, return empty pipeline that can be expanded
        self.current_pipeline = pipeline
    
    def run_pipeline(self, context: Optional[Context] = None) -> Context:
        """
        Run the current pipeline.
        
        Args:
            context: Optional context to use, creates new one if None
            
        Returns:
            Result context with pipeline outputs
        """
        if self.current_pipeline is None:
            raise ValueError("No pipeline configured. Call create_*_pipeline first.")
        
        if context is None:
            context = Context()
        
        # Create runner and execute
        runner = PipelineRunner(self.current_pipeline)
        runner.build_graph()
        result = runner.run(context)
        
        self.last_result = result
        self.current_context = result
        
        return result
    
    def get_detections(self) -> List[Detection]:
        """
        Get detections from the last pipeline run.
        
        Returns:
            List of Detection objects, or empty list if none
        """
        if self.last_result is None:
            return []
        
        detections = self.last_result.data.get(ContextDataType.DETECTIONS, [])
        return detections if detections else []
    
    def get_prompts(self) -> Dict[str, str]:
        """
        Get prompts from the pipeline configuration.
        
        Returns:
            Dictionary of prompt names to prompt text
        """
        # TODO: Extract prompts from pipeline configuration
        # For now return empty dict
        return {}
    
    def update_prompts(self, prompts: Dict[str, str]) -> None:
        """
        Update prompts in the pipeline configuration.
        
        Args:
            prompts: Dictionary of prompt names to prompt text
        """
        # TODO: Update pipeline configuration with new prompts
        pass
    
    def load_from_scenario(self, scenario: Dict[str, Any]) -> None:
        """
        Configure pipeline from a scenario definition.
        
        Args:
            scenario: Scenario dictionary loaded from YAML
        """
        # TODO: Parse scenario and create appropriate pipeline
        # This is a key integration point for Phase 2
        pass


def convert_detection_to_dict(detection: Detection) -> Dict[str, Any]:
    """
    Convert a Detection object to a dictionary for UI display.
    
    Args:
        detection: Detection object to convert
        
    Returns:
        Dictionary representation of the detection
    """
    return {
        'id': detection.id,
        'box': detection.box,
        'object_category': detection.object_category,
        'conf': detection.conf,
        'children': [convert_detection_to_dict(child) for child in detection.children]
    }


def load_scenario_file(filepath: str) -> Dict[str, Any]:
    """
    Load a scenario from a YAML file.
    
    Args:
        filepath: Path to the YAML scenario file
        
    Returns:
        Scenario dictionary
    """
    import yaml
    
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    return data


def save_scenario_file(filepath: str, scenario: Dict[str, Any]) -> None:
    """
    Save a scenario to a YAML file.
    
    Args:
        filepath: Path to save the YAML file
        scenario: Scenario dictionary to save
    """
    import yaml
    
    with open(filepath, 'w') as f:
        yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)
