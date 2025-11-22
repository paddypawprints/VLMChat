"""
Scenario parser for the annotation tool.

This module parses scenario YAML files and configures pipelines
to generate images and detections matching the scenario specification.
"""

from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ScenarioParser:
    """
    Parser for scenario YAML files.
    
    Converts scenario definitions into pipeline-compatible structures.
    """
    
    def __init__(self, scenario_data: Dict[str, Any]):
        """
        Initialize parser with scenario data.
        
        Args:
            scenario_data: Dictionary loaded from scenario YAML file
        """
        self.scenario_data = scenario_data
        self.image_settings = scenario_data.get('image_settings', {})
        self.scenarios = scenario_data.get('scenarios', [])
    
    def get_image_size(self) -> Tuple[int, int]:
        """
        Get the image dimensions from settings.
        
        Returns:
            Tuple of (width, height)
        """
        width = self.image_settings.get('width', 1920)
        height = self.image_settings.get('height', 1080)
        return (width, height)
    
    def get_scenario_by_id(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific scenario by its ID.
        
        Args:
            scenario_id: The scenario ID to find
            
        Returns:
            Scenario dictionary or None if not found
        """
        for scenario in self.scenarios:
            if scenario.get('id') == scenario_id:
                return scenario
        return None
    
    def get_all_scenario_ids(self) -> List[str]:
        """
        Get all scenario IDs in the file.
        
        Returns:
            List of scenario IDs
        """
        return [s.get('id', f'scenario_{i}') for i, s in enumerate(self.scenarios)]
    
    def extract_prompts(self, scenario: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract all prompts from a scenario.
        
        Args:
            scenario: Scenario dictionary
            
        Returns:
            Dictionary mapping prompt keys to prompt text
        """
        prompts = {}
        
        # Main scene prompt
        if 'prompt' in scenario:
            prompts['scene_prompt'] = scenario['prompt']
        
        # Entity descriptions
        entities = scenario.get('entities', [])
        for entity in entities:
            entity_id = entity.get('id', 'unknown')
            if 'description' in entity:
                prompts[f'entity_{entity_id}'] = entity['description']
        
        # Match queries from expected outcomes
        expected = scenario.get('expected_outcome', {})
        matches = expected.get('matches', [])
        for i, match in enumerate(matches):
            if 'query' in match:
                match_key = f"match_{match.get('type', 'query')}_{i}"
                prompts[match_key] = match['query']
        
        return prompts
    
    def extract_clusters(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract expected cluster information from scenario.
        
        Args:
            scenario: Scenario dictionary
            
        Returns:
            List of cluster definitions
        """
        expected = scenario.get('expected_outcome', {})
        clusters = expected.get('clusters', [])
        return clusters
    
    def extract_entities(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract entity definitions from scenario.
        
        Args:
            scenario: Scenario dictionary
            
        Returns:
            List of entity definitions
        """
        return scenario.get('entities', [])
    
    def get_environment_info(self, scenario: Dict[str, Any]) -> Dict[str, str]:
        """
        Get environment information from scenario.
        
        Args:
            scenario: Scenario dictionary
            
        Returns:
            Environment dictionary (time, weather, lighting)
        """
        return scenario.get('environment', {})
    
    def update_scenario_prompts(self, scenario_id: str, prompts: Dict[str, str]) -> bool:
        """
        Update prompts in a scenario.
        
        Args:
            scenario_id: ID of scenario to update
            prompts: New prompt dictionary
            
        Returns:
            True if updated successfully, False otherwise
        """
        scenario = self.get_scenario_by_id(scenario_id)
        if scenario is None:
            logger.warning(f"Scenario {scenario_id} not found")
            return False
        
        # Update main prompt if present
        if 'scene_prompt' in prompts:
            scenario['prompt'] = prompts['scene_prompt']
        
        # Update entity descriptions
        entities = scenario.get('entities', [])
        for entity in entities:
            entity_id = entity.get('id')
            entity_key = f'entity_{entity_id}'
            if entity_key in prompts:
                entity['description'] = prompts[entity_key]
        
        # Update match queries
        expected = scenario.get('expected_outcome', {})
        matches = expected.get('matches', [])
        for i, match in enumerate(matches):
            match_key = f"match_{match.get('type', 'query')}_{i}"
            if match_key in prompts:
                match['query'] = prompts[match_key]
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the parser's data back to dictionary format.
        
        Returns:
            Full scenario data dictionary
        """
        return self.scenario_data


def parse_scenario_file(filepath: str) -> ScenarioParser:
    """
    Parse a scenario YAML file.
    
    Args:
        filepath: Path to the scenario YAML file
        
    Returns:
        ScenarioParser instance
    """
    import yaml
    
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    return ScenarioParser(data)
