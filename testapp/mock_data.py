"""
Mock data structures for testing the annotation tool UI.

This module provides sample data to test the UI before pipeline integration.
"""

from typing import List, Dict, Any, Tuple


class MockDetection:
    """Mock detection object for UI testing."""
    
    def __init__(self, id: int, box: Tuple[int, int, int, int], 
                 object_category: str, conf: float = 0.0):
        self.id = id
        self.box = box  # (x1, y1, x2, y2)
        self.object_category = object_category
        self.conf = conf
        self.children: List['MockDetection'] = []
    
    def add_child(self, detection: 'MockDetection') -> None:
        self.children.append(detection)
    
    def __repr__(self) -> str:
        return f"Detection(id={self.id}, cat='{self.object_category}', conf={self.conf:.2f})"


def get_mock_detections() -> List[MockDetection]:
    """
    Generate mock detections for UI testing.
    
    Returns:
        List of mock Detection objects with hierarchical structure.
    """
    # Create main cluster
    main_cluster = MockDetection(1, (100, 100, 300, 400), "cluster", conf=0.95)
    
    # Add person to main cluster
    person = MockDetection(2, (120, 120, 280, 380), "person", conf=0.92)
    main_cluster.add_child(person)
    
    # Add bicycle to main cluster
    bicycle = MockDetection(3, (100, 250, 300, 400), "bicycle", conf=0.88)
    main_cluster.add_child(bicycle)
    
    # Create second cluster (walker)
    walker_cluster = MockDetection(4, (400, 150, 550, 400), "cluster", conf=0.87)
    walker = MockDetection(5, (410, 160, 540, 390), "person", conf=0.84)
    walker_cluster.add_child(walker)
    
    # Create third cluster (dog walker)
    dog_cluster = MockDetection(6, (600, 200, 800, 450), "cluster", conf=0.90)
    dog_walker = MockDetection(7, (620, 210, 720, 440), "person", conf=0.86)
    dog = MockDetection(8, (680, 350, 790, 440), "dog", conf=0.82)
    dog_cluster.add_child(dog_walker)
    dog_cluster.add_child(dog)
    
    return [main_cluster, walker_cluster, dog_cluster]


def get_mock_prompts() -> Dict[str, str]:
    """
    Generate mock prompts for UI testing.
    
    Returns:
        Dictionary of prompt names to prompt text.
    """
    return {
        "main_query": "person riding bicycle",
        "attribute_query": "blue t-shirt",
        "relationship_query": "person walking dog",
        "cluster_prompt_1": "Find person on bicycle",
        "cluster_prompt_2": "Find pedestrians",
        "cluster_prompt_3": "Find dog and owner",
    }


def get_mock_scenario() -> Dict[str, Any]:
    """
    Generate mock scenario data for UI testing.
    
    Returns:
        Dictionary representing a scenario configuration.
    """
    return {
        "id": "std_ride_bike_crowded",
        "type": "standard",
        "environment": {
            "time": "day",
            "weather": "sunny",
            "lighting": "high contrast shadows"
        },
        "prompt": "Daytime CCTV close-up view of a person riding a bicycle on pavement, crowded street with pedestrians",
        "entities": [
            {
                "id": "obj_person",
                "label": "person",
                "description": "person riding a bicycle, with dark skin and black hair, wearing green shorts and a blue t-shirt",
                "size": "large",
                "position": "center"
            },
            {
                "id": "obj_bike",
                "label": "bicycle",
                "size": "large",
                "relation": {
                    "type": "overlap",
                    "target": "obj_person",
                    "strength": 0.8
                }
            }
        ],
        "expected_outcome": {
            "clusters": [
                {
                    "id": "cluster_main",
                    "location": "center",
                    "members": ["obj_person", "obj_bike"]
                }
            ],
            "matches": [
                {
                    "query": "person riding bicycle",
                    "type": "relationship",
                    "target_id": "cluster_main"
                }
            ]
        }
    }
