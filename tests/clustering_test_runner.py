"""
Test runner for the ObjectClusterer.
...
"""

import json
import os
import sys
from typing import List, Dict, Any, Tuple

# --- Add this 5-line path modification to match main.py ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(REPO_ROOT, 'src')
for p in (REPO_ROOT, SRC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


# --- Import classes to be tested ---
# These imports will now work:
from vlmchat.object_detector.detection_base import Detection
from vlmchat.object_detector.object_clusterer import ObjectClusterer, ISemanticCostProvider

# --- End Imports ---


class MockSemanticProvider(ISemanticCostProvider):
    """
    A mock semantic provider that uses a pre-loaded dictionary
    to return costs.
    """
    def __init__(self, cost_dict: Dict[str, Dict[str, float]]):
        self._costs = cost_dict
        self._default_cost = 1.0

    def start(self) -> None:
        """Mock start."""
        pass # Nothing to do

    def stop(self) -> None:
        """Mock stop."""
        pass # Nothing to do

    def get_pair_cost(self, category_a: str, category_b: str) -> float:
        """
        Gets the semantic cost using the N*N PAIR ("A and B") algorithm.
        """
        return self.get_cost(category_a, category_b)
        
    def get_single_cost(self, category_a: str, category_b: str) -> float:
        """
        Gets the semantic cost using the N*K SINGLE (A vs P, B vs P) algorithm.
        """
        return self.get_cost(category_a, category_b)

        
    def get_cost(self, category_a: str, category_b: str) -> float:
        """Gets the pre-defined cost from the dictionary."""
        try:
            return self._costs[category_a][category_b]
        except KeyError:
            # Symmetrical check
            try:
                return self._costs[category_b][category_a]
            except KeyError:
                # print(f"Warning: No mock cost for {category_a}/{category_b}, returning default.")
                return self._default_cost


def parse_detections_from_json(json_list: List[Dict[str, Any]]) -> List[Detection]:
    """Converts a list of JSON detection dicts into Detection objects."""
    detections = []
    for d_json in json_list:
        try:
            det = Detection(
                box=tuple(d_json['box']), # type: ignore
                object_category=d_json['object_category'],
                conf=d_json['conf']
            )
            
            # Recursively parse children
            if 'children' in d_json and d_json['children']:
                det.children = parse_detections_from_json(d_json['children'])
                
            detections.append(det)
        except KeyError as e:
            print(f"  Error: Test data is missing key {e}")
            raise
    return detections


def run_test_case(test_data: Dict[str, Any], test_number: int) -> bool:
    """
    Runs a single test case from a JSON object.
    
    Returns True on pass, False on fail.
    """
    print(f"--- Running Test: {test_number} ---")
    
    try:
        test_name = test_data['test_name']
        print(f"  Name: {test_name}")

        # 1. Parse inputs
        params = test_data['clusterer_params']
        mock_costs = test_data['mock_semantic_costs']
        input_detections = parse_detections_from_json(test_data['input_detections'])
        
        # 2. Initialize the mock semantic provider
        mock_provider = MockSemanticProvider(mock_costs)
        semantic_weights = {}
        semantic_weights["pair"] = params['semantic_weights'][0]
        semantic_weights["single"] = params['semantic_weights'][1]
        # 3. Initialize the ObjectClusterer with test params
        clusterer = ObjectClusterer(
            source=None, # We are testing in isolation
            semantic_provider=mock_provider,
            max_clusters=params['max_clusters'],
            merge_threshold=params['merge_threshold'],
            semantic_weights = semantic_weights
        )
        
        # Start the clusterer (and its mock provider)
        clusterer.start()

        # 4. Run the clustering logic
        # We pass image=None because the clusterer logic doesn't use it
        actual_clusters = clusterer._detect_internal(image=None, detections=input_detections)
        
        # 5. Get the expected output
        expected_clusters = []
        for det_json in test_data['expected_clusters']:
            expected_clusters.append(Detection.deserialize(det_json))

        # 6. Convert actual output to a comparable JSON format
        #actual_clusters_json = detections_to_json(actual_clusters)
        
        # 7. Compare
        # Simple JSON comparison. NOTE: This is order-sensitive.
        # For a more robust test, you'd sort or use a set-based comparison.
        comp = False 
        if len(actual_clusters) == len(expected_clusters):
            # Use zip() to pair items by index
            for det1, det2 in zip(actual_clusters, expected_clusters):
                if not det1.compare(det2):
                    break
            comp = True
        if comp:
            print(f"  Result: \033[92mPASS\033[0m") # Green text
            return True
        else:
            print(f"  Result: \033[91mFAIL\033[0m") # Red text
            print("\n  Expected:")
            for det in expected_clusters:
                print(f"{det}")
            print("\n  Actual:")
            for det in actual_clusters:
                print(f"{det}")
            return False

    except Exception as e:
        print(f"  Result: \033[91mERROR\033[0m") # Red text
        print(f"  An exception occurred: {e}")
        import traceback
        traceback.print_exc()

        return False

def main():
    """
    Main function to find and run all test cases.
    """
    # Assumes test cases are in this file in the 'tests' subdirectory
    test_cases_file = "tests/clustering_test_data.json"
    
    if not os.path.exists(test_cases_file):
        print(f"Error: Test file not found: {test_cases_file}")
        return

    print(f"Loading test cases from {test_cases_file}...\n")
    
    with open(test_cases_file, 'r') as f:
        try:
            test_data = json.load(f)
            test_case_list = test_data["test_cases"]
        except Exception as e:
            print(f"Error parsing test file: {e}")
            return
            
    if not test_case_list:
        print("Error: No 'test_cases' array found in test file.")
        return

    print(f"Found {len(test_case_list)} test case(s)...\n")
    
    pass_count = 0
    fail_count = 0
    
    for i, test_case in enumerate(test_case_list):
        if run_test_case(test_case, i + 1):
            pass_count += 1
        else:
            fail_count += 1
            
    print("\n--- Test Summary ---")
    print(f"Total Tests: {pass_count + fail_count}")
    print(f"\033[92mPassed: {pass_count}\033[0m")
    if fail_count > 0:
        print(f"\033[91mFailed: {fail_count}\033[0m")
    else:
        print(f"Failed: {fail_count}")


if __name__ == "__main__":
    if 'ObjectClusterer' not in globals() or ObjectClusterer is None:
        print("Test runner cannot start: Required classes were not imported.")
    else:
        main()