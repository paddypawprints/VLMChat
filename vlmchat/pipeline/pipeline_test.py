"""
Pipeline Testing Suite

This module contains comprehensive tests for the VLMChat pipeline system,
including tests for:
- Basic pipeline connectivity and data flow
- YOLO object detection
- CLIP vision-language matching
- Detection clustering and merging
- Color enhancement and manipulation
- Semantic attribute detection

Originally part of pipeline_runner.py, this file was extracted to improve
maintainability and separation of concerns.

Tests 16 & 17 specifically evaluate CLIP's color detection capabilities
and are documented in CLIP_TESTING_RESULTS.md.

Usage:
    python -m src.pipeline.pipeline_test                    # Run all non-interactive tests
    python -m src.pipeline.pipeline_test --interactive      # Include interactive/GUI tests
    python -m src.pipeline.pipeline_test --test 16          # Run specific test(s)
    python -m src.pipeline.pipeline_test --test 1,3,5       # Run multiple tests
    python -m src.pipeline.pipeline_test --test 1-4         # Run test range
    python -m src.pipeline.pipeline_test --log-level INFO   # Set log level
"""

if __name__ == "__main__":
    import sys
    import os
    import json
    import logging
    import argparse
    
    # Add project root to sys.path so that relative imports work
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Add src directory to sys.path for direct imports
    src_root = os.path.join(project_root, 'src')
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pipeline Runner Tests')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run interactive tests (includes GUI tests)')
    parser.add_argument('--log-level', type=str, default='WARNING',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set logging level (default: WARNING)')
    parser.add_argument('--test', type=str, default=None,
                       help='Run specific test(s) only (e.g., "1", "3,5", "8-9", or "all")')
    args = parser.parse_args()
    
    interactive_mode = args.interactive
    
    # Parse test selection
    selected_tests = None
    if args.test:
        selected_tests = set()
        for part in args.test.split(','):
            part = part.strip()
            if '-' in part:
                # Range like "1-3"
                start, end = part.split('-')
                selected_tests.update(range(int(start), int(end) + 1))
            elif part.lower() == 'all':
                selected_tests = None  # Run all tests
                break
            else:
                selected_tests.add(int(part))
    
    def should_run_test(test_num: int) -> bool:
        """Check if a test should run based on command line args."""
        return selected_tests is None or test_num in selected_tests
    
    # =========================================================================
    # CONSOLIDATED IMPORTS FOR TESTS
    # =========================================================================
    # Import all modules needed by tests once at the top
    import cv2
    import numpy as np
    from PIL import Image
    
    # Pipeline task imports
    from .tasks.detection_expander_task import DetectionExpanderTask
    from .tasks.color_enhance_task import ColorEnhanceTask
    from .tasks.color_swap_task import ColorSwapTask
    from .tasks.prompt_embedding_source_task import PromptEmbeddingSourceTask
    from .task_base import BaseTask, Context, ContextDataType
    
    # Object detection imports
    from ..object_detector.yolo_detector_cpu import YoloV8Detector
    from ..object_detector.object_clusterer import ObjectClusterer
    from ..object_detector.detection_base import Detection
    
    # =========================================================================
    # COMMON TEST UTILITIES
    # =========================================================================
    
    class CustomImageTask(BaseTask):
        """Task that loads an image from disk instead of from a camera."""
        def __init__(self, image_path, task_id="custom_image"):
            super().__init__(task_id)
            self.image_path = image_path
            self.output_contract = {ContextDataType.IMAGE: Image.Image}
        
        def configure(self, params):
            pass
        
        def run(self, context):
            img_bgr = cv2.imread(self.image_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Could not load image: {self.image_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            context.data[ContextDataType.IMAGE] = [pil_img]
            return context
    
    class DetectorTaskWrapper(BaseTask):
        """Wrapper that makes a detector compatible with the task interface."""
        def __init__(self, detector, task_id="detector"):
            super().__init__(task_id)
            self.detector = detector
            self.input_contract = {ContextDataType.IMAGE: Image.Image}
            self.output_contract = {ContextDataType.DETECTIONS: list}
        
        def configure(self, params):
            pass
        
        def run(self, context):
            img_list = context.data.get(ContextDataType.IMAGE, [])
            if not img_list:
                return context
            img = img_list[0]  # Get first image
            img_array = np.array(img)
            detections = self.detector.detect(img_array)
            context.data[ContextDataType.DETECTIONS] = detections
            return context
    
    # =========================================================================
    # TEST CONFIGURATION
    # =========================================================================
    
    from dataclasses import dataclass
    from typing import Optional, List, Dict, Any
    
    @dataclass
    class TestConfig:
        """Centralized configuration for pipeline tests."""
        # YOLO detector settings
        yolo_model: str = "yolov8n.pt"
        yolo_confidence: float = 0.5
        
        # CLIP settings
        min_similarity: float = 0.25
        
        # Detection processing
        expansion_factor: float = 0.20  # 20% box expansion
        detection_merge_target: int = 8  # Target number of detections after merge
        
        # Clusterer settings
        max_clusters: Optional[int] = None  # From config file
        context_filter_threshold: float = 0.20
        proximity_weight: float = 0.8
        size_weight: float = 0.5
        
        # Display settings
        display_time_ms: int = 3000  # Milliseconds (0=skip, -1=wait for key)
        
        # Prompts
        context_prompts: List[str] = None
        attribute_prompts: List[str] = None
        test_prompts: List[str] = None
        
        def __post_init__(self):
            """Set default prompts if not provided."""
            if self.context_prompts is None:
                self.context_prompts = []
            if self.attribute_prompts is None:
                self.attribute_prompts = []
            if self.test_prompts is None:
                self.test_prompts = []
        
        @property
        def all_prompts(self) -> List[str]:
            """Get all prompts combined."""
            return self.context_prompts + self.attribute_prompts + self.test_prompts
    
    # Configuration for each test
    TEST_CONFIGS: Dict[int, TestConfig] = {
        1: TestConfig(),  # Basic linear pipeline
        2: TestConfig(),  # Branch pipeline
        3: TestConfig(),  # FirstComplete
        4: TestConfig(),  # OrderedMerge
        5: TestConfig(),  # Factory test
        6: TestConfig(),  # Factory with configuration
        7: TestConfig(yolo_confidence=0.3),  # CameraTask configure
        8: TestConfig(yolo_confidence=0.3, display_time_ms=0),  # YOLO with viewer (0=passthrough)
        9: TestConfig(yolo_confidence=0.3, max_clusters=5, display_time_ms=0),  # Clusterer with viewer (0=passthrough)
        10: TestConfig(
            yolo_confidence=0.3,
            max_clusters=5,
            detection_merge_target=8,
            min_similarity=0.25,
            display_time_ms=0  # 0=passthrough, set to >0 for display
        ),
        11: TestConfig(
            yolo_confidence=0.5,
            min_similarity=0.25,
            expansion_factor=0.20,
            test_prompts=["a white hat", "a person riding a horse", "a chair"]
        ),
        12: TestConfig(
            min_similarity=0.25,
            test_prompts=["a white hat", "a person riding a horse", "a chair"]
        ),
        13: TestConfig(
            yolo_confidence=0.5,
            min_similarity=0.25,
            expansion_factor=0.20,
            test_prompts=["a white hat", "a person riding a horse", "a chair"]
        ),
        14: TestConfig(
            yolo_confidence=0.5,
            min_similarity=0.25,
            expansion_factor=0.20,
            test_prompts=["a white hat", "a person riding a horse", "a chair"]
        ),
        15: TestConfig(
            yolo_confidence=0.5,
            detection_merge_target=8,
            min_similarity=0.25,
            expansion_factor=0.20,
            context_filter_threshold=0.20,
            display_time_ms=3000,
            context_prompts=["a person riding a horse"],
            attribute_prompts=["a person wearing a white hat", "a person wearing a red shirt", 
                             "a white hat", "a chair"]
        ),
        16: TestConfig(
            yolo_confidence=0.25,
            min_similarity=0.15,
            expansion_factor=0.20,
            context_prompts=["a person riding a horse"],
            attribute_prompts=["a person wearing a white hat", "a person wearing a red shirt",
                             "a person wearing a blue shirt", "a person wearing a black shirt",
                             "a person wearing a white shirt", "a white hat", "a chair"]
        ),
        17: TestConfig(
            yolo_confidence=0.25,
            min_similarity=0.15,
            expansion_factor=0.20,
            context_prompts=["a person riding a horse"],
            attribute_prompts=["a person wearing a white hat", "a person wearing a red shirt",
                             "a person wearing a blue shirt", "a person wearing a black shirt",
                             "a person wearing a white shirt", "a white hat"]
        ),
    }
    
    # =========================================================================
    # TEST INFRASTRUCTURE HELPERS
    # =========================================================================
    
    def create_test_pipeline(factory, pipeline_id: str, tasks_config: List[tuple], 
                            edges: List[tuple]) -> tuple:
        """
        Helper to create pipeline with tasks and edges.
        
        Args:
            factory: PipelineFactory instance
            pipeline_id: Identifier for the pipeline connector
            tasks_config: List of (type, id, params) tuples
            edges: List of (from_task, to_task) tuples where tasks are by id or index
            
        Returns:
            (pipeline, tasks_dict) where tasks_dict maps task_id to task instance
        """
        pipeline = factory.create_connector("connector", pipeline_id)
        tasks = {}
        
        # Create all tasks
        for task_type, task_id, params in tasks_config:
            task = factory.create_task(task_type, task_id, params or {})
            tasks[task_id] = task
            pipeline.add_task(task)
        
        # Add edges
        for from_id, to_id in edges:
            pipeline.add_edge(tasks[from_id], tasks[to_id])
        
        return pipeline, tasks
    
    def run_test_pipeline(pipeline, collector, context: Optional[Context] = None) -> Context:
        """
        Helper to execute pipeline and return results.
        
        Args:
            pipeline: Pipeline connector to execute
            collector: Metrics collector
            context: Optional initial context (creates new if None)
            
        Returns:
            Result context after pipeline execution
        """
        runner = PipelineRunner(pipeline, collector=collector)
        if context is None:
            context = Context()
        result = runner.run(context)
        runner.shutdown()
        return result
    
    def print_pipeline_results(result: Context, data_types: List[str] = None, 
                              title: str = "Pipeline Results"):
        """
        Standardized result printing.
        
        Args:
            result: Context with pipeline results
            data_types: List of ContextDataType names to display (None = common types)
            title: Section title
        """
        if data_types is None:
            data_types = ["IMAGE", "DETECTIONS", "CROPS", "EMBEDDINGS", 
                         "PROMPT_EMBEDDINGS", "MATCHES"]
        
        print(f"\n{title}:")
        for dtype_name in data_types:
            dtype = getattr(ContextDataType, dtype_name, None)
            if dtype:
                has_data = dtype in result.data
                if has_data:
                    data = result.data[dtype]
                    if isinstance(data, list):
                        print(f"  {dtype_name}: {len(data)} items")
                    elif isinstance(data, dict):
                        print(f"  {dtype_name}: {len(data)} keys")
                    else:
                        print(f"  {dtype_name}: present")
                else:
                    print(f"  {dtype_name}: not present")
    
    def safe_import_test(test_name: str, required_modules: List[str]) -> bool:
        """
        Check if required modules are available without printing scary traces.
        
        Args:
            test_name: Name of test for error message
            required_modules: List of module names to check
            
        Returns:
            True if all modules available, False otherwise
        """
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError:
                print(f"⚠️  Skipping {test_name} (missing dependency: {module_name})")
                return False
        return True
    
    # =========================================================================
    # RESULT PRINTING UTILITIES
    # =========================================================================
    
    def print_test_header(test_num, title, goal=None, method=None, image_name=None):
        """Print standardized test header."""
        print("\n" + "=" * 60)
        print(f"[TEST {test_num}] {title}")
        print("=" * 60)
        if goal:
            print(f"Goal: {goal}")
        if method:
            print(f"Method: {method}")
        if image_name:
            print(f"Image: {image_name}")
        if goal or method or image_name:
            print("-" * 60)
    
    def print_step(step_num, description):
        """Print pipeline step."""
        print(f"\nStep {step_num}: {description}")
    
    def print_path_header(path_name, description=""):
        """Print path execution header."""
        print(f"PATH {path_name}: {description}")
        print("=" * 70)
    
    def print_results_header(title):
        """Print results section header."""
        print(f"RESULTS: {title}")
        print("=" * 70)
    
    def print_detection_count(detections):
        """Print detection count."""
        count = len(detections) if detections else 0
        print(f"  Detections: {count}")
    
    def print_similarity_scores(matches, prompts_filter=None, threshold=0.15, 
                               all_prompts=None, color_markers=False):
        """
        Print similarity scores for matches.
        
        Args:
            matches: List of match dictionaries with 'prompt_text'/'prompt', 'similarity', 'all_scores'
            prompts_filter: Optional list of prompts to filter display
            threshold: Minimum similarity threshold for checkmark
            all_prompts: Full prompt list for detailed scoring
            color_markers: Use color emojis for red/blue prompts
        """
        for match in matches:
            # Handle both old and new key formats
            prompt = match.get('prompt_text', match.get('prompt', ''))
            similarity = match.get('similarity', 0.0)
            all_scores = match.get('all_scores', [])
            detection_id = match.get('detection_index', match.get('detection_id'))
            
            if detection_id is not None:
                print(f"\nDetection #{detection_id}: '{prompt}' (best match: {similarity:.3f})")
            else:
                print(f"\n  '{prompt}' → {similarity:.3f}")
            
            # Print detailed scores if available
            if all_scores and all_prompts:
                for idx, p in enumerate(all_prompts):
                    if prompts_filter and p not in prompts_filter:
                        continue
                    
                    score = all_scores[idx] if idx < len(all_scores) else 0.0
                    marker = "✓" if score >= threshold else "✗"
                    
                    # Optional color markers for color-related prompts
                    if color_markers:
                        if "blue" in p.lower():
                            marker = "🔵"
                        elif "red" in p.lower():
                            marker = "🔴"
                        elif "black" in p.lower():
                            marker = "⚫"
                        elif "white" in p.lower():
                            marker = "⚪"
                    
                    indent = "  " if detection_id is not None else "    "
                    print(f"{indent}{marker} {score:.3f} - {p}")
    
    def print_test_footer(success=True, test_num=None):
        """Print test completion status."""
        if success:
            msg = f"✅ Test {test_num} completed!" if test_num else "✅ Test completed!"
        else:
            msg = f"❌ Test {test_num} failed!" if test_num else "❌ Test failed!"
        print(f"\n{msg}")
    
    def print_analysis_section(lines):
        """
        Print analysis section with formatted lines.
        
        Args:
            lines: List of strings or tuples (indent_level, text)
        """
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        for line in lines:
            if isinstance(line, tuple):
                indent_level, text = line
                print("  " * indent_level + text)
            else:
                print(line)
        print("=" * 70)
    
    # =========================================================================
    # TEST METHOD DEFINITIONS
    # =========================================================================
    
    def run_test_16_supersaturated_colors(factory, clip_model, semantic_provider, 
                                         config: TestConfig, interactive_mode: bool):
        """
        Test 16: Color Supersaturation for Attribute Detection
        
        Tests if enhanced color saturation (2.5x) improves CLIP's ability
        to detect color attributes like "red shirt" or "blue shirt".
        """
        print_test_header(
            16,
            "Color Supersaturation for Attribute Detection",
            goal="Test if 2.5x saturation boost improves CLIP color detection",
            method="Apply ColorEnhanceTask to Path B before CLIP encoding",
            image_name="trail riders (people with white hats)"
        )
        
        try:
            # Check if CLIP is available
            if clip_model is None or semantic_provider is None:
                print("⚠️  CLIPModel or SemanticProvider not initialized, skipping Test 16")
                print("⚠️  Skipping Test 16 (dependencies not available): CLIPModel initialization failed")
                return
            
            # Get prompts from config
            CONTEXT_PROMPTS = config.context_prompts
            ATTRIBUTE_PROMPTS = config.attribute_prompts
            ALL_PROMPTS = config.all_prompts
            
            print(f"\nContext prompts: {CONTEXT_PROMPTS}")
            print(f"Attribute prompts: {ATTRIBUTE_PROMPTS}")
            
            # Create custom image loader for trail-riders.jpg
            img_path = os.path.join("src", "camera", "trail-riders.jpg")
            camera_task16 = CustomImageTask(img_path, task_id="camera16")
            
            # Create YOLO detector directly
            yolo_det16 = YoloV8Detector(
                model_name=config.yolo_model
            )
            yolo_det16.start()
            detector_task16 = DetectorTaskWrapper(yolo_det16, task_id="detector16")
            
            # Clusterer for context path
            clusterer_task16 = factory.create_task("yolo_detector", "clusterer16", {
                "type": "clusterer",
                "source": "yolo16",
                "semantic_provider": "semantic_provider"
            })
            clusterer_task16.detector = ObjectClusterer(
                source=detector_task16.detector,
                semantic_provider=semantic_provider,
                filter_prompts=CONTEXT_PROMPTS,
                filter_threshold=0.20,
                proximity_weight=0.8,
                size_weight=0.5
            )
            clusterer_task16.detector.start()
            
            # Expander (20%)
            expander_task16 = DetectionExpanderTask(
                expansion_factor=0.20,
                task_id="expander16"
            )
            
            # Color enhancement (2.5x saturation)
            enhance_task16 = ColorEnhanceTask(
                saturation_factor=2.5,
                task_id="color_enhance16"
            )
            
            # CLIP tasks
            clip_task16 = factory.create_task("clip_vision", "clip16", {})
            clip_task16.clip_model = clip_model
            
            compare_task16 = factory.create_task("clip_compare", "compare16", {
                "min_similarity": "0.15"
            })
            
            # Import and create prompt embedding sources directly
            from .tasks.prompt_embedding_source_task import PromptEmbeddingSourceTask
            
            prompt_task_context16 = PromptEmbeddingSourceTask(
                task_id="prompt_context16",
                prompts=CONTEXT_PROMPTS,
                clip_model=clip_model
            )
            
            prompt_task_attr16 = PromptEmbeddingSourceTask(
                task_id="prompt_attr16",
                prompts=ATTRIBUTE_PROMPTS,
                clip_model=clip_model
            )
            
            # Execute pipeline
            print("\n" + "=" * 70)
            print("EXECUTING DUAL-PATH COMPARISON: NORMAL vs 2.5x SATURATION")
            print("=" * 70)
            
            # Step 1: Capture image
            print_step(1, "Capture image")
            ctx16 = Context()
            ctx16 = camera_task16.run(ctx16)
            
            # Step 2: YOLO detection
            print_step(2, "YOLO detection")
            ctx16 = detector_task16.run(ctx16)
            yolo_dets = ctx16.data.get(ContextDataType.DETECTIONS, [])
            print_detection_count(yolo_dets)
            
            # Filter to person detections only
            person_dets = [d for d in yolo_dets if d.object_category.lower() == "person"]
            print(f"  Person detections: {len(person_dets)}")
            
            # Path A: Normal (baseline - no saturation boost)
            print("\n" + "=" * 70)
            print("PATH A: NORMAL COLOR (Baseline)")
            print("=" * 70)
            ctx_normal = Context()
            ctx_normal = prompt_task_attr16.run(ctx_normal)  # Add prompt embeddings
            ctx_normal.data[ContextDataType.IMAGE] = ctx16.data[ContextDataType.IMAGE]
            ctx_normal.data[ContextDataType.DETECTIONS] = person_dets
            
            ctx_normal = expander_task16.run(ctx_normal)
            # No color enhancement here
            ctx_normal = clip_task16.run(ctx_normal)
            ctx_normal = compare_task16.run(ctx_normal)
            
            normal_matches = ctx_normal.data.get(ContextDataType.MATCHES, [])
            print(f"  Matches: {len(normal_matches)}")
            
            # Path B: Enhanced (2.5x saturation)
            print("\n" + "=" * 70)
            print("PATH B: 2.5x SATURATION BOOST")
            print("=" * 70)
            ctx_enhanced = Context()
            ctx_enhanced = prompt_task_attr16.run(ctx_enhanced)  # Add prompt embeddings
            ctx_enhanced.data[ContextDataType.IMAGE] = ctx16.data[ContextDataType.IMAGE]
            ctx_enhanced.data[ContextDataType.DETECTIONS] = person_dets
            
            ctx_enhanced = expander_task16.run(ctx_enhanced)
            ctx_enhanced = enhance_task16.run(ctx_enhanced)  # Apply 2.5x saturation
            ctx_enhanced = clip_task16.run(ctx_enhanced)
            ctx_enhanced = compare_task16.run(ctx_enhanced)
            
            enhanced_matches = ctx_enhanced.data.get(ContextDataType.MATCHES, [])
            print(f"  Matches: {len(enhanced_matches)}")
            
            # Display results side-by-side
            print("\n" + "=" * 70)
            print("RESULTS COMPARISON")
            print("=" * 70)
            
            print("\n--- PATH A: NORMAL COLOR (Baseline) ---")
            print_similarity_scores(
                normal_matches, 
                all_prompts=ATTRIBUTE_PROMPTS,
                threshold=0.15,
                color_markers=True
            )
            
            print("\n--- PATH B: 2.5x SATURATION BOOST ---")
            print_similarity_scores(
                enhanced_matches,
                all_prompts=ATTRIBUTE_PROMPTS,
                threshold=0.15,
                color_markers=True
            )
            
            # Cleanup
            if clusterer_task16.detector:
                clusterer_task16.detector.stop()
            detector_task16.detector.stop()
            
            print_test_footer(success=True, test_num=16)
            
        except Exception as e:
            print(f"❌ Test 16 failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_test_17_color_channel_swap(factory, clip_model, semantic_provider, 
                                       config: TestConfig, interactive_mode: bool):
        """
        Test 17: Red/Blue Channel Swap
        
        Tests if CLIP follows actual pixel colors by swapping R/B channels.
        Blue shirt becomes red after swap - does CLIP follow the pixels?
        """
        print_test_header(
            17,
            "Red/Blue Channel Swap",
            goal="Test if CLIP follows pixel colors or uses context",
            method="Swap R/B channels - test color perception",
            image_name="trail riders (people with white hats)"
        )
        
        try:
            # Check if CLIP is available
            if clip_model is None or semantic_provider is None:
                print("⚠️  CLIPModel or SemanticProvider not initialized, skipping Test 17")
                print("⚠️  Skipping Test 17 (dependencies not available): CLIPModel initialization failed")
                return
            
            # Get prompts from config
            CONTEXT_PROMPTS = config.context_prompts
            ATTRIBUTE_PROMPTS = config.attribute_prompts
            ALL_PROMPTS = config.all_prompts
            
            print(f"\nAttribute prompts: {ATTRIBUTE_PROMPTS}")
            print("⚠️  R/B SWAP: Blue shirt will appear RED!")
            
            # Create custom image loader for trail-riders.jpg
            img_path = os.path.join("src", "camera", "trail-riders.jpg")
            camera_task17 = CustomImageTask(img_path, task_id="camera17")
            
            # Create YOLO detector directly
            yolo_det17 = YoloV8Detector(
                model_name=config.yolo_model
            )
            yolo_det17.start()
            detector_task17 = DetectorTaskWrapper(yolo_det17, task_id="detector17")
            
            # Clusterer for context path
            
            # Expander
            expander_task17 = DetectionExpanderTask(
                expansion_factor=config.expansion_factor,
                task_id="expander17"
            )
            
            # Color swap (R ← B)
            color_swap_task17 = ColorSwapTask(
                swap_channels=(0, 2),  # Swap red and blue
                task_id="rb_swap17"
            )
            
            # CLIP tasks
            clip_task17 = factory.create_task("clip_vision", "clip17", {})
            clip_task17.clip_model = clip_model
            
            compare_task17 = factory.create_task("clip_compare", "compare17", {
                "min_similarity": str(config.min_similarity)
            })
            
            # Import and create prompt embedding source directly
            from .tasks.prompt_embedding_source_task import PromptEmbeddingSourceTask
            
            prompt_task_attr17 = PromptEmbeddingSourceTask(
                task_id="prompt_attr17",
                prompts=ATTRIBUTE_PROMPTS,
                clip_model=clip_model
            )
            
            # Execute pipeline
            print("\n" + "=" * 70)
            print("EXECUTING DUAL-PATH COMPARISON: NORMAL vs R/B SWAPPED")
            print("=" * 70)
            
            # Step 1: Capture image
            print_step(1, "Capture image")
            ctx17 = Context()
            ctx17 = camera_task17.run(ctx17)
            
            # Step 2: YOLO detection
            print_step(2, "YOLO detection")
            ctx17 = detector_task17.run(ctx17)
            yolo_dets = ctx17.data.get(ContextDataType.DETECTIONS, [])
            print_detection_count(yolo_dets)
            
            # Filter to person detections only
            person_dets = [d for d in yolo_dets if d.object_category.lower() == "person"]
            print(f"  Person detections: {len(person_dets)}")
            
            # Path A: Normal (baseline - no color swap)
            print("\n" + "=" * 70)
            print("PATH A: NORMAL COLOR (Baseline)")
            print("=" * 70)
            ctx_normal = Context()
            ctx_normal = prompt_task_attr17.run(ctx_normal)  # Add prompt embeddings
            ctx_normal.data[ContextDataType.IMAGE] = ctx17.data[ContextDataType.IMAGE]
            ctx_normal.data[ContextDataType.DETECTIONS] = person_dets
            
            ctx_normal = expander_task17.run(ctx_normal)
            # No color swap here
            ctx_normal = clip_task17.run(ctx_normal)
            ctx_normal = compare_task17.run(ctx_normal)
            
            normal_matches = ctx_normal.data.get(ContextDataType.MATCHES, [])
            print(f"  Matches: {len(normal_matches)}")
            
            # DEBUG: Save normal crops for comparison
            debug_dir = "/tmp/vlmchat_test17_debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save normal crops (using a temp expander since we need crops)
            from .tasks.detection_expander_task import DetectionExpanderTask as TempExpander
            temp_ctx = Context()
            temp_ctx.data[ContextDataType.IMAGE] = ctx17.data[ContextDataType.IMAGE]
            temp_ctx.data[ContextDataType.DETECTIONS] = person_dets
            temp_expander = TempExpander(expansion_factor=0.20, task_id="temp_exp")
            temp_ctx = temp_expander.run(temp_ctx)
            
            for idx, det in enumerate(temp_ctx.data.get(ContextDataType.DETECTIONS, [])):
                if hasattr(det, 'metadata') and det.metadata and 'expanded_crop' in det.metadata:
                    crop = det.metadata['expanded_crop']
                    crop.save(os.path.join(debug_dir, f"detection_{idx}_normal.png"))
            
            # Path B: R/B swapped (blue becomes red, red becomes blue)
            print("\n" + "=" * 70)
            print("PATH B: R/B CHANNEL SWAP (Blue ↔ Red)")
            print("=" * 70)
            ctx_swapped = Context()
            ctx_swapped = prompt_task_attr17.run(ctx_swapped)  # Add prompt embeddings
            ctx_swapped.data[ContextDataType.IMAGE] = ctx17.data[ContextDataType.IMAGE]
            ctx_swapped.data[ContextDataType.DETECTIONS] = person_dets
            
            ctx_swapped = expander_task17.run(ctx_swapped)
            ctx_swapped = color_swap_task17.run(ctx_swapped)  # Apply R/B swap
            ctx_swapped = clip_task17.run(ctx_swapped)
            ctx_swapped = compare_task17.run(ctx_swapped)
            
            swapped_matches = ctx_swapped.data.get(ContextDataType.MATCHES, [])
            print(f"  Matches: {len(swapped_matches)}")
            
            # DEBUG: Save swapped crops to verify color swap is working
            debug_dir = "/tmp/vlmchat_test17_debug"
            os.makedirs(debug_dir, exist_ok=True)
            person_dets_with_crops = ctx_swapped.data.get(ContextDataType.DETECTIONS, [])
            for idx, det in enumerate(person_dets_with_crops):
                if hasattr(det, 'metadata') and det.metadata and 'enhanced_crop' in det.metadata:
                    crop = det.metadata['enhanced_crop']
                    crop.save(os.path.join(debug_dir, f"detection_{idx}_swapped.png"))
            print(f"\n🔍 DEBUG: Swapped crops saved to {debug_dir}/")
            
            # Display results side-by-side
            print("\n" + "=" * 70)
            print("RESULTS COMPARISON")
            print("=" * 70)
            print("\n⚠️  GROUND TRUTH (Visual Inspection):")
            print("  Detection #0: Black shirt person")
            print("  Detection #1: Blue shirt person")
            print("  Detection #2: White shirt person")
            print("\n" + "=" * 70)
            
            print("\n--- PATH A: NORMAL COLOR (Baseline) ---")
            print_similarity_scores(
                normal_matches,
                all_prompts=ATTRIBUTE_PROMPTS,
                threshold=0.15,
                color_markers=True
            )
            
            print("\n--- PATH B: R/B CHANNEL SWAP (Blue ↔ Red) ---")
            print_similarity_scores(
                swapped_matches,
                all_prompts=ATTRIBUTE_PROMPTS,
                threshold=0.15,
                color_markers=True
            )
            
            # Analysis
            print_analysis_section([
                "Expected if CLIP follows pixel colors:",
                (1, "→ PATH B 'red shirt' scores should INCREASE (blue pixels become red)"),
                (1, "→ PATH B 'blue shirt' scores should DECREASE (blue pixels become red)"),
                "",
                "Expected if CLIP uses semantic context:",
                (1, "→ Scores remain similar between paths (CLIP recognizes clothing despite color change)")
            ])
            
            # Cleanup
            detector_task17.detector.stop()
            
            print_test_footer(success=True, test_num=17)
            
        except Exception as e:
            print(f"❌ Test 17 failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_test_1_linear_pipeline(factory, collector, config: TestConfig, session, interactive_mode: bool = False):
        """Test 1: Linear Pipeline - Source → Detector → Embedder"""
        print_test_header(1, "Linear Pipeline: Source → Detector → Embedder")
        
        # Create tasks
        pipeline = Connector("linear_pipeline")
        source = ImageSourceTask()
        detector = DetectorTask("detector")
        embedder = EmbeddingTask()
        
        pipeline.add_task(source)
        pipeline.add_task(detector)
        pipeline.add_task(embedder)
        pipeline.add_edge(source, detector)
        pipeline.add_edge(detector, embedder)
        
        # Execute
        result = run_test_pipeline(pipeline, collector)
        print_pipeline_results(result, ["IMAGE", "CROPS", "EMBEDDINGS"])
        
        # Show metrics
        print(f"\nMetrics Summary (Test 1):")
        for ts_name, inst in session._instruments:
            exported = inst.export()
            print(f"  {exported['name']}: {json.dumps({k: v for k, v in exported.items() if k not in ['type', 'name', 'binding_keys']}, indent=4)}")
        
        print_test_footer(success=True, test_num=1)
    
    def run_test_2_branching_pipeline(factory, collector, config: TestConfig, session, interactive_mode: bool = False):
        """Test 2: Branching Pipeline with Split/Merge"""
        print_test_header(2, "Branching Pipeline: Source → Split → [DetectorA, DetectorB] → Merge → Embedder")
        
        # Create tasks
        branch_pipeline = Connector("branch_pipeline")
        source = ImageSourceTask()
        split_connector = Connector("splitter")
        detector_a = DetectorTask("detector_a")
        detector_b = DetectorTask("detector_b")
        merge_connector = Connector("merger")
        embedder = EmbeddingTask()
        
        # Set connector contracts
        split_connector.input_contract = {ContextDataType.IMAGE: list}
        split_connector.output_contract = {ContextDataType.IMAGE: list}
        merge_connector.input_contract = {ContextDataType.CROPS: list}
        merge_connector.output_contract = {ContextDataType.CROPS: list}
        
        # Build pipeline
        branch_pipeline.add_task(source)
        branch_pipeline.add_task(split_connector)
        branch_pipeline.add_task(detector_a)
        branch_pipeline.add_task(detector_b)
        branch_pipeline.add_task(merge_connector)
        branch_pipeline.add_task(embedder)
        
        branch_pipeline.add_edge(source, split_connector)
        split_connector.output_tasks = [detector_a, detector_b]
        branch_pipeline.add_edge(split_connector, detector_a)
        branch_pipeline.add_edge(split_connector, detector_b)
        branch_pipeline.add_edge(detector_a, merge_connector)
        branch_pipeline.add_edge(detector_b, merge_connector)
        branch_pipeline.add_edge(merge_connector, embedder)
        
        # Execute
        result = run_test_pipeline(branch_pipeline, collector)
        print_pipeline_results(result, ["IMAGE", "CROPS", "EMBEDDINGS"])
        
        # Show metrics
        print(f"\nMetrics Summary (Test 2):")
        for ts_name, inst in session._instruments:
            exported = inst.export()
            print(f"  {exported['name']}: {json.dumps({k: v for k, v in exported.items() if k not in ['type', 'name', 'binding_keys']}, indent=4)}")
        
        print_test_footer(success=True, test_num=2)
    
    def run_test_3_first_complete(factory, collector, config: TestConfig, session, interactive_mode: bool = False):
        """Test 3: FirstCompleteConnector"""
        from .connectors import FirstCompleteConnector
        
        print_test_header(3, "FirstCompleteConnector: Source → Split → [DetectorC, DetectorD] → FirstComplete")
        
        # Create tasks
        pipeline = Connector("first_complete_pipeline")
        source = ImageSourceTask()
        split_connector = Connector("splitter3")
        detector_c = DetectorTask("detector_c")
        detector_d = DetectorTask("detector_d")
        first_complete = FirstCompleteConnector("first_complete")
        
        # Set contracts
        split_connector.input_contract = {ContextDataType.IMAGE: list}
        split_connector.output_contract = {ContextDataType.IMAGE: list}
        first_complete.input_contract = {ContextDataType.CROPS: list}
        first_complete.output_contract = {ContextDataType.CROPS: list}
        
        # Build pipeline
        pipeline.add_task(source)
        pipeline.add_task(split_connector)
        pipeline.add_task(detector_c)
        pipeline.add_task(detector_d)
        pipeline.add_task(first_complete)
        
        pipeline.add_edge(source, split_connector)
        split_connector.output_tasks = [detector_c, detector_d]
        pipeline.add_edge(split_connector, detector_c)
        pipeline.add_edge(split_connector, detector_d)
        pipeline.add_edge(detector_c, first_complete)
        pipeline.add_edge(detector_d, first_complete)
        
        # Execute
        result = run_test_pipeline(pipeline, collector)
        
        print(f"\nFinal context (should only have first detector's crops):")
        print(f"  Images: {result.data.get(ContextDataType.IMAGE, [])}")
        print(f"  Crops: {result.data.get(ContextDataType.CROPS, [])}")
        
        print_test_footer(success=True, test_num=3)
    
    def run_test_4_ordered_merge(factory, collector, config: TestConfig, session, interactive_mode: bool = False):
        """Test 4: OrderedMergeConnector"""
        from .connectors import OrderedMergeConnector
        
        print_test_header(4, "OrderedMergeConnector: Source → Split → [DetectorE, DetectorF] → OrderedMerge(2,1)")
        
        # Create tasks
        pipeline = Connector("ordered_merge_pipeline")
        source = ImageSourceTask()
        split_connector = Connector("splitter4")
        detector_e = DetectorTask("detector_e")
        detector_f = DetectorTask("detector_f")
        ordered_merge = OrderedMergeConnector("ordered_merge")
        ordered_merge.configure({"order": "2,1"})  # Reverse order
        
        # Set contracts
        split_connector.input_contract = {ContextDataType.IMAGE: list}
        split_connector.output_contract = {ContextDataType.IMAGE: list}
        ordered_merge.input_contract = {ContextDataType.CROPS: list}
        ordered_merge.output_contract = {ContextDataType.CROPS: list}
        
        # Build pipeline
        pipeline.add_task(source)
        pipeline.add_task(split_connector)
        pipeline.add_task(detector_e)
        pipeline.add_task(detector_f)
        pipeline.add_task(ordered_merge)
        
        pipeline.add_edge(source, split_connector)
        split_connector.output_tasks = [detector_e, detector_f]
        pipeline.add_edge(split_connector, detector_e)
        pipeline.add_edge(split_connector, detector_f)
        pipeline.add_edge(detector_e, ordered_merge)
        pipeline.add_edge(detector_f, ordered_merge)
        
        # Execute
        result = run_test_pipeline(pipeline, collector)
        
        print(f"\nFinal context (detector_f crops should appear before detector_e):")
        print(f"  Images: {result.data.get(ContextDataType.IMAGE, [])}")
        print(f"  Crops: {result.data.get(ContextDataType.CROPS, [])}")
        print(f"  Expected order: ['detector_f_crop_0', 'detector_f_crop_1', 'detector_e_crop_0', 'detector_e_crop_1']")
        
        print_test_footer(success=True, test_num=4)
    
    def run_test_5_pipeline_factory(factory, collector, config: TestConfig, session, interactive_mode: bool = False):
        """Test 5: PipelineFactory - Dynamic task creation"""
        from .connectors import FirstCompleteConnector, OrderedMergeConnector
        
        print_test_header(5, "PipelineFactory: Dynamic task creation")
        
        # Register tasks in a local factory for this test
        test_factory = PipelineFactory()
        test_factory.register_task("image_source", ImageSourceTask)
        test_factory.register_task("detector", DetectorTask)
        test_factory.register_task("embedder", EmbeddingTask)
        test_factory.register_connector("connector", Connector)
        test_factory.register_connector("first_complete", FirstCompleteConnector)
        test_factory.register_connector("ordered_merge", OrderedMergeConnector)
        
        print(f"\nRegistered tasks: {test_factory.list_tasks()}")
        print(f"Registered connectors: {test_factory.list_connectors()}")
        
        # Build pipeline using factory
        pipeline = test_factory.create_connector("connector", "factory_pipeline")
        source = test_factory.create_task("image_source", "source5")
        detector = test_factory.create_task("detector", "detector5")
        embedder = test_factory.create_task("embedder", "embedder5")
        
        pipeline.add_task(source)
        pipeline.add_task(detector)
        pipeline.add_task(embedder)
        pipeline.add_edge(source, detector)
        pipeline.add_edge(detector, embedder)
        
        # Execute
        result = run_test_pipeline(pipeline, collector)
        print_pipeline_results(result, ["IMAGE", "CROPS", "EMBEDDINGS"])
        
        print_test_footer(success=True, test_num=5)
    
    def run_test_6_factory_config(factory, collector, config: TestConfig, session, interactive_mode: bool = False):
        """Test 6: PipelineFactory with Configured OrderedMergeConnector"""
        from .connectors import OrderedMergeConnector
        
        print_test_header(6, "PipelineFactory: Configured OrderedMergeConnector")
        
        # Create tasks using factory
        pipeline = factory.create_connector("connector", "factory_pipeline2")
        source = factory.create_task("image_source", "source6")
        splitter = factory.create_connector("connector", "splitter6")
        detector_g = factory.create_task("detector", "detector_g")
        detector_h = factory.create_task("detector", "detector_h")
        merger = factory.create_connector("ordered_merge", "merger6", params={"order": "2,1"})
        
        # Set contracts
        splitter.input_contract = {ContextDataType.IMAGE: list}
        splitter.output_contract = {ContextDataType.IMAGE: list}
        merger.input_contract = {ContextDataType.CROPS: list}
        merger.output_contract = {ContextDataType.CROPS: list}
        
        # Build pipeline
        pipeline.add_task(source)
        pipeline.add_task(splitter)
        pipeline.add_task(detector_g)
        pipeline.add_task(detector_h)
        pipeline.add_task(merger)
        
        pipeline.add_edge(source, splitter)
        splitter.output_tasks = [detector_g, detector_h]
        pipeline.add_edge(splitter, detector_g)
        pipeline.add_edge(splitter, detector_h)
        pipeline.add_edge(detector_g, merger)
        pipeline.add_edge(detector_h, merger)
        
        # Execute
        result = run_test_pipeline(pipeline, collector)
        
        print(f"\nFinal context (detector_h crops before detector_g):")
        print(f"  Crops: {result.data.get(ContextDataType.CROPS, [])}")
        print(f"  Expected: ['detector_h_crop_0', 'detector_h_crop_1', 'detector_g_crop_0', 'detector_g_crop_1']")
        
        print_test_footer(success=True, test_num=6)
    
    def run_test_7_configure(factory, collector, config: TestConfig, session, interactive_mode: bool = False):
        """Test 7: CameraTask + DetectorTask with configure()"""
        print_test_header(7, "CameraTask + DetectorTask with configure()")
        
        try:
            # Create camera and detector via factory with configure()
            cam_configured = factory.create_task("camera", "cam_config", {
                "type": "none",
                "device": "0",
                "resolution": "640x480"
            })
            det_configured = factory.create_task("yolo_detector", "det_config", {
                "type": "yolo_cpu",
                "model": config.yolo_model,
                "confidence": str(config.yolo_confidence)
            })
            
            print(f"Camera created via configure(): {cam_configured.camera is not None}")
            print(f"Detector created via configure(): {det_configured.detector is not None}")
            
            # Build and run pipeline
            pipeline = factory.create_connector("connector", "config_pipeline")
            pipeline.add_task(cam_configured)
            pipeline.add_task(det_configured)
            pipeline.add_edge(cam_configured, det_configured)
            
            result = run_test_pipeline(pipeline, collector)
            
            print(f"\nPipeline with configured tasks completed:")
            print(f"  Has IMAGE: {ContextDataType.IMAGE in result.data}")
            print(f"  Has DETECTIONS: {ContextDataType.DETECTIONS in result.data}")
            if ContextDataType.DETECTIONS in result.data:
                dets = result.data[ContextDataType.DETECTIONS]
                print(f"  Detection count: {len(dets)}")
                if dets:
                    print(f"\n  Detections found:")
                    for i, det in enumerate(dets):
                        print(f"    [{i}] {det.object_category} @ confidence {det.conf:.2f}")
            
            print_test_footer(success=True, test_num=7)
            
        except ImportError as e:
            print(f"⚠️  Skipping Test 7 (missing dependency: {getattr(e, 'name', str(e))})")
        except Exception as e:
            print(f"❌ Test 7 failed: {e}")
            if hasattr(__builtins__, 'args') and getattr(args, 'log_level', 'WARNING') == 'DEBUG':
                import traceback
                traceback.print_exc()
    
    def run_test_8_interactive_yolo(factory, collector, config: TestConfig, session, interactive_mode: bool = False):
        """Test 8: Real Pipeline - NoneCamera → YOLO → DetectionViewer"""
        print_test_header(8, "Real Pipeline: NoneCamera → YOLO → DetectionViewer")
        
        try:
            from object_detector.detection_viewer import DetectionViewer
            from object_detector.image_viewer import ImageViewer
            
            # Create viewer for display (will passthrough if display_time_ms=0)
            viewer = ImageViewer(window_name="Pipeline Test - Detection Viewer")
            
            # Create pipeline using factory
            pipeline = factory.create_connector("connector", "test_pipeline")
            
            # Create camera and detector
            camera_task = factory.create_task("camera", "none_camera", {"type": "none", "device": "0"})
            detector_task = factory.create_task("yolo_detector", "yolo_det", {
                "type": "yolo_cpu",
                "model": config.yolo_model,
                "confidence": str(config.yolo_confidence)
            })
            
            # Wrap detector with DetectionViewer
            detection_viewer = None
            if detector_task.detector:
                detection_viewer = DetectionViewer(
                    source=detector_task.detector,
                    viewer=viewer,
                    display_time_ms=config.display_time_ms
                )
                detection_viewer.start()
                print(f"DetectionViewer created and started")
            
            # Build and execute pipeline
            pipeline.add_task(camera_task)
            pipeline.add_task(detector_task)
            pipeline.add_edge(camera_task, detector_task)
            
            print("Pipeline structure:")
            print(f"  {camera_task.task_id} → {detector_task.task_id}")
            
            result = run_test_pipeline(pipeline, collector)
            
            print(f"\nPipeline execution completed:")
            print(f"  Has IMAGE: {ContextDataType.IMAGE in result.data}")
            print(f"  Has DETECTIONS: {ContextDataType.DETECTIONS in result.data}")
            
            if ContextDataType.DETECTIONS in result.data:
                dets = result.data[ContextDataType.DETECTIONS]
                print(f"  Detection count: {len(dets)}")
                
                if dets:
                    print(f"\n  Detections found:")
                    for i, det in enumerate(dets[:5]):
                        print(f"    [{i}] {det.object_category} @ confidence {det.conf:.2f}")
                    if len(dets) > 5:
                        print(f"    ... and {len(dets) - 5} more")
                    
                    # Display with viewer (will passthrough if display_time_ms=0)
                    if ContextDataType.IMAGE in result.data and detection_viewer and config.display_time_ms > 0:
                        image_list = result.data[ContextDataType.IMAGE]
                        image = image_list[0] if isinstance(image_list, list) and image_list else image_list
                        print(f"\n  Visualizing detections (display for {config.display_time_ms/1000:.1f} seconds)...")
                        detection_viewer._detect_internal(image, dets)
                        
                        import time
                        start_time = time.time()
                        display_seconds = config.display_time_ms / 1000
                        while time.time() - start_time < display_seconds and viewer.is_visible():
                            viewer.show(wait_ms=30)
            
            # Cleanup
            if detection_viewer:
                detection_viewer.stop()
            viewer.close()
            
            print_test_footer(success=True, test_num=8)
            
        except ImportError as e:
            print(f"⚠️  Skipping Test 8 (missing dependency: {getattr(e, 'name', str(e))})")
        except Exception as e:
            print(f"❌ Test 8 failed: {e}")
            if hasattr(__builtins__, 'args') and getattr(args, 'log_level', 'WARNING') == 'DEBUG':
                import traceback
                traceback.print_exc()
    
    def run_test_9_interactive_clusterer(factory, collector, clip_model, semantic_provider, 
                                         config: TestConfig, interactive_mode: bool = False):
        """Test 9: NoneCamera → YOLO → Clusterer → DetectionViewer"""
        print_test_header(9, "NoneCamera → YOLO → Clusterer → DetectionViewer")
        
        try:
            # Check if semantic provider is available
            if semantic_provider is None:
                print("⚠️  SemanticProvider not initialized, skipping Test 9")
                print("⚠️  Skipping Test 9 (dependencies not available): SemanticProvider initialization failed")
                return
            
            # TODO: Implement full test with clusterer and viewer
            print("⚠️  Test 9 requires full implementation - placeholder for now")
            print_test_footer(success=True, test_num=9)
            
        except Exception as e:
            print(f"❌ Test 9 failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_test_10_full_pipeline(factory, collector, clip_model, semantic_provider, 
                                   config: TestConfig, interactive_mode: bool = False):
        """Test 10: Full Pipeline YOLO → [Clusterer, Pass] → DetectionMerge → ClipVision"""
        print_test_header(10, "Full Pipeline: YOLO → [Clusterer, Pass] → DetectionMerge(8) → ClipVision")
        
        # Check if CLIP dependencies are available
        if clip_model is None or semantic_provider is None:
            print("⚠️  CLIPModel or SemanticProvider not initialized, skipping Test 10")
            print("⚠️  Skipping Test 10 (dependencies not available): CLIPModel initialization failed")
            return
        
        # Load configuration from config.json
        from utils.config import VLMChatConfig
        try:
            vlm_config = VLMChatConfig.load_from_file("config.json")
            clusterer_config = vlm_config.clusterer
        except Exception as e:
            print(f"Warning: Could not load config.json, using defaults: {e}")
            from utils.config import ClustererConfig
            clusterer_config = ClustererConfig()
        
        # Load clusterer configuration
        MAX_CLUSTERS = clusterer_config.max_clusters
        MERGE_THRESHOLD = clusterer_config.merge_threshold
        PROXIMITY_WEIGHT = clusterer_config.proximity_weight
        SIZE_WEIGHT = clusterer_config.size_weight
        SEMANTIC_PAIR_WEIGHT = clusterer_config.semantic_pair_weight
        SEMANTIC_SINGLE_WEIGHT = clusterer_config.semantic_single_weight
        
        try:
            from object_detector.detection_viewer import DetectionViewer
            from object_detector.image_viewer import ImageViewer
            
            # Validate dependencies
            if clip_model is None or semantic_provider is None:
                print("⚠️  CLIPModel or SemanticProvider not initialized, skipping Test 10")
                raise ImportError("CLIPModel initialization failed")
            
            print(f"\nTest 10 Configuration:")
            print(f"  YOLO Model: {config.yolo_model}")
            print(f"  YOLO Confidence: {config.yolo_confidence}")
            print(f"  Max Clusters: {MAX_CLUSTERS}")
            print(f"  Merge Threshold: {MERGE_THRESHOLD}")
            print(f"  Detection Merge Target: {config.detection_merge_target}")
            print(f"  Display Time: {config.display_time_ms}ms {'(wait for keypress)' if config.display_time_ms == -1 else '(skip display)' if config.display_time_ms == 0 else ''}")
            
            # Create viewer for display (will passthrough if display_time_ms=0)
            viewer = ImageViewer(window_name="Pipeline Test 10 - Merged & Embedded")
            
            # Create camera (NoneCamera type)
            camera_task10 = factory.create_task("camera", "none_camera10", {
                "type": "none",
                "device": "0"
            })
            
            # Create detector (YOLO CPU)
            detector_task10 = factory.create_task("yolo_detector", "yolo_det10", {
                "type": "yolo_cpu",
                "model": config.yolo_model,
                "confidence": str(config.yolo_confidence)
            })
            
            # Branch 0: Clusterer
            clusterer_task10 = factory.create_task("yolo_detector", "clusterer10", {
                "type": "clusterer",
                "source": "yolo_det10",
                "semantic_provider": "semantic_provider",
                "max_clusters": str(MAX_CLUSTERS),
                "merge_threshold": str(MERGE_THRESHOLD),
                "proximity_weight": str(PROXIMITY_WEIGHT),
                "size_weight": str(SIZE_WEIGHT),
                "semantic_pair_weight": str(SEMANTIC_PAIR_WEIGHT),
                "semantic_single_weight": str(SEMANTIC_SINGLE_WEIGHT)
            })
            
            # Branch 1: Pass (bypass)
            pass_task10 = factory.create_task("pass", "pass10")
            
            # Detection merge connector (merges both branches)
            merge_connector = factory.create_connector("detection_merge", "merge10", {
                "target_count": str(config.detection_merge_target)
            })
            merge_connector.target_count = config.detection_merge_target
            
            # CLIP vision task
            clip_task10 = factory.create_task("clip_vision", "clip10", {
                "clip_model": "clip_model"
            })
            
            # Configure clusterer with actual objects
            clusterer_task10.detector = None
            clusterer_task10.configure({
                "type": "clusterer",
                "source": detector_task10.detector,
                "semantic_provider": semantic_provider,
                "max_clusters": str(MAX_CLUSTERS),
                "merge_threshold": str(MERGE_THRESHOLD),
                "proximity_weight": str(PROXIMITY_WEIGHT),
                "size_weight": str(SIZE_WEIGHT),
                "semantic_pair_weight": str(SEMANTIC_PAIR_WEIGHT),
                "semantic_single_weight": str(SEMANTIC_SINGLE_WEIGHT)
            })
            
            # Configure CLIP task with actual model
            clip_task10.clip_model = clip_model
            
            # Execute pipeline manually step by step
            print("\nExecuting pipeline...")
            
            # Step 1: Camera
            ctx10 = Context()
            ctx10 = camera_task10.run(ctx10)
            print(f"  Step 1 (Camera): IMAGE={ContextDataType.IMAGE in ctx10.data}")
            
            # Step 2: YOLO detector
            ctx10 = detector_task10.run(ctx10)
            yolo_dets = ctx10.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 2 (YOLO): {len(yolo_dets)} detections")
            
            # Step 3: Fan-out to Clusterer and Pass
            immutable_cache = {}
            split_contexts = ctx10.split(2, immutable_cache)
            
            # Branch 0: Clusterer
            ctx_cluster = clusterer_task10.run(split_contexts[0])
            cluster_dets = ctx_cluster.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 3a (Clusterer): {len(cluster_dets)} detections")
            
            # Branch 1: Pass (returns original)
            ctx_pass = pass_task10.run(split_contexts[1])
            pass_dets = ctx_pass.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 3b (Pass): {len(pass_dets)} detections")
            
            # Step 4: Merge both branches
            ctx10 = merge_connector.merge_strategy([ctx_cluster, ctx_pass])
            merged_dets = ctx10.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 4 (Merge): {len(merged_dets)} detections")
            
            # Step 5: CLIP vision
            ctx10 = clip_task10.run(ctx10)
            embeddings = ctx10.data.get(ContextDataType.EMBEDDINGS, [])
            print(f"  Step 5 (CLIP): {len(embeddings)} embeddings")
            
            result10 = ctx10
            
            print(f"\nPipeline execution completed:")
            print(f"  Has IMAGE: {ContextDataType.IMAGE in result10.data}")
            print(f"  Has DETECTIONS: {ContextDataType.DETECTIONS in result10.data}")
            print(f"  Has EMBEDDINGS: {ContextDataType.EMBEDDINGS in result10.data}")
            
            if ContextDataType.DETECTIONS in result10.data:
                dets = result10.data[ContextDataType.DETECTIONS]
                print(f"  Detection count (merged): {len(dets)}")
                
                if dets:
                    print(f"\n  Merged detections:")
                    for i, det in enumerate(dets):
                        children_str = f" [{len(det.children)} children]" if det.children else ""
                        print(f"    [{i}] ID={det.id} {det.object_category} @ {det.conf:.2f}{children_str}")
            
            if ContextDataType.EMBEDDINGS in result10.data:
                embeddings = result10.data[ContextDataType.EMBEDDINGS]
                print(f"\n  CLIP Embeddings generated: {len(embeddings)}")
                if embeddings:
                    print(f"    First embedding shape: {embeddings[0].shape}")
                    print(f"    First embedding dtype: {embeddings[0].dtype}")
            
            # Display merged detections
            if ContextDataType.DETECTIONS in result10.data and ContextDataType.IMAGE in result10.data:
                image_list = result10.data[ContextDataType.IMAGE]
                image = image_list[0] if isinstance(image_list, list) and image_list else image_list
                dets = result10.data[ContextDataType.DETECTIONS]
                
                if dets:
                    print("\n  Visualizing merged detections...")
                    from object_detector.object_clusterer import ObjectClusterer
                    temp_clusterer = ObjectClusterer(
                        source=None,
                        semantic_provider=semantic_provider,
                        max_clusters=MAX_CLUSTERS,
                        merge_threshold=MERGE_THRESHOLD
                    )
                    temp_viewer = DetectionViewer(
                        source=temp_clusterer,
                        viewer=viewer,
                        display_time_ms=config.display_time_ms
                    )
                    temp_viewer._detect_internal(image, dets)
                    
                    import time
                    start_time = time.time()
                    while time.time() - start_time < 3 and viewer.is_visible():
                        viewer.show(wait_ms=30)
                    
                    temp_viewer.stop()
                    temp_clusterer.stop()
            
            # Cleanup
            viewer.close()
            if hasattr(clusterer_task10, 'detector') and clusterer_task10.detector:
                clusterer_task10.detector.stop()
            
            print_test_footer(success=True, test_num=10)
            
        except ImportError as e:
            print(f"⚠️  Skipping Test 10 (dependencies not available): {e}")
        except Exception as e:
            print(f"❌ Test 10 failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_test_11_semantic_matching(factory, collector, clip_model, semantic_provider,
                                      config: TestConfig, interactive_mode: bool = False):
        """Test 11: Full Semantic Pipeline with PromptEmbedding → YOLO → Cluster → CLIP → Compare"""
        print_test_header(11, "Full Semantic Pipeline: PromptEmbedding → Camera → YOLO → Cluster/Pass → Merge → ClipVision → ClipCompare")
        
        # Load clusterer configuration
        from utils.config import VLMChatConfig
        try:
            vlm_config = VLMChatConfig.load_from_file("config.json")
            clusterer_config = vlm_config.clusterer
        except Exception as e:
            print(f"Warning: Could not load config.json, using defaults: {e}")
            from utils.config import ClustererConfig
            clusterer_config = ClustererConfig()
        
        # Clusterer parameters from config
        MAX_CLUSTERS = clusterer_config.max_clusters
        MERGE_THRESHOLD = 1.1
        PROXIMITY_WEIGHT = 1.5
        SIZE_WEIGHT = 2.0
        SEMANTIC_PAIR_WEIGHT = 1.0
        SEMANTIC_SINGLE_WEIGHT = clusterer_config.semantic_single_weight
        
        # Get test prompts from config
        test_prompts = config.test_prompts if config.test_prompts else [
            "a person on a horse",
            "a person riding a bicycle",
            "a group of people"
        ]
        
        try:
            from object_detector.detection_viewer import DetectionViewer
            from object_detector.image_viewer import ImageViewer
            
            if clip_model is None or semantic_provider is None:
                print("⚠️  CLIPModel or SemanticProvider not initialized, skipping Test 11")
                raise ImportError("CLIPModel initialization failed")
            
            print(f"\nTest 11 Configuration:")
            print(f"  YOLO Model: {config.yolo_model}")
            print(f"  YOLO Confidence: {config.yolo_confidence}")
            print(f"  Max Clusters: {MAX_CLUSTERS}")
            print(f"  Detection Merge Target: {config.detection_merge_target}")
            print(f"  Min Similarity: {config.min_similarity}")
            print(f"  Test Prompts: {test_prompts}")
            
            # Create pipeline tasks
            from .tasks.prompt_embedding_source_task import PromptEmbeddingSourceTask
            
            prompt_task11 = PromptEmbeddingSourceTask(
                prompts=test_prompts,
                clip_model=clip_model,
                task_id="prompt_source11"
            )
            camera_task11 = factory.create_task("camera", "none_camera11", {
                "type": "none",
                "device": "0"
            })
            
            detector_task11 = factory.create_task("yolo_detector", "yolo_det11", {
                "type": "yolo_cpu",
                "model": config.yolo_model,
                "confidence": str(config.yolo_confidence)
            })
            
            clusterer_task11 = factory.create_task("yolo_detector", "clusterer11", {
                "type": "clusterer",
                "source": "yolo_det11",
                "semantic_provider": "semantic_provider",
                "max_clusters": str(MAX_CLUSTERS),
                "merge_threshold": str(MERGE_THRESHOLD),
                "proximity_weight": str(PROXIMITY_WEIGHT),
                "size_weight": str(SIZE_WEIGHT),
                "semantic_pair_weight": str(SEMANTIC_PAIR_WEIGHT),
                "semantic_single_weight": str(SEMANTIC_SINGLE_WEIGHT)
            })
            
            # Create viewer for displaying clustered detections (only if interactive)
            viewer11 = None
            viewer_task11 = None
            if interactive_mode:
                viewer11 = ImageViewer(window_name="Test 11 - Clustered Detections")
            
            pass_task11 = factory.create_task("pass", "pass11")
            
            merge_connector11 = factory.create_connector("detection_merge", "merge11", {
                "target_count": str(config.detection_merge_target)
            })
            merge_connector11.target_count = config.detection_merge_target
            
            clip_task11 = factory.create_task("clip_vision", "clip11", {
                "clip_model": "clip_model"
            })
            clip_task11.clip_model = clip_model
            
            compare_task11 = factory.create_task("clip_compare", "compare11", {
                "min_similarity": str(config.min_similarity),
                "filter_detections": "false"
            })
            
            # Configure clusterer
            clusterer_task11.detector = None
            clusterer_task11.configure({
                "type": "clusterer",
                "source": detector_task11.detector,
                "semantic_provider": semantic_provider,
                "max_clusters": str(MAX_CLUSTERS),
                "merge_threshold": str(MERGE_THRESHOLD),
                "proximity_weight": str(PROXIMITY_WEIGHT),
                "size_weight": str(SIZE_WEIGHT),
                "semantic_pair_weight": str(SEMANTIC_PAIR_WEIGHT),
                "semantic_single_weight": str(SEMANTIC_SINGLE_WEIGHT)
            })
            
            # Create DetectionViewer only if interactive mode
            display_time_ms = 3000 if interactive_mode else 0
            if interactive_mode and viewer11:
                viewer_task11 = DetectionViewer(
                    source=clusterer_task11.detector,
                    viewer=viewer11,
                    display_time_ms=display_time_ms
                )
                viewer_task11.start()
            
            # Enable audit mode for clusterer
            if hasattr(clusterer_task11, 'detector') and clusterer_task11.detector:
                clusterer_task11.detector.start(audit=True)
            
            # Execute pipeline
            print("\n  Executing pipeline...")
            
            ctx11 = Context()
            ctx11 = prompt_task11.run(ctx11)
            print(f"  Step 1 (Prompt Source): PROMPT_EMBEDDINGS in context={ContextDataType.PROMPT_EMBEDDINGS in ctx11.data}")
            
            if ContextDataType.PROMPT_EMBEDDINGS in ctx11.data:
                prompt_data = ctx11.data[ContextDataType.PROMPT_EMBEDDINGS]
                print(f"    Loaded {len(prompt_data.get('prompts', []))} prompts")
            
            ctx11 = camera_task11.run(ctx11)
            print(f"  Step 2 (Camera): IMAGE={ContextDataType.IMAGE in ctx11.data}")
            
            ctx11 = detector_task11.run(ctx11)
            yolo_dets11 = ctx11.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 3 (YOLO): {len(yolo_dets11)} detections")
            
            # Fan-out to Clusterer and Pass
            immutable_cache11 = {}
            split_contexts11 = ctx11.split(2, immutable_cache11)
            
            ctx_cluster11 = clusterer_task11.run(split_contexts11[0])
            cluster_dets11 = ctx_cluster11.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 4a (Clusterer): {len(cluster_dets11)} detections")
            
            # Get and display clusterer audit log
            if hasattr(clusterer_task11, 'detector') and clusterer_task11.detector:
                audit_log = clusterer_task11.detector.get_last_audit_log()
                print("\n" + "="*70)
                print(str(audit_log))
                print("="*70 + "\n")
            
            ctx_pass11 = pass_task11.run(split_contexts11[1])
            pass_dets11 = ctx_pass11.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 4b (Pass): {len(pass_dets11)} detections")
            
            # Merge both branches
            ctx11 = merge_connector11.merge_strategy([ctx_cluster11, ctx_pass11])
            merged_dets11 = ctx11.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 5 (Merge): {len(merged_dets11)} detections")
            
            # Display merged detections before CLIP vision encoding (only if interactive)
            if interactive_mode and viewer_task11 and ContextDataType.IMAGE in ctx11.data:
                image_list = ctx11.data[ContextDataType.IMAGE]
                image = image_list[0] if isinstance(image_list, list) and image_list else image_list
                print(f"  Step 6 (Viewer): Displaying {len(merged_dets11)} merged detections for {display_time_ms}ms...")
                viewer_task11._detect_internal(image, merged_dets11)
            elif not interactive_mode:
                print(f"  Step 6 (Viewer): Skipped (not in interactive mode)")
            
            # CLIP vision
            ctx11 = clip_task11.run(ctx11)
            embeddings11 = ctx11.data.get(ContextDataType.EMBEDDINGS, [])
            print(f"  Step 7 (CLIP Vision): {len(embeddings11)} embeddings")
            
            # CLIP compare (semantic matching)
            ctx11 = compare_task11.run(ctx11)
            matches11 = ctx11.data.get(ContextDataType.MATCHES, [])
            print(f"  Step 8 (Prompt Similarity): {len(matches11)} matches")
            
            result11 = ctx11
            
            print(f"\nPipeline execution completed:")
            print(f"  Has IMAGE: {ContextDataType.IMAGE in result11.data}")
            print(f"  Has DETECTIONS: {ContextDataType.DETECTIONS in result11.data}")
            print(f"  Has EMBEDDINGS: {ContextDataType.EMBEDDINGS in result11.data}")
            print(f"  Has PROMPT_EMBEDDINGS: {ContextDataType.PROMPT_EMBEDDINGS in result11.data}")
            print(f"  Has MATCHES: {ContextDataType.MATCHES in result11.data}")
            
            # Display results: For each detection, show all prompts ranked by similarity
            if ContextDataType.MATCHES in result11.data:
                matches = result11.data[ContextDataType.MATCHES]
                print(f"\n  Prompt Similarity Rankings (threshold >= {config.min_similarity}):")
                print(f"  {'='*70}")
                
                for i, match in enumerate(matches):
                    det_idx = match['detection_index']
                    det = merged_dets11[det_idx] if det_idx < len(merged_dets11) else None
                    det_str = f"{det.object_category} @ {det.conf:.2f}" if det else "unknown"
                    det_id = det.id if det and hasattr(det, 'id') else "N/A"
                    
                    all_scores = match.get('all_scores', [])
                    
                    print(f"\n  Detection #{det_idx} (ID: {det_id}): {det_str}")
                    
                    if all_scores and len(all_scores) == len(test_prompts):
                        ranked_prompts = list(zip(test_prompts, all_scores))
                        ranked_prompts.sort(key=lambda x: x[1], reverse=True)
                        
                        for rank, (prompt, score) in enumerate(ranked_prompts[:3], 1):
                            marker = "★" if rank == 1 else " "
                            print(f"    {marker} {rank}. '{prompt}' → {score:.3f}")
                    else:
                        print(f"    Best match: '{match['prompt_text']}' → {match['similarity']:.3f}")
            
            # Cleanup
            if hasattr(clusterer_task11, 'detector') and clusterer_task11.detector:
                clusterer_task11.detector.stop()
            
            print_test_footer(success=True, test_num=11)
            
        except ImportError as e:
            print(f"⚠️  Skipping Test 11 (dependencies not available): {e}")
        except Exception as e:
            print(f"❌ Test 11 failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_test_12_direct_clip(factory, collector, clip_model, config: TestConfig, 
                                 interactive_mode: bool = False):
        """Test 12: Direct CLIP encoding without YOLO/Clusterer - entire image matching"""
        print_test_header(12, "Direct CLIP: Camera → ClipVision → ClipCompare (no detections)")
        
        # Get test prompts from config
        test_prompts = config.test_prompts if config.test_prompts else [
            "a person on a horse",
            "a person riding a bicycle",
            "a group of people"
        ]
        
        try:
            if clip_model is None:
                print("⚠️  CLIPModel not initialized, skipping Test 12")
                raise ImportError("CLIPModel initialization failed")
            
            print(f"\nTest 12 Configuration:")
            print(f"  Min Similarity: {config.min_similarity}")
            print(f"  Test Prompts: {test_prompts}")
            
            # Create pipeline tasks
            from .tasks.prompt_embedding_source_task import PromptEmbeddingSourceTask
            
            prompt_task12 = PromptEmbeddingSourceTask(
                prompts=test_prompts,
                clip_model=clip_model,
                task_id="prompt_source12"
            )
            
            camera_task12 = factory.create_task("camera", "none_camera12", {
                "type": "none",
                "device": "0"
            })
            
            # Note: No YOLO, no clusterer - ClipVision will encode the entire image
            clip_task12 = factory.create_task("clip_vision", "clip12", {
                "clip_model": "clip_model"
            })
            clip_task12.clip_model = clip_model
            
            compare_task12 = factory.create_task("clip_compare", "compare12", {
                "min_similarity": str(config.min_similarity),
                "filter_detections": "false"
            })
            
            # Execute pipeline
            print("\n  Executing pipeline...")
            
            ctx12 = Context()
            ctx12 = prompt_task12.run(ctx12)
            print(f"  Step 1 (Prompt Source): PROMPT_EMBEDDINGS in context={ContextDataType.PROMPT_EMBEDDINGS in ctx12.data}")
            
            if ContextDataType.PROMPT_EMBEDDINGS in ctx12.data:
                prompt_data = ctx12.data[ContextDataType.PROMPT_EMBEDDINGS]
                print(f"    Loaded {len(prompt_data.get('prompts', []))} prompts")
            
            ctx12 = camera_task12.run(ctx12)
            print(f"  Step 2 (Camera): IMAGE={ContextDataType.IMAGE in ctx12.data}")
            
            ctx12 = clip_task12.run(ctx12)
            embeddings12 = ctx12.data.get(ContextDataType.EMBEDDINGS, [])
            print(f"  Step 3 (CLIP Vision): {len(embeddings12)} embeddings (entire image)")
            
            ctx12 = compare_task12.run(ctx12)
            matches12 = ctx12.data.get(ContextDataType.MATCHES, [])
            print(f"  Step 4 (Prompt Similarity): {len(matches12)} matches")
            
            result12 = ctx12
            
            print(f"\nPipeline execution completed:")
            print(f"  Has IMAGE: {ContextDataType.IMAGE in result12.data}")
            print(f"  Has EMBEDDINGS: {ContextDataType.EMBEDDINGS in result12.data}")
            print(f"  Has PROMPT_EMBEDDINGS: {ContextDataType.PROMPT_EMBEDDINGS in result12.data}")
            print(f"  Has MATCHES: {ContextDataType.MATCHES in result12.data}")
            
            # Display results
            if ContextDataType.MATCHES in result12.data:
                matches = result12.data[ContextDataType.MATCHES]
                print(f"\n  Prompt Similarity Rankings for Entire Image:")
                print(f"  {'='*70}")
                
                if matches:
                    match = matches[0]  # Should only be one match (entire image)
                    all_scores = match.get('all_scores', [])
                    
                    print(f"\n  Entire Image Similarity Scores:")
                    
                    if all_scores and len(all_scores) == len(test_prompts):
                        ranked_prompts = list(zip(test_prompts, all_scores))
                        ranked_prompts.sort(key=lambda x: x[1], reverse=True)
                        
                        for rank, (prompt, score) in enumerate(ranked_prompts, 1):
                            marker = "★" if rank == 1 else " "
                            threshold_note = f" (below threshold)" if score < config.min_similarity else ""
                            print(f"    {marker} {rank}. '{prompt}' → {score:.3f}{threshold_note}")
                    else:
                        print(f"    Best match: '{match['prompt_text']}' → {match['similarity']:.3f}")
                else:
                    print("\n  No embeddings created (check if image was loaded)")
            else:
                print("\n  No MATCHES in context - check pipeline execution")
            
            print_test_footer(success=True, test_num=12)
            
        except ImportError as e:
            print(f"⚠️  Skipping Test 12 (dependencies not available): {e}")
        except Exception as e:
            print(f"❌ Test 12 failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_test_13_detection_expander(factory, collector, clip_model, semantic_provider,
                                        config: TestConfig, interactive_mode: bool = False):
        """Test 13: Pipeline with Detection Expander for improved context"""
        print_test_header(13, "With Detection Expander: Camera → YOLO → Clusterer → Expander → ClipVision → ClipCompare")
        
        # Load clusterer configuration
        from utils.config import VLMChatConfig
        try:
            vlm_config = VLMChatConfig.load_from_file("config.json")
            clusterer_config = vlm_config.clusterer
        except Exception as e:
            print(f"Warning: Could not load config.json, using defaults: {e}")
            from utils.config import ClustererConfig
            clusterer_config = ClustererConfig()
        
        MAX_CLUSTERS = clusterer_config.max_clusters
        MERGE_THRESHOLD = clusterer_config.merge_threshold
        PROXIMITY_WEIGHT = clusterer_config.proximity_weight
        SIZE_WEIGHT = clusterer_config.size_weight
        SEMANTIC_PAIR_WEIGHT = clusterer_config.semantic_pair_weight
        SEMANTIC_SINGLE_WEIGHT = clusterer_config.semantic_single_weight
        
        test_prompts = config.test_prompts if config.test_prompts else [
            "a person on a horse",
            "a white hat",
            "a chair"
        ]
        
        try:
            from object_detector.detection_viewer import DetectionViewer
            from object_detector.image_viewer import ImageViewer
            from .tasks.detection_expander_task import DetectionExpanderTask
            
            if clip_model is None or semantic_provider is None:
                print("⚠️  CLIPModel or SemanticProvider not initialized, skipping Test 13")
                raise ImportError("CLIPModel initialization failed")
            
            display_time_ms = 3000 if interactive_mode else 0
            
            print(f"\nTest 13 Configuration:")
            print(f"  YOLO Model: {config.yolo_model}")
            print(f"  YOLO Confidence: {config.yolo_confidence}")
            print(f"  Max Clusters: {MAX_CLUSTERS}")
            print(f"  Detection Merge Target: {config.detection_merge_target}")
            print(f"  Expansion Factor: {config.expansion_factor:.0%}")
            print(f"  Min Similarity: {config.min_similarity}")
            print(f"  Test Prompts: {test_prompts}")
            
            from .tasks.prompt_embedding_source_task import PromptEmbeddingSourceTask
            
            prompt_task13 = PromptEmbeddingSourceTask(
                prompts=test_prompts,
                clip_model=clip_model,
                task_id="prompt_source13"
            )
            
            camera_task13 = factory.create_task("camera", "none_camera13", {
                "type": "none",
                "device": "0"
            })
            
            detector_task13 = factory.create_task("yolo_detector", "yolo_det13", {
                "type": "yolo_cpu",
                "model": config.yolo_model,
                "confidence": str(config.yolo_confidence)
            })
            
            clusterer_task13 = factory.create_task("yolo_detector", "clusterer13", {
                "type": "clusterer",
                "source": "yolo_det13",
                "semantic_provider": "semantic_provider",
                "max_clusters": str(MAX_CLUSTERS),
                "merge_threshold": str(MERGE_THRESHOLD),
                "proximity_weight": str(PROXIMITY_WEIGHT),
                "size_weight": str(SIZE_WEIGHT),
                "semantic_pair_weight": str(SEMANTIC_PAIR_WEIGHT),
                "semantic_single_weight": str(SEMANTIC_SINGLE_WEIGHT)
            })
            
            expander_task13 = DetectionExpanderTask(
                expansion_factor=config.expansion_factor,
                task_id="expander13"
            )
            
            viewer13 = None
            viewer_task13 = None
            if interactive_mode:
                viewer13 = ImageViewer(window_name="Test 13 - Expanded Detections")
            
            pass_task13 = factory.create_task("pass", "pass13")
            
            merge_connector13 = factory.create_connector("detection_merge", "merge13", {
                "target_count": str(config.detection_merge_target)
            })
            merge_connector13.target_count = config.detection_merge_target
            
            clip_task13 = factory.create_task("clip_vision", "clip13", {
                "clip_model": "clip_model"
            })
            clip_task13.clip_model = clip_model
            
            compare_task13 = factory.create_task("clip_compare", "compare13", {
                "min_similarity": str(config.min_similarity),
                "filter_detections": "false"
            })
            
            clusterer_task13.detector = None
            clusterer_task13.configure({
                "type": "clusterer",
                "source": detector_task13.detector,
                "semantic_provider": semantic_provider,
                "max_clusters": str(MAX_CLUSTERS),
                "merge_threshold": str(MERGE_THRESHOLD),
                "proximity_weight": str(PROXIMITY_WEIGHT),
                "size_weight": str(SIZE_WEIGHT),
                "semantic_pair_weight": str(SEMANTIC_PAIR_WEIGHT),
                "semantic_single_weight": str(SEMANTIC_SINGLE_WEIGHT)
            })
            
            if interactive_mode and viewer13:
                viewer_task13 = DetectionViewer(
                    source=clusterer_task13.detector,
                    viewer=viewer13,
                    display_time_ms=display_time_ms
                )
                viewer_task13.start()
            
            if hasattr(clusterer_task13, 'detector') and clusterer_task13.detector:
                clusterer_task13.detector.start(audit=True)
            
            print("\n  Executing pipeline...")
            
            ctx13 = Context()
            ctx13 = prompt_task13.run(ctx13)
            print(f"  Step 1 (Prompt Source): PROMPT_EMBEDDINGS in context={ContextDataType.PROMPT_EMBEDDINGS in ctx13.data}")
            
            if ContextDataType.PROMPT_EMBEDDINGS in ctx13.data:
                prompt_data = ctx13.data[ContextDataType.PROMPT_EMBEDDINGS]
                print(f"    Loaded {len(prompt_data.get('prompts', []))} prompts")
            
            ctx13 = camera_task13.run(ctx13)
            print(f"  Step 2 (Camera): IMAGE={ContextDataType.IMAGE in ctx13.data}")
            
            ctx13 = detector_task13.run(ctx13)
            yolo_dets13 = ctx13.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 3 (YOLO): {len(yolo_dets13)} detections")
            
            immutable_cache13 = {}
            split_contexts13 = ctx13.split(2, immutable_cache13)
            
            ctx_cluster13 = clusterer_task13.run(split_contexts13[0])
            cluster_dets13 = ctx_cluster13.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 4a (Clusterer): {len(cluster_dets13)} detections")
            
            if hasattr(clusterer_task13, 'detector') and clusterer_task13.detector:
                audit_log = clusterer_task13.detector.get_last_audit_log()
                print("\n" + "="*70)
                print(str(audit_log))
                print("="*70 + "\n")
            
            ctx_pass13 = pass_task13.run(split_contexts13[1])
            pass_dets13 = ctx_pass13.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 4b (Pass): {len(pass_dets13)} detections")
            
            ctx13 = merge_connector13.merge_strategy([ctx_cluster13, ctx_pass13])
            merged_dets13 = ctx13.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 5 (Merge): {len(merged_dets13)} detections")
            
            ctx13 = expander_task13.run(ctx13)
            expanded_dets13 = ctx13.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 6 (Expander): {len(expanded_dets13)} detections expanded by {config.expansion_factor:.0%}")
            
            if interactive_mode and viewer_task13 and ContextDataType.IMAGE in ctx13.data:
                image_list = ctx13.data[ContextDataType.IMAGE]
                image = image_list[0] if isinstance(image_list, list) and image_list else image_list
                print(f"  Step 7 (Viewer): Displaying {len(expanded_dets13)} expanded detections for {display_time_ms}ms...")
                viewer_task13._detect_internal(image, expanded_dets13)
            elif not interactive_mode:
                print(f"  Step 7 (Viewer): Skipped (not in interactive mode)")
            
            ctx13 = clip_task13.run(ctx13)
            embeddings13 = ctx13.data.get(ContextDataType.EMBEDDINGS, [])
            print(f"  Step 8 (CLIP Vision): {len(embeddings13)} embeddings from expanded boxes")
            
            ctx13 = compare_task13.run(ctx13)
            matches13 = ctx13.data.get(ContextDataType.MATCHES, [])
            print(f"  Step 9 (Prompt Similarity): {len(matches13)} matches")
            
            result13 = ctx13
            
            print(f"\nPipeline execution completed:")
            print(f"  Has IMAGE: {ContextDataType.IMAGE in result13.data}")
            print(f"  Has DETECTIONS: {ContextDataType.DETECTIONS in result13.data}")
            print(f"  Has EMBEDDINGS: {ContextDataType.EMBEDDINGS in result13.data}")
            print(f"  Has PROMPT_EMBEDDINGS: {ContextDataType.PROMPT_EMBEDDINGS in result13.data}")
            print(f"  Has MATCHES: {ContextDataType.MATCHES in result13.data}")
            
            if ContextDataType.MATCHES in result13.data:
                matches = result13.data[ContextDataType.MATCHES]
                print(f"\n  Prompt Similarity Rankings with Expanded Boxes (threshold >= {config.min_similarity}):")
                print(f"  {'='*70}")
                
                for i, match in enumerate(matches):
                    det_idx = match['detection_index']
                    det = expanded_dets13[det_idx] if det_idx < len(expanded_dets13) else None
                    det_str = f"{det.object_category} @ {det.conf:.2f}" if det else "unknown"
                    det_id = det.id if det and hasattr(det, 'id') else "N/A"
                    
                    all_scores = match.get('all_scores', [])
                    
                    print(f"\n  Detection #{det_idx} (ID: {det_id}): {det_str}")
                    
                    if all_scores and len(all_scores) == len(test_prompts):
                        ranked_prompts = list(zip(test_prompts, all_scores))
                        ranked_prompts.sort(key=lambda x: x[1], reverse=True)
                        
                        for rank, (prompt, score) in enumerate(ranked_prompts[:3], 1):
                            marker = "★" if rank == 1 else " "
                            print(f"    {marker} {rank}. '{prompt}' → {score:.3f}")
                    else:
                        print(f"    Best match: '{match['prompt_text']}' → {match['similarity']:.3f}")
            
            if hasattr(clusterer_task13, 'detector') and clusterer_task13.detector:
                clusterer_task13.detector.stop()
            
            print_test_footer(success=True, test_num=13)
            
        except ImportError as e:
            print(f"⚠️  Skipping Test 13 (dependencies not available): {e}")
        except Exception as e:
            print(f"❌ Test 13 failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_test_14_boat_image_expander(factory, collector, clip_model, semantic_provider,
                                         config: TestConfig, interactive_mode: bool = False):
        """Test 14: Detection Expander with trail-riders.jpg image"""
        print_test_header(14, "Detection Expander with trail-riders.jpg: Camera → YOLO → Clusterer → Expander → ClipVision → ClipCompare")
        
        # Load clusterer configuration
        from utils.config import VLMChatConfig
        try:
            vlm_config = VLMChatConfig.load_from_file("config.json")
            clusterer_config = vlm_config.clusterer
        except Exception as e:
            print(f"Warning: Could not load config.json, using defaults: {e}")
            from utils.config import ClustererConfig
            clusterer_config = ClustererConfig()
        
        MAX_CLUSTERS = clusterer_config.max_clusters
        MERGE_THRESHOLD = clusterer_config.merge_threshold
        PROXIMITY_WEIGHT = clusterer_config.proximity_weight
        SIZE_WEIGHT = clusterer_config.size_weight
        SEMANTIC_PAIR_WEIGHT = clusterer_config.semantic_pair_weight
        SEMANTIC_SINGLE_WEIGHT = clusterer_config.semantic_single_weight
        
        BOAT_IMAGE_PATH = "src/models/MobileClip/trail-riders.jpg"
        test_prompts = config.test_prompts if config.test_prompts else [
            "a rowboat",
            "a person in a hat",
            "a chair",
            "a person riding a bicycle"
        ]
        
        try:
            from object_detector.detection_viewer import DetectionViewer
            from object_detector.image_viewer import ImageViewer
            from .tasks.detection_expander_task import DetectionExpanderTask
            from PIL import Image
            import cv2
            
            if clip_model is None or semantic_provider is None:
                print("⚠️  CLIPModel or SemanticProvider not initialized, skipping Test 14")
                raise ImportError("CLIPModel initialization failed")
            
            display_time_ms = 3000 if interactive_mode else 0
            
            print(f"\nTest 14 Configuration:")
            print(f"  Image: {BOAT_IMAGE_PATH}")
            print(f"  YOLO Model: {config.yolo_model}")
            print(f"  YOLO Confidence: {config.yolo_confidence}")
            print(f"  Max Clusters: {MAX_CLUSTERS}")
            print(f"  Detection Merge Target: {config.detection_merge_target}")
            print(f"  Expansion Factor: {config.expansion_factor:.0%}")
            print(f"  Min Similarity: {config.min_similarity}")
            print(f"  Test Prompts: {test_prompts}")
            
            from .tasks.prompt_embedding_source_task import PromptEmbeddingSourceTask
            from .task_base import BaseTask
            
            prompt_task14 = PromptEmbeddingSourceTask(
                prompts=test_prompts,
                clip_model=clip_model,
                task_id="prompt_source14"
            )
            
            class CustomImageTask(BaseTask):
                def __init__(self, image_path: str, task_id: str = "custom_image"):
                    super().__init__(task_id)
                    self.image_path = image_path
                    self.output_contract = {ContextDataType.IMAGE: Image.Image}
                
                def configure(self, params):
                    pass
                
                def run(self, context: Context) -> Context:
                    img_bgr = cv2.imread(self.image_path)
                    if img_bgr is None:
                        raise ValueError(f"Failed to load image: {self.image_path}")
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(img_rgb)
                    context.data[ContextDataType.IMAGE] = [pil_image]
                    return context
            
            camera_task14 = CustomImageTask(BOAT_IMAGE_PATH, "boat_camera14")
            
            detector_task14 = factory.create_task("yolo_detector", "yolo_det14", {
                "type": "yolo_cpu",
                "model": config.yolo_model,
                "confidence": str(config.yolo_confidence)
            })
            
            clusterer_task14 = factory.create_task("yolo_detector", "clusterer14", {
                "type": "clusterer",
                "source": "yolo_det14",
                "semantic_provider": "semantic_provider",
                "max_clusters": str(MAX_CLUSTERS),
                "merge_threshold": str(MERGE_THRESHOLD),
                "proximity_weight": str(PROXIMITY_WEIGHT),
                "size_weight": str(SIZE_WEIGHT),
                "semantic_pair_weight": str(SEMANTIC_PAIR_WEIGHT),
                "semantic_single_weight": str(SEMANTIC_SINGLE_WEIGHT)
            })
            
            expander_task14 = DetectionExpanderTask(
                expansion_factor=config.expansion_factor,
                task_id="expander14"
            )
            
            viewer14 = None
            viewer_task14 = None
            if interactive_mode:
                viewer14 = ImageViewer(window_name="Test 14 - Boat Image Expanded Detections")
            
            pass_task14 = factory.create_task("pass", "pass14")
            
            merge_connector14 = factory.create_connector("detection_merge", "merge14", {
                "target_count": str(config.detection_merge_target)
            })
            merge_connector14.target_count = config.detection_merge_target
            
            clip_task14 = factory.create_task("clip_vision", "clip14", {
                "clip_model": "clip_model"
            })
            clip_task14.clip_model = clip_model
            
            compare_task14 = factory.create_task("clip_compare", "compare14", {
                "min_similarity": str(config.min_similarity),
                "filter_detections": "false"
            })
            
            clusterer_task14.detector = None
            clusterer_task14.configure({
                "type": "clusterer",
                "source": detector_task14.detector,
                "semantic_provider": semantic_provider,
                "max_clusters": str(MAX_CLUSTERS),
                "merge_threshold": str(MERGE_THRESHOLD),
                "proximity_weight": str(PROXIMITY_WEIGHT),
                "size_weight": str(SIZE_WEIGHT),
                "semantic_pair_weight": str(SEMANTIC_PAIR_WEIGHT),
                "semantic_single_weight": str(SEMANTIC_SINGLE_WEIGHT)
            })
            
            if interactive_mode and viewer14:
                viewer_task14 = DetectionViewer(
                    source=clusterer_task14.detector,
                    viewer=viewer14,
                    display_time_ms=display_time_ms
                )
                viewer_task14.start()
            
            if hasattr(clusterer_task14, 'detector') and clusterer_task14.detector:
                clusterer_task14.detector.start(audit=True)
            
            print("\n  Executing pipeline...")
            
            ctx14 = Context()
            ctx14 = prompt_task14.run(ctx14)
            print(f"  Step 1 (Prompt Source): PROMPT_EMBEDDINGS in context={ContextDataType.PROMPT_EMBEDDINGS in ctx14.data}")
            
            if ContextDataType.PROMPT_EMBEDDINGS in ctx14.data:
                prompt_data = ctx14.data[ContextDataType.PROMPT_EMBEDDINGS]
                print(f"    Loaded {len(prompt_data.get('prompts', []))} prompts")
            
            ctx14 = camera_task14.run(ctx14)
            print(f"  Step 2 (Load Image): IMAGE={ContextDataType.IMAGE in ctx14.data}")
            
            ctx14 = detector_task14.run(ctx14)
            yolo_dets14 = ctx14.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 3 (YOLO): {len(yolo_dets14)} detections")
            
            immutable_cache14 = {}
            split_contexts14 = ctx14.split(2, immutable_cache14)
            
            ctx_cluster14 = clusterer_task14.run(split_contexts14[0])
            cluster_dets14 = ctx_cluster14.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 4a (Clusterer): {len(cluster_dets14)} detections")
            
            if hasattr(clusterer_task14, 'detector') and clusterer_task14.detector:
                audit_log = clusterer_task14.detector.get_last_audit_log()
                print("\n" + "="*70)
                print(str(audit_log))
                print("="*70 + "\n")
            
            ctx_pass14 = pass_task14.run(split_contexts14[1])
            pass_dets14 = ctx_pass14.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 4b (Pass): {len(pass_dets14)} detections")
            
            ctx14 = merge_connector14.merge_strategy([ctx_cluster14, ctx_pass14])
            merged_dets14 = ctx14.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 5 (Merge): {len(merged_dets14)} detections")
            
            ctx14 = expander_task14.run(ctx14)
            expanded_dets14 = ctx14.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 6 (Expander): {len(expanded_dets14)} detections expanded by {config.expansion_factor:.0%}")
            
            if interactive_mode and viewer_task14 and ContextDataType.IMAGE in ctx14.data:
                image_list = ctx14.data[ContextDataType.IMAGE]
                image = image_list[0] if isinstance(image_list, list) and image_list else image_list
                print(f"  Step 7 (Viewer): Displaying {len(expanded_dets14)} expanded detections for {display_time_ms}ms...")
                viewer_task14._detect_internal(image, expanded_dets14)
            elif not interactive_mode:
                print(f"  Step 7 (Viewer): Skipped (not in interactive mode)")
            
            ctx14 = clip_task14.run(ctx14)
            embeddings14 = ctx14.data.get(ContextDataType.EMBEDDINGS, [])
            print(f"  Step 8 (CLIP Vision): {len(embeddings14)} embeddings from expanded boxes")
            
            ctx14 = compare_task14.run(ctx14)
            matches14 = ctx14.data.get(ContextDataType.MATCHES, [])
            print(f"  Step 9 (Prompt Similarity): {len(matches14)} matches")
            
            result14 = ctx14
            
            print(f"\nPipeline execution completed:")
            print(f"  Has IMAGE: {ContextDataType.IMAGE in result14.data}")
            print(f"  Has DETECTIONS: {ContextDataType.DETECTIONS in result14.data}")
            print(f"  Has EMBEDDINGS: {ContextDataType.EMBEDDINGS in result14.data}")
            print(f"  Has PROMPT_EMBEDDINGS: {ContextDataType.PROMPT_EMBEDDINGS in result14.data}")
            print(f"  Has MATCHES: {ContextDataType.MATCHES in result14.data}")
            
            if ContextDataType.MATCHES in result14.data:
                matches = result14.data[ContextDataType.MATCHES]
                print(f"\n  Prompt Similarity Rankings with Expanded Boxes (threshold >= {config.min_similarity}):")
                print(f"  {'='*70}")
                
                for i, match in enumerate(matches):
                    det_idx = match['detection_index']
                    det = expanded_dets14[det_idx] if det_idx < len(expanded_dets14) else None
                    det_str = f"{det.object_category} @ {det.conf:.2f}" if det else "unknown"
                    det_id = det.id if det and hasattr(det, 'id') else "N/A"
                    
                    all_scores = match.get('all_scores', [])
                    
                    print(f"\n  Detection #{det_idx} (ID: {det_id}): {det_str}")
                    
                    if all_scores and len(all_scores) == len(test_prompts):
                        ranked_prompts = list(zip(test_prompts, all_scores))
                        ranked_prompts.sort(key=lambda x: x[1], reverse=True)
                        
                        for rank, (prompt, score) in enumerate(ranked_prompts, 1):
                            marker = "★" if rank == 1 else " "
                            print(f"    {marker} {rank}. '{prompt}' → {score:.3f}")
                    else:
                        print(f"    Best match: '{match['prompt_text']}' → {match['similarity']:.3f}")
            
            if hasattr(clusterer_task14, 'detector') and clusterer_task14.detector:
                clusterer_task14.detector.stop()
            
            print_test_footer(success=True, test_num=14)
            
        except ImportError as e:
            print(f"⚠️  Skipping Test 14 (dependencies not available): {e}")
        except Exception as e:
            print(f"❌ Test 14 failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_test_15_dual_path_context_attributes(factory, collector, clip_model, semantic_provider,
                                                   config: TestConfig, interactive_mode: bool = False):
        """Test 15: Dual-path pipeline with context filtering and attribute matching"""
        print_test_header(15, "Dual-Path: Context (Filtered+Clustered) + Attributes (Person Crops)")
        
        from utils.config import VLMChatConfig
        try:
            vlm_config = VLMChatConfig.load_from_file("config.json")
            clusterer_config = vlm_config.clusterer
        except Exception as e:
            print(f"Warning: Could not load config.json, using defaults: {e}")
            from utils.config import ClustererConfig
            clusterer_config = ClustererConfig()
        
        MAX_CLUSTERS = clusterer_config.max_clusters
        MERGE_THRESHOLD = 1.1
        PROXIMITY_WEIGHT = config.proximity_weight if config.proximity_weight else 1.7
        SIZE_WEIGHT = config.size_weight if config.size_weight else 2.0
        SEMANTIC_PAIR_WEIGHT = 1.0
        SEMANTIC_SINGLE_WEIGHT = clusterer_config.semantic_single_weight
        
        context_prompts = config.context_prompts if config.context_prompts else ["a person riding a horse"]
        attribute_prompts = config.attribute_prompts if config.attribute_prompts else [
            "a person wearing a white hat",
            "a person wearing a red shirt",
            "a white hat",
            "a chair"
        ]
        ALL_PROMPTS = context_prompts + attribute_prompts
        
        try:
            from object_detector.detection_viewer import DetectionViewer
            from object_detector.image_viewer import ImageViewer
            from .tasks.detection_expander_task import DetectionExpanderTask
            from object_detector.object_clusterer import ObjectClusterer
            from collections import defaultdict
            
            if clip_model is None or semantic_provider is None:
                print("⚠️  CLIPModel or SemanticProvider not initialized, skipping Test 15")
                raise ImportError("CLIPModel initialization failed")
            
            display_time_ms = 3000 if interactive_mode else 0
            
            print(f"\nTest 15 Configuration:")
            print(f"  YOLO Model: {config.yolo_model}")
            print(f"  YOLO Confidence: {config.yolo_confidence}")
            print(f"  Max Clusters: {MAX_CLUSTERS}")
            print(f"  Detection Merge Target: {config.detection_merge_target}")
            print(f"  Expansion Factor: {config.expansion_factor:.0%}")
            print(f"  Context Filter Threshold: {config.context_filter_threshold}")
            print(f"  Min Similarity: {config.min_similarity}")
            print(f"  Context Prompts: {context_prompts}")
            print(f"  Attribute Prompts: {attribute_prompts}")
            
            from .tasks.prompt_embedding_source_task import PromptEmbeddingSourceTask
            
            prompt_task15 = PromptEmbeddingSourceTask(
                prompts=ALL_PROMPTS,
                clip_model=clip_model,
                task_id="prompt_source15"
            )
            
            camera_task15 = factory.create_task("camera", "none_camera15", {
                "type": "none",
                "device": "0"
            })
            
            detector_task15 = factory.create_task("yolo_detector", "yolo_det15", {
                "type": "yolo_cpu",
                "model": config.yolo_model,
                "confidence": str(config.yolo_confidence)
            })
            
            clusterer_task15 = factory.create_task("yolo_detector", "context_clusterer15", {
                "type": "clusterer",
                "source": "yolo_det15",
                "semantic_provider": "semantic_provider",
                "max_clusters": str(MAX_CLUSTERS),
                "merge_threshold": str(MERGE_THRESHOLD),
                "proximity_weight": str(PROXIMITY_WEIGHT),
                "size_weight": str(SIZE_WEIGHT),
                "semantic_pair_weight": str(SEMANTIC_PAIR_WEIGHT),
                "semantic_single_weight": str(SEMANTIC_SINGLE_WEIGHT)
            })
            
            clusterer_task15.detector = ObjectClusterer(
                source=detector_task15.detector,
                semantic_provider=semantic_provider,
                max_clusters=MAX_CLUSTERS,
                merge_threshold=MERGE_THRESHOLD,
                proximity_weight=PROXIMITY_WEIGHT,
                size_weight=SIZE_WEIGHT,
                semantic_weights={"pair": SEMANTIC_PAIR_WEIGHT, "single": SEMANTIC_SINGLE_WEIGHT},
                filter_prompts=context_prompts,
                filter_threshold=config.context_filter_threshold
            )
            
            pass_task15 = factory.create_task("pass", "pass_attr15")
            
            merge_connector15 = factory.create_connector("detection_merge", "merge15", {
                "target_count": str(config.detection_merge_target)
            })
            merge_connector15.target_count = config.detection_merge_target
            
            expander_task15 = DetectionExpanderTask(
                expansion_factor=config.expansion_factor,
                task_id="expander15"
            )
            
            clip_task15 = factory.create_task("clip_vision", "clip15", {
                "clip_model": "clip_model"
            })
            clip_task15.clip_model = clip_model
            
            compare_task15 = factory.create_task("clip_compare", "compare15", {
                "min_similarity": "0.15",
                "filter_detections": "false"
            })
            
            viewer15 = None
            viewer_task15 = None
            if interactive_mode:
                viewer15 = ImageViewer(window_name="Test 15 - Dual-Path Results")
                viewer_task15 = DetectionViewer(
                    source=clusterer_task15.detector,
                    viewer=viewer15,
                    display_time_ms=display_time_ms
                )
                viewer_task15.start()
            
            clusterer_task15.detector.start(audit=True)
            
            print(f"\n  Executing dual-path pipeline...")
            
            ctx15 = Context()
            ctx15 = prompt_task15.run(ctx15)
            print(f"  Step 1 (Prompt Source): PROMPT_EMBEDDINGS in context={ContextDataType.PROMPT_EMBEDDINGS in ctx15.data}")
            
            if ContextDataType.PROMPT_EMBEDDINGS in ctx15.data:
                prompt_data = ctx15.data[ContextDataType.PROMPT_EMBEDDINGS]
                print(f"    Loaded {len(prompt_data.get('prompts', []))} prompts ({len(context_prompts)} context, {len(attribute_prompts)} attribute)")
            
            ctx15 = camera_task15.run(ctx15)
            print(f"  Step 2 (Camera): IMAGE={ContextDataType.IMAGE in ctx15.data}")
            
            ctx15 = detector_task15.run(ctx15)
            yolo_dets15 = ctx15.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 3 (YOLO): {len(yolo_dets15)} detections")
            
            immutable_cache15 = {}
            split_contexts15 = ctx15.split(2, immutable_cache15)
            
            ctx_context15 = clusterer_task15.run(split_contexts15[0])
            context_dets15 = ctx_context15.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 4a (Context Clusterer): {len(context_dets15)} detections (filtered for context)")
            
            if hasattr(clusterer_task15, 'detector') and clusterer_task15.detector:
                audit_log = clusterer_task15.detector.get_last_audit_log()
                print("\n" + "="*70)
                print("--- Context Clustering Audit ---")
                print(str(audit_log))
                print("="*70 + "\n")
            
            ctx_attr15 = pass_task15.run(split_contexts15[1])
            attr_dets15 = ctx_attr15.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 4b (Attribute Pass): {len(attr_dets15)} detections (all original)")
            
            person_dets15 = [det for det in attr_dets15 if det.object_category.lower() == "person"]
            filtered_out_count = len(attr_dets15) - len(person_dets15)
            ctx_attr15.data[ContextDataType.DETECTIONS] = person_dets15
            print(f"  Step 4c (Attribute Filter): {len(person_dets15)} person detections kept (filtered out {filtered_out_count} non-person)")
            
            ctx_context15 = expander_task15.run(ctx_context15)
            expanded_context_dets15 = ctx_context15.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 5a (Context Expander): {len(expanded_context_dets15)} context detections expanded by {config.expansion_factor:.0%}")
            
            ctx_attr15 = expander_task15.run(ctx_attr15)
            expanded_attr_dets15 = ctx_attr15.data.get(ContextDataType.DETECTIONS, [])
            print(f"  Step 5b (Attribute Expander): {len(expanded_attr_dets15)} attribute detections expanded by {config.expansion_factor:.0%}")
            
            if interactive_mode and viewer_task15 and ContextDataType.IMAGE in ctx_context15.data:
                image_list = ctx_context15.data[ContextDataType.IMAGE]
                image = image_list[0] if isinstance(image_list, list) and image_list else image_list
                print(f"  Step 6 (Viewer): Displaying {len(expanded_context_dets15)} expanded context detections for {display_time_ms}ms...")
                viewer_task15._detect_internal(image, expanded_context_dets15)
            elif not interactive_mode:
                print(f"  Step 6 (Viewer): Skipped (not in interactive mode)")
            
            ctx_context15 = clip_task15.run(ctx_context15)
            context_embeddings15 = ctx_context15.data.get(ContextDataType.EMBEDDINGS, [])
            print(f"  Step 7a (CLIP Context): {len(context_embeddings15)} embeddings from expanded context detections")
            
            ctx_attr15 = clip_task15.run(ctx_attr15)
            attr_embeddings15 = ctx_attr15.data.get(ContextDataType.EMBEDDINGS, [])
            print(f"  Step 7b (CLIP Attributes): {len(attr_embeddings15)} embeddings from expanded attribute detections")
            
            ctx_context15 = compare_task15.run(ctx_context15)
            all_context_matches = ctx_context15.data.get(ContextDataType.MATCHES, [])
            context_matches15 = [m for m in all_context_matches if m['prompt_text'] in context_prompts]
            print(f"  Step 8a (Context Matching): {len(context_matches15)} context matches (from {len(all_context_matches)} total)")
            
            ctx_attr15 = compare_task15.run(ctx_attr15)
            all_attr_matches = ctx_attr15.data.get(ContextDataType.MATCHES, [])
            
            print(f"  Step 8b (Attribute Matching): {len(all_attr_matches)} total matches returned")
            if all_attr_matches:
                print(f"    Sample matches: {[(m['prompt_text'], m['similarity']) for m in all_attr_matches[:3]]}")
            
            attr_matches15 = [m for m in all_attr_matches if m['prompt_text'] in attribute_prompts]
            print(f"    After filtering to attribute prompts: {len(attr_matches15)} matches")
            
            all_matches15 = context_matches15 + attr_matches15
            print(f"  Step 9 (Merge): {len(all_matches15)} total matches")
            
            print(f"\nPipeline execution completed:")
            print(f"  Context Path: {len(context_matches15)} matches from {len(context_dets15)} filtered/clustered detections")
            print(f"  Attribute Path: {len(attr_matches15)} matches from {len(attr_dets15)} original detections")
            
            if all_matches15:
                print(f"\n  Dual-Path Rankings (Context: {config.min_similarity}+, Attributes: 0.15+):")
                print("  " + "=" * 70)
                
                matches_by_detection = defaultdict(list)
                
                for match in all_matches15:
                    det_id = match.get('detection_id')
                    prompt_text = match['prompt_text']
                    
                    if prompt_text in context_prompts:
                        match['path_type'] = 'CONTEXT'
                    elif prompt_text in attribute_prompts:
                        match['path_type'] = 'ATTR'
                    else:
                        match['path_type'] = 'UNKNOWN'
                    
                    matches_by_detection[det_id].append(match)
                
                for det_id in sorted(matches_by_detection.keys()):
                    det_matches = sorted(matches_by_detection[det_id], 
                                       key=lambda x: x['similarity'], 
                                       reverse=True)
                    
                    first_match = det_matches[0]
                    det_obj = first_match.get('detection')
                    
                    if det_obj:
                        print(f"\n  Detection #{first_match.get('detection_index', '?')} "
                              f"(ID: {det_id}): {det_obj.object_category} @ "
                              f"{det_obj.conf:.2f}")
                    else:
                        print(f"\n  Detection #{first_match.get('detection_index', '?')} (ID: {det_id}):")
                    
                    for i, match in enumerate(det_matches[:5]):
                        prompt_text = match['prompt_text']
                        similarity = match['similarity']
                        path_type = match.get('path_type', '?')
                        
                        marker = "★" if i == 0 else " "
                        print(f"    {marker} {i+1}. [{path_type}] '{prompt_text}' → {similarity:.3f}")
            else:
                print("\n  No matches found above thresholds")
            
            print("\n" + "=" * 70)
            print("DUAL-PATH SIMILARITY COMPARISON")
            print("=" * 70)
            print(f"\nPath A (Context - Expanded Boxes):")
            print(f"  Detections: {len(context_dets15)} (filtered & clustered from {len(yolo_dets15)} YOLO)")
            print(f"  Expansion: {config.expansion_factor:.0%}")
            print(f"  Prompts: {context_prompts}")
            print(f"  Threshold: {config.min_similarity}")
            
            if all_context_matches:
                print(f"  All Context Scores (showing all prompts for each detection):")
                
                prompt_data_ctx = ctx_context15.data.get(ContextDataType.PROMPT_EMBEDDINGS, {})
                all_prompts_ctx = prompt_data_ctx.get('prompts', [])
                
                for match in sorted(all_context_matches, key=lambda x: x.get('detection_index', -1)):
                    det_idx = match.get('detection_index', '?')
                    det_id = match.get('detection_id', '?')
                    det_obj = match.get('detection')
                    category = det_obj.object_category if det_obj else '?'
                    all_scores = match.get('all_scores', [])
                    
                    print(f"\n    Detection #{det_idx} (ID: {det_id}, {category}):")
                    
                    for prompt_idx, prompt in enumerate(all_prompts_ctx):
                        if prompt in context_prompts and prompt_idx < len(all_scores):
                            score = all_scores[prompt_idx]
                            marker = "✓" if score >= config.min_similarity else "✗"
                            print(f"      {marker} '{prompt}' → {score:.3f}")
            else:
                print(f"  No matches found")
            
            print(f"\nPath B (Attributes - Expanded Crops):")
            print(f"  Detections: {len(person_dets15)} person (from {len(yolo_dets15)} YOLO, filtered out {len(yolo_dets15) - len(person_dets15)} non-person)")
            print(f"  Expansion: {config.expansion_factor:.0%} (padding for better attribute visibility)")
            print(f"  Prompts: {attribute_prompts}")
            print(f"  Threshold: 0.15")
            
            if all_attr_matches:
                print(f"  All Attribute Scores (showing all prompts for each detection):")
                
                prompt_data = ctx_attr15.data.get(ContextDataType.PROMPT_EMBEDDINGS, {})
                all_prompts = prompt_data.get('prompts', [])
                
                for match in sorted(all_attr_matches, key=lambda x: x.get('detection_index', -1)):
                    det_idx = match.get('detection_index', '?')
                    det_id = match.get('detection_id', '?')
                    det_obj = match.get('detection')
                    category = det_obj.object_category if det_obj else '?'
                    all_scores = match.get('all_scores', [])
                    
                    print(f"\n    Detection #{det_idx} (ID: {det_id}, {category}):")
                    
                    for prompt_idx, prompt in enumerate(all_prompts):
                        if prompt in attribute_prompts and prompt_idx < len(all_scores):
                            score = all_scores[prompt_idx]
                            marker = "✓" if score >= 0.15 else "✗"
                            print(f"      {marker} '{prompt}' → {score:.3f}")
            else:
                print(f"  No matches found (bug?)")
            
            print("\n" + "=" * 70)
            
            if hasattr(clusterer_task15, 'detector') and clusterer_task15.detector:
                audit_log = clusterer_task15.detector.get_last_audit_log()
                if audit_log:
                    print("\n" + "=" * 70)
                    print("CONTEXT CLUSTERING AUDIT LOG")
                    print("=" * 70)
                    print(str(audit_log))
                    print("=" * 70)
            
            detector_task15.detector.stop()
            if clusterer_task15.detector:
                clusterer_task15.detector.stop()
            if viewer_task15:
                viewer_task15.stop()
            
            print_test_footer(success=True, test_num=15)
            
        except ImportError as e:
            print(f"⚠️  Skipping Test 15 (dependencies not available): {e}")
        except Exception as e:
            print(f"❌ Test 15 failed: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # END TEST METHOD DEFINITIONS
    # =========================================================================
    
    # Configure logging for tests
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Import with relative imports (now they will work)
    from .task_base import Connector, BaseTask, Context, ContextDataType
    from ..metrics.metrics_collector import Collector, Session
    from ..metrics.instruments import MinMaxAvgLastInstrument, AverageDurationInstrument, CountInstrument
    
    # Create metrics collector and session
    collector = Collector("pipeline_test")
    session = Session(collector)
    
    # Manually create a few instruments for sanity check
    # Track task execution times
    session.add_instrument(
        AverageDurationInstrument("task.execution.duration", binding_keys=["task_id"]),
        "task.execution.duration"
    )
    
    # Track task execution counts
    session.add_instrument(
        CountInstrument("task.execution.count", binding_keys=["task_id", "status"]),
        "task.execution.count"
    )
    
    # Track context data sizes
    session.add_instrument(
        MinMaxAvgLastInstrument("context.data.size", binding_keys=["task_id", "data_type"]),
        "context.data.size"
    )
    
    # Example task implementations
    class ImageSourceTask(BaseTask):
        def __init__(self):
            super().__init__("image_source")
            self.output_contract = {ContextDataType.IMAGE: list}
        
        def run(self, context: Context) -> Context:
            context.data[ContextDataType.IMAGE] = ["image1.jpg", "image2.jpg"]
            print(f"[{self.task_id}] Produced 2 images")
            return context
    
    class DetectorTask(BaseTask):
        def __init__(self, task_id: str):
            super().__init__(task_id)
            self.input_contract = {ContextDataType.IMAGE: list}
            self.output_contract = {ContextDataType.CROPS: list}
        
        def run(self, context: Context) -> Context:
            images = context.data.get(ContextDataType.IMAGE, [])
            crops = [f"{self.task_id}_crop_{i}" for i in range(len(images))]
            if ContextDataType.CROPS not in context.data:
                context.data[ContextDataType.CROPS] = []
            context.data[ContextDataType.CROPS].extend(crops)
            print(f"[{self.task_id}] Produced {len(crops)} crops")
            return context
    
    class EmbeddingTask(BaseTask):
        def __init__(self):
            super().__init__("embedder")
            self.input_contract = {ContextDataType.CROPS: list}
            self.output_contract = {ContextDataType.EMBEDDINGS: list}
        
        def run(self, context: Context) -> Context:
            crops = context.data.get(ContextDataType.CROPS, [])
            embeddings = [f"emb_{crop}" for crop in crops]
            context.data[ContextDataType.EMBEDDINGS] = embeddings
            print(f"[{self.task_id}] Produced {len(embeddings)} embeddings")
            return context
    
    print("=" * 60)
    print("Pipeline Runner - Sanity Test")
    print("=" * 60)
    
    # Initialize factory for all tests
    from .pipeline_factory import PipelineFactory
    from .connectors import FirstCompleteConnector, OrderedMergeConnector
    
    factory = PipelineFactory()
    factory.register_task("image_source", ImageSourceTask)
    factory.register_task("detector", DetectorTask)
    factory.register_task("embedder", EmbeddingTask)
    factory.register_connector("connector", Connector)
    factory.register_connector("first_complete", FirstCompleteConnector)
    factory.register_connector("ordered_merge", OrderedMergeConnector)
    
    # Register detection merge connector
    try:
        from .detection_merge_connector import DetectionMergeConnector
        factory.register_connector("detection_merge", DetectionMergeConnector)
    except ImportError:
        pass
    
    # Register real pipeline tasks
    try:
        from .tasks.camera_task import CameraTask as RealCameraTask
        from .detector_task import DetectorTask as RealDetectorTask
        from .pass_task import PassTask
        from .tasks.clip_vision_task import ClipVisionTask
        from .tasks.clip_compare_task import ClipCompareTask
        from .timeout_task import TimeoutTask
        from .loop_connector import LoopConnector
        from .tasks.prompt_embedding_source_task import PromptEmbeddingSourceTask
        factory.register_task("camera", RealCameraTask)
        factory.register_task("yolo_detector", RealDetectorTask)
        factory.register_task("pass", PassTask)
        factory.register_task("clip_vision", ClipVisionTask)
        factory.register_task("clip_compare", ClipCompareTask)
        factory.register_task("prompt_embedding_source", PromptEmbeddingSourceTask)
        factory.register_task("timeout", TimeoutTask)
        factory.register_connector("loop", LoopConnector)
    except ImportError:
        pass
    
    # Test 1: Linear pipeline
    if should_run_test(1):
        run_test_1_linear_pipeline(factory, collector, TEST_CONFIGS[1], session, interactive_mode)
    
    # Test 2: Branching pipeline with split/merge
    if should_run_test(2):
        run_test_2_branching_pipeline(factory, collector, TEST_CONFIGS[2], session, interactive_mode)
    
    # Test 3: FirstCompleteConnector
    if should_run_test(3):
        run_test_3_first_complete(factory, collector, TEST_CONFIGS[3], session, interactive_mode)
    
    # Test 4: OrderedMergeConnector
    if should_run_test(4):
        run_test_4_ordered_merge(factory, collector, TEST_CONFIGS[4], session, interactive_mode)
    
    # Test 5: PipelineFactory
    if should_run_test(5):
        run_test_5_pipeline_factory(factory, collector, TEST_CONFIGS[5], session, interactive_mode)
    
    # Test 6: Factory with configured connector
    if should_run_test(6):
        run_test_6_factory_config(factory, collector, TEST_CONFIGS[6], session, interactive_mode)
    
    # Test 7: CameraTask with configure()
    if should_run_test(7):
        run_test_7_configure(factory, collector, TEST_CONFIGS[7], session, interactive_mode)
    
    # Test 8: NoneCamera → YOLO → DetectionViewer (Interactive only)
    if should_run_test(8):
        run_test_8_interactive_yolo(factory, collector, TEST_CONFIGS[8], session, interactive_mode)
    
    # =========================================================================
    # Initialize CLIP Model and Semantic Provider for Tests 9-17
    # =========================================================================
    
    clip_model = None
    semantic_provider = None
    SEMANTIC_TEST_PROMPTS = [
        "a person on a horse",
        "a person riding a bicycle",
        "a group of people",
        "a person with a hat",
        "a car on a road",
        "a boat on water"
    ]
    
    # Try to initialize CLIP model and semantic provider
    try:
        from models.MobileClip.clip_model import CLIPModel
        from object_detector.semantic_provider import ClipSemanticProvider
        from utils.config import VLMChatConfig
        from object_detector.coco_categories import CocoCategory
        
        # Load config
        try:
            vlm_config = VLMChatConfig.load_from_file("config.json")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not load config.json: {e}")
            print("Using default CLIP configuration")
            # Create minimal mock config
            class MockConfig:
                class ModelConfig:
                    clip_model_name = "MobileCLIP2-S0"
                    clip_pretrained_path = "src/models/MobileClip/ml-mobileclip/mobileclip2_s0.pt"
                    clip_model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
                model = ModelConfig()
            vlm_config = MockConfig()  # type: ignore
        
        # Initialize CLIP model
        print("\n" + "=" * 60)
        print("Initializing CLIP Model and Semantic Provider...")
        print("-" * 60)
        clip_model = CLIPModel(config=vlm_config, collector=collector)
        print("✓ CLIPModel initialized")
        
        # Initialize semantic provider
        semantic_provider = ClipSemanticProvider(
            clip_model=clip_model,
            user_prompts=SEMANTIC_TEST_PROMPTS,
            embeddings_cache_path="category_pair_embeddings.json",
            batch_size=5
        )
        semantic_provider.start()
        print("✓ SemanticProvider initialized")
        print("=" * 60 + "\n")
        
    except ImportError as e:
        print(f"\n⚠️  Warning: Could not initialize CLIP ({e})")
        print("Tests 10-17 will be skipped")
    except Exception as e:
        print(f"\n⚠️  Warning: CLIP initialization failed: {e}")
        print("Tests 9-17 will be skipped")
    
    # Test 9: NoneCamera → YOLO → Clusterer → DetectionViewer (Interactive only)
    if should_run_test(9):
        run_test_9_interactive_clusterer(factory, collector, clip_model, semantic_provider, TEST_CONFIGS[9], interactive_mode)
    
    # Test 10: Full Pipeline with YOLO -> [Clusterer, Pass] -> DetectionMerge(8) -> ClipVision
    if should_run_test(10):
        run_test_10_full_pipeline(factory, collector, clip_model, semantic_provider,
                                   TEST_CONFIGS[10], interactive_mode)
    
    # Test 11: Full semantic matching pipeline with ClipCompare
    if should_run_test(11):
        run_test_11_semantic_matching(factory, collector, clip_model, semantic_provider,
                                      TEST_CONFIGS[11], interactive_mode)
    
    # Test 12: Direct image to CLIP without YOLO/Clusterer
    if should_run_test(12):
        run_test_12_direct_clip(factory, collector, clip_model, TEST_CONFIGS[12], interactive_mode)
    
    # Test 13: Pipeline with Detection Expander
    if should_run_test(13):
        run_test_13_detection_expander(factory, collector, clip_model, semantic_provider,
                                        TEST_CONFIGS[13], interactive_mode)
    
    # Test 14: Detection Expander with trail-riders.jpg
    if should_run_test(14):
        run_test_14_boat_image_expander(factory, collector, clip_model, semantic_provider,
                                         TEST_CONFIGS[14], interactive_mode)
    
    # Test 15: Dual-path context + attribute matching
    if should_run_test(15):
        run_test_15_dual_path_context_attributes(factory, collector, clip_model, semantic_provider,
                                                  TEST_CONFIGS[15], interactive_mode)
    
    # Test 16: Supersaturated colors for attribute detection
    if should_run_test(16):
        run_test_16_supersaturated_colors(factory, clip_model, semantic_provider, 
                                         TEST_CONFIGS[16], interactive_mode)
    
    # Test 17: Red/Blue channel swap test
    if should_run_test(17):
        run_test_17_color_channel_swap(factory, clip_model, semantic_provider, 
                                       TEST_CONFIGS[17], interactive_mode)
    
    print("\n" + "=" * 60)
    if interactive_mode:
        print("All tests (including interactive) completed successfully!")
    else:
        print("All non-interactive tests completed successfully!")
    print("\nUsage:")
    print("  python -m src.pipeline.pipeline_test                    # Run all non-interactive tests")
    print("  python -m src.pipeline.pipeline_test --interactive      # Include interactive/GUI tests")
    print("  python -m src.pipeline.pipeline_test --test 9           # Run specific test(s)")
    print("  python -m src.pipeline.pipeline_test --test 1,3,5       # Run multiple tests")
    print("  python -m src.pipeline.pipeline_test --test 1-4         # Run test range")
    print("  python -m src.pipeline.pipeline_test --test 9 --interactive  # Run test 9 with GUI")
    print("  python -m src.pipeline.pipeline_test --log-level INFO   # Set log level (DEBUG/INFO/WARNING/ERROR)")
    print("=" * 60)

