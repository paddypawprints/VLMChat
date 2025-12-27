"""
Clusterer task for hierarchical clustering of detections.

Performs semantic similarity-based clustering using CLIP embeddings
and spatial constraints.
"""

import logging
from typing import List, Optional, Dict
from pathlib import Path

from ..core.task_base import BaseTask, Context, ContextDataType, register_task
from ..detection import Detection, CocoCategory
from ..cache.image import ImageContainer

logger = logging.getLogger(__name__)


@register_task('clusterer')
class ClustererTask(BaseTask):
    """
    Hierarchical clustering of detections using semantic similarity.
    
    Groups detections based on:
    - Spatial proximity (packing efficiency)
    - Size compatibility
    - Semantic similarity (CLIP embeddings with text prompts)
    
    Contract:
        Input: IMAGE[input_label] - Detection objects to cluster
               TEXT[prompts_label] - Text prompts for semantic matching
        Output: IMAGE[output_label] - Clustered Detection objects with children
    
    Usage:
        # Basic clustering
        yolo() -> clusterer()
        
        # With prompts
        yolo() -> clusterer(prompts=["person riding horse"])
        
        # Configure thresholds
        clusterer(max_clusters=5, merge_threshold=0.7)
    """
    
    def __init__(
        self,
        task_id: str = "clusterer",
        input_label: str = "detections",
        output_label: str = "clustered",
        prompts_label: str = "prompts",
        max_clusters: int = 10,
        max_detections: int = 8,
        merge_threshold: float = 0.6,
        proximity_weight: float = 1.0,
        size_weight: float = 1.0,
        semantic_weight: float = 1.5,
        visual_weight: float = 2.0,
        prob_gain: float = 10.0,
        enable_semantic: bool = True,
        enable_visual: bool = True,
        enable_cluster_validation: bool = False,
        cluster_prompt_threshold: float = 0.25,
        max_validation_candidates: int = 16
    ):
        """
        Initialize clusterer task.
        
        Args:
            task_id: Unique task identifier
            input_label: Label to read detections from
            output_label: Label to write clustered detections to
            prompts_label: Label to read text prompts from
            max_clusters: Maximum number of clusters to output
            max_detections: Maximum detections to process (limited by vision batch size)
            merge_threshold: Similarity threshold for merging (0-1, higher=more selective)
            proximity_weight: Weight for spatial proximity similarity
            size_weight: Weight for size compatibility similarity
            semantic_weight: Weight for detection-prompt similarity (instance-level)
            visual_weight: Weight for CLIP visual similarity (instance-level)
            prob_gain: Sigmoid gain for probability conversion
            enable_semantic: Enable prompt-based semantic similarity
            enable_visual: Enable CLIP visual similarity
            enable_cluster_validation: Validate merged clusters against prompt
            cluster_prompt_threshold: Minimum prompt similarity for valid cluster
            max_validation_candidates: Max candidates to validate per iteration
        """
        super().__init__(task_id)
        
        self.input_label = input_label
        self.output_label = output_label
        self.prompts_label = prompts_label
        self.max_clusters = max_clusters
        self.max_detections = max_detections
        self.merge_threshold = merge_threshold
        self.proximity_weight = proximity_weight
        self.size_weight = size_weight
        self.semantic_weight = semantic_weight
        self.visual_weight = visual_weight
        self.prob_gain = prob_gain
        self.enable_semantic = enable_semantic
        self.enable_visual = enable_visual
        self.enable_cluster_validation = enable_cluster_validation
        self.cluster_prompt_threshold = cluster_prompt_threshold
        self.max_validation_candidates = max_validation_candidates
        
        # Declare contracts
        from ..cache.types import CachedItemType
        self.input_contract = {
            ContextDataType.IMAGE: {
                input_label: (CachedItemType.IMAGE, None)
            },
            ContextDataType.TEXT: {
                prompts_label: (CachedItemType.TEXT, None)
            }
        }
        self.output_contract = {
            ContextDataType.IMAGE: {
                output_label: (CachedItemType.IMAGE, None)
            }
        }
        
        # Internal state
        self._clip_service = None
        self._clip_vision = None
        self._semantic_cache: Optional[Dict] = None
    
    def __del__(self):
        """Cleanup CLIP models and release GPU memory."""
        try:
            if self._clip_vision is not None:
                del self._clip_vision
                self._clip_vision = None
            
            if self._clip_service is not None:
                del self._clip_service
                self._clip_service = None
        except Exception:
            pass  # Ignore cleanup errors
    
    def configure(self, **kwargs) -> None:
        """
        Configure from DSL parameters.
        
        Args:
            **kwargs: Configuration parameters
        """
        if "input" in kwargs:
            self.input_label = kwargs["input"]
            from ..cache.types import CachedItemType
            self.input_contract[ContextDataType.IMAGE] = {
                self.input_label: (CachedItemType.IMAGE, None)
            }
        
        if "output" in kwargs:
            self.output_label = kwargs["output"]
            from ..cache.types import CachedItemType
            self.output_contract[ContextDataType.IMAGE] = {
                self.output_label: (CachedItemType.IMAGE, None)
            }
        
        if "prompts_label" in kwargs:
            self.prompts_label = kwargs["prompts_label"]
        
        if "max_clusters" in kwargs:
            self.max_clusters = int(kwargs["max_clusters"])
        
        if "merge_threshold" in kwargs:
            self.merge_threshold = float(kwargs["merge_threshold"])
        
        if "proximity_weight" in kwargs:
            self.proximity_weight = float(kwargs["proximity_weight"])
        
        if "size_weight" in kwargs:
            self.size_weight = float(kwargs["size_weight"])
        
        if "semantic_weight" in kwargs:
            self.semantic_weight = float(kwargs["semantic_weight"])
        
        if "prob_gain" in kwargs:
            self.prob_gain = float(kwargs["prob_gain"])
    
    def run(self, context: Context) -> Context:
        """
        Cluster detections using hierarchical similarity-based merging.
        
        Args:
            context: Pipeline context with detections and prompts
            
        Returns:
            Context with clustered detections
        """
        # Get detections
        if ContextDataType.IMAGE not in context.data:
            logger.debug(f"No IMAGE data in context")
            return context
        
        if self.input_label not in context.data[ContextDataType.IMAGE]:
            logger.debug(f"No detections in label '{self.input_label}'")
            return context
        
        detections_data = context.data[ContextDataType.IMAGE][self.input_label]
        
        if not detections_data:
            logger.debug(f"Empty detection list in label '{self.input_label}'")
            return context
        
        # Extract Detection objects
        detections: List[Detection] = []
        for item in detections_data:
            if isinstance(item, Detection):
                detections.append(item)
        
        if not detections:
            logger.debug(f"No Detection objects found in '{self.input_label}'")
            return context
        
        # Get text prompts if available
        prompts: List[str] = []
        if ContextDataType.TEXT in context.data:
            if self.prompts_label in context.data[ContextDataType.TEXT]:
                prompts_data = context.data[ContextDataType.TEXT][self.prompts_label]
                prompts = [str(p) for p in prompts_data if p]
        
        # Consume detections (will re-emit after clustering)
        context.clear(data_type=ContextDataType.IMAGE, label=self.input_label)
        
        # Perform clustering
        clustered = self._cluster_detections(detections, prompts)
        
        logger.info(f"[{self.task_id}] Clustered {len(detections)} detections → {len(clustered)} clusters")
        
        # Emit clustered detections
        for detection in clustered:
            context.add_data(ContextDataType.IMAGE, detection, self.output_label)
        
        return context
    
    def _cluster_detections(
        self, 
        detections: List[Detection],
        prompts: List[str]
    ) -> List[Detection]:
        """
        Perform hierarchical clustering on detections.
        
        Args:
            detections: List of Detection objects
            prompts: Text prompts for semantic similarity
            
        Returns:
            List of clustered Detection objects (with children)
        """
        from ...object_detector.object_clusterer import (
            _ClusterNode, _box_area, _union_box, CocoCategory
        )
        
        # Initialize CLIP service if needed and prompts available
        if prompts and not self._clip_service:
            try:
                from ..services.semantic import ClipSemanticService
                from ..models.clip_text import ClipTextTensorRT
                
                # Initialize CLIP text model
                engine_path = "/home/patrick/Dev/model-rt-build/platform/jetson/mobile-clip/text_fp16.engine"
                clip_model = ClipTextTensorRT(engine_path=engine_path)
                self._clip_service = ClipSemanticService(clip_text_model=clip_model)
            except Exception as e:
                logger.warning(f"Could not initialize ClipSemanticService: {e}")
        
        # Initialize CLIP vision encoder if needed
        if not self._clip_vision:
            try:
                from ..models.clip_vision import ClipVisionTensorRT
                
                engine_path = "/home/patrick/Dev/model-rt-build/platform/jetson/mobile-clip/image_fp16.engine"
                self._clip_vision = ClipVisionTensorRT(
                    engine_path=engine_path,
                    model_name="MobileCLIP2-S0",
                    input_size=256,
                    mean=(0.0, 0.0, 0.0),
                    std=(1.0, 1.0, 1.0)
                )
                logger.info(f"[{self.task_id}] CLIP vision encoder initialized")
            except Exception as e:
                logger.warning(f"Could not initialize CLIP vision encoder: {e}")
                self._clip_vision = None
        
        # Limit to top N detections by confidence (for vision batch size)
        if len(detections) > self.max_detections:
            logger.info(f"[{self.task_id}] Limiting {len(detections)} detections to top {self.max_detections} by confidence")
            detections = sorted(detections, key=lambda d: d.confidence, reverse=True)[:self.max_detections]
        
        # Build semantic probabilities cache if we have prompts
        semantic_probs = {}
        if self._clip_service and prompts:
            try:
                # Query all category pairs against prompts
                results = self._clip_service.query_all_pairs(
                    prompts=prompts,
                    as_probabilities=True,
                    prob_gain=self.prob_gain
                )
                
                # Build lookup: category -> max probability across all prompts
                for prompt, pairs in results.items():
                    for cat_a, cat_b, prob in pairs:
                        # Store max probability for each category
                        if cat_a not in semantic_probs or prob > semantic_probs[cat_a]:
                            semantic_probs[cat_a] = prob
                        if cat_b not in semantic_probs or prob > semantic_probs[cat_b]:
                            semantic_probs[cat_b] = prob
                            
            except Exception as e:
                logger.warning(f"Error computing semantic probabilities: {e}")
        
        # Pre-encode visual embeddings and compute detection-prompt similarities
        visual_embeddings = {}
        prompt_embeddings = None
        detection_prompt_similarities = {}  # Store prompt_similarity separately
        category_raw_prompt_sims = {}  # Store category→prompt raw similarities for validation
        
        if self._clip_vision and self.enable_visual:
            try:
                from ..image.formats import ImageFormat
                
                logger.info(f"[{self.task_id}] Pre-encoding {len(detections)} detection crops with CLIP vision")
                
                # Materialize all detection crops as numpy arrays (BGR format from OpenCV)
                crops = []
                for det in detections:
                    crop_img = det.materialize(format=ImageFormat.NUMPY)
                    crops.append(crop_img)
                
                # Encode all crops in a single batch call
                embeddings = self._clip_vision.encode(crops, use_cache=True)
                
                # Store embeddings indexed by detection id
                for det, emb in zip(detections, embeddings):
                    visual_embeddings[id(det)] = emb
                
                logger.info(f"[{self.task_id}] Encoded {len(visual_embeddings)} visual embeddings")
                
                # Compute prompt embeddings for semantic similarity and validation
                if prompts and self._clip_service:
                    try:
                        import torch
                        import numpy as np
                        # Get text embeddings from CLIP model
                        prompt_embeddings = self._clip_service.clip_model.encode(prompts, use_cache=True)
                        
                        # Compute detection-prompt similarities for semantic matching
                        if self.enable_semantic:
                            # Collect all raw similarities first
                            raw_similarities = []
                            detection_raw_prompt_sims = {}  # Store raw for validation
                            
                            # Get raw category embeddings for comparison
                            category_raw_sims = {}
                            category_embeddings = {}
                            if self._clip_service:
                                for det in detections:
                                    cat_label = det.category.label if hasattr(det, 'category') else str(det.object_category)
                                    if cat_label not in category_raw_sims:
                                        # Encode category text and compute raw similarity with prompts
                                        cat_emb = self._clip_service.clip_model.encode([cat_label], use_cache=True)[0]
                                        category_embeddings[cat_label] = cat_emb
                                        cat_sims = []
                                        for prompt_emb in prompt_embeddings:
                                            sim = (cat_emb * prompt_emb).sum().item()
                                            cat_sims.append(sim)
                                        category_raw_sims[cat_label] = max(cat_sims) if cat_sims else 0.0
                            
                            # Store for validation use
                            category_raw_prompt_sims = category_raw_sims.copy()
                            
                            logger.info(f"[{self.task_id}] Comparing RAW similarity calculations:")
                            logger.info(f"  {'Category':<12} {'Cat→Pr':<10} {'Crop→Pr':<10} {'Cat→Crop':<10} {'Cat-Pr Δ':<10} {'Cat-Crop Δ':<10}")
                            
                            for det in detections:
                                det_emb = visual_embeddings.get(id(det))
                                if det_emb is not None:
                                    # Compute cosine similarity with each prompt
                                    similarities = []
                                    for prompt_emb in prompt_embeddings:
                                        sim = (det_emb * prompt_emb).sum().item()
                                        similarities.append(sim)
                                    max_sim = max(similarities) if similarities else 0.5
                                    raw_similarities.append(max_sim)
                                    detection_raw_prompt_sims[id(det)] = max_sim
                                    
                                    # Compare: category→prompt, crop→prompt, category→crop
                                    cat_label = det.category.label if hasattr(det, 'category') else str(det.object_category)
                                    cat_raw_prompt = category_raw_sims.get(cat_label, 0.0)
                                    
                                    # Category→Crop similarity
                                    cat_emb = category_embeddings.get(cat_label)
                                    cat_crop_sim = (cat_emb * det_emb).sum().item() if cat_emb is not None else 0.0
                                    
                                    diff_prompt = cat_raw_prompt - max_sim
                                    diff_crop = cat_raw_prompt - cat_crop_sim
                                    logger.info(f"  {cat_label:<12} {cat_raw_prompt:<10.3f} {max_sim:<10.3f} {cat_crop_sim:<10.3f} {diff_prompt:+10.3f} {diff_crop:+10.3f}")
                                else:
                                    raw_similarities.append(0.5)
                                    detection_raw_prompt_sims[id(det)] = 0.5
                            
                            # Apply sigmoid transformation (same as semantic service)
                            # prob = 1 / (1 + exp(-gain * (sim - center)))
                            if raw_similarities:
                                raw_sims_array = np.array(raw_similarities)
                                prob_center = float(np.median(raw_sims_array))
                                probs = 1.0 / (1.0 + np.exp(-self.prob_gain * (raw_sims_array - prob_center)))
                                
                                # Store transformed probabilities (for candidate sorting)
                                for det, crop_sigmoid in zip(detections, probs):
                                    detection_prompt_similarities[id(det)] = float(crop_sigmoid)
                                
                                logger.info(f"[{self.task_id}] Computed detection-prompt similarities (median={prob_center:.3f})")
                    except Exception as e:
                        logger.warning(f"Error computing prompt embeddings: {e}")
                        prompt_embeddings = None
                
            except Exception as e:
                logger.warning(f"Error encoding visual embeddings: {e}")
                import traceback
                traceback.print_exc()
        
        # Create initial cluster nodes
        nodes: List[_ClusterNode] = []
        for det in detections:
            cat_enum = det.category
            bbox = det.bbox
            area = _box_area(bbox)
            
            node = _ClusterNode(
                detection=det,
                pixel_area=area,
                categories={cat_enum},
                category_areas={cat_enum: [area]},
                creation_cost=0.0
            )
            nodes.append(node)
        
        # Hierarchical clustering - merge anything above threshold
        iteration = 0
        logger.info(f"[{self.task_id}] Starting clustering: {len(nodes)} nodes, threshold={self.merge_threshold}")
        if semantic_probs:
            logger.info(f"[{self.task_id}] Semantic probabilities: {semantic_probs}")
        
        # Keep merging as long as we can find pairs above threshold
        while len(nodes) > 1:
            iteration += 1
            
            # Collect merge candidates based on SPATIAL proximity and size only
            # This identifies local clusters - objects that are actually near each other
            candidates = []
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    a, b = nodes[i], nodes[j]
                    
                    # Calculate ONLY proximity and size for candidate identification
                    from ...object_detector.object_clusterer import _union_box, _box_area
                    
                    a_box = a.detection.bbox if hasattr(a.detection, 'bbox') else a.detection.box
                    b_box = b.detection.bbox if hasattr(b.detection, 'bbox') else b.detection.box
                    
                    # Proximity (packing efficiency)
                    merged_box = _union_box(a_box, b_box)
                    merged_box_area = _box_area(merged_box)
                    total_pixel_area = a.pixel_area + b.pixel_area
                    sim_prox = min(1.0, total_pixel_area / merged_box_area) if merged_box_area > 0 else 0.0
                    
                    # Size similarity
                    sim_size = 1.0
                    shared_categories = a.categories.intersection(b.categories)
                    if shared_categories:
                        ratios = []
                        for cat in shared_categories:
                            areas_a = a.category_areas.get(cat, [])
                            areas_b = b.category_areas.get(cat, [])
                            if areas_a and areas_b:
                                avg_a = sum(areas_a) / len(areas_a)
                                avg_b = sum(areas_b) / len(areas_b)
                                if avg_a > 0 and avg_b > 0:
                                    ratio = min(avg_a, avg_b) / max(avg_a, avg_b)
                                    ratios.append(ratio)
                        if ratios:
                            sim_size = sum(ratios) / len(ratios)
                    else:
                        min_area = min(a.pixel_area, b.pixel_area)
                        max_area = max(a.pixel_area, b.pixel_area)
                        sim_size = min_area / max_area if max_area > 0 else 0.0
                    
                    # Spatial score: weighted combination of proximity and size
                    spatial_score = (
                        self.proximity_weight * sim_prox +
                        self.size_weight * sim_size
                    ) / (self.proximity_weight + self.size_weight)
                    
                    # Semantic score: use category-based probabilities for candidate selection
                    # (gives better signal than instance-level prompt matching)
                    import math
                    if semantic_probs:
                        a_label = a.detection.category.label if hasattr(a.detection, 'category') else str(a.detection.object_category)
                        b_label = b.detection.category.label if hasattr(b.detection, 'category') else str(b.detection.object_category)
                        a_prob = semantic_probs.get(a_label, 0.5)
                        b_prob = semantic_probs.get(b_label, 0.5)
                        semantic_score = math.sqrt(a_prob * b_prob)
                    else:
                        # Fall back to instance-level prompt similarities if no category probs
                        a_prompt_sim = detection_prompt_similarities.get(id(a.detection), 0.5)
                        b_prompt_sim = detection_prompt_similarities.get(id(b.detection), 0.5)
                        semantic_score = math.sqrt(a_prompt_sim * b_prompt_sim)
                    
                    # Combined score: balance spatial locality with semantic relevance
                    combined_score = (spatial_score + semantic_score) / 2.0
                    
                    candidates.append({
                        'indices': (i, j),
                        'nodes': (a, b),
                        'spatial_score': spatial_score,
                        'semantic_score': semantic_score,
                        'combined_score': combined_score,
                        'sim_prox': sim_prox,
                        'sim_size': sim_size
                    })
            
            # No candidates - stop clustering
            if not candidates:
                logger.info(f"[{self.task_id}] Stopping: no more pairs to evaluate")
                break
            
            # Log all candidate scores before sorting
            logger.info(f"[{self.task_id}] Iteration {iteration}: All {len(candidates)} merge candidates:")
            for idx, cand in enumerate(candidates):
                a, b = cand['nodes']
                a_label = a.detection.category.label
                b_label = b.detection.category.label
                logger.info(
                    f"  [{idx}] ({a_label}, {b_label}): "
                    f"combined={cand['combined_score']:.3f} "
                    f"spatial={cand['spatial_score']:.3f} (prox={cand['sim_prox']:.3f}, size={cand['sim_size']:.3f}) "
                    f"semantic={cand['semantic_score']:.3f}"
                )
            
            # Sort by COMBINED score (spatial + semantic) - prioritizes relevant local clusters
            candidates.sort(key=lambda c: c['combined_score'], reverse=True)
            
            # Pure spatial clustering: only merge spatially-tight candidates
            # Let CLIP handle semantic evaluation after clustering
            logger.info(f"[{self.task_id}] Iteration {iteration}: Processing {len(candidates)} spatial candidates")
            
            # Limit to top N candidates to avoid quadratic explosion
            merge_candidates = candidates[:self.max_validation_candidates]
            
            # Accept candidates with high spatial scores (tight packing)
            # This prevents merging distant clusters together
            spatial_threshold = 0.75  # Require 75% packing efficiency
            valid_candidates = []
            for cand in merge_candidates:
                # Check if spatial relationship is tight enough
                if cand['spatial_score'] < spatial_threshold:
                    logger.debug(f"  REJECT: spatial={cand['spatial_score']:.3f} < threshold={spatial_threshold}")
                    continue
                
                # Compute full similarity for this candidate
                a, b = cand['nodes']
                full_sim, components = self._calculate_similarity(
                    a, b, semantic_probs, visual_embeddings, detection_prompt_similarities
                )
                
                cand['valid'] = True
                cand['similarity'] = full_sim
                cand['components'] = components
                cand['improvement'] = cand['spatial_score']  # Use spatial score for sorting
                valid_candidates.append(cand)
            
            if not valid_candidates:
                logger.info(f"[{self.task_id}] Stopping: no candidates meet spatial threshold")
                break
            
            # Sort by combined score (highest first) for deterministic ordering
            valid_candidates.sort(key=lambda c: c['improvement'], reverse=True)
            
            logger.info(f"[{self.task_id}] Iteration {iteration}: Merging {len(valid_candidates)} spatial pairs")
            
            # Track which nodes have been merged to avoid conflicts
            merged_nodes = set()
            merged_count = 0
            
            for cand in valid_candidates:
                i, j = cand['indices']
                # Skip if either node was already merged
                if i in merged_nodes or j in merged_nodes:
                    continue
                
                a, b = cand['nodes']
                merged = self._merge_nodes(a, b, cand['similarity'])
                
                # Mark nodes as merged
                merged_nodes.add(i)
                merged_nodes.add(j)
                
                # Remove old nodes and add merged
                nodes = [n for idx, n in enumerate(nodes) if idx not in (i, j)]
                nodes.append(merged)
                
                merged_count += 1
                
                a_label = a.detection.category.label
                b_label = b.detection.category.label
                a_cid = getattr(a.detection, 'cluster_id', '?')
                b_cid = getattr(b.detection, 'cluster_id', '?')
                logger.info(
                    f"[{self.task_id}]   Merged C{a_cid}:{a_label} + C{b_cid}:{b_label} "
                    f"sim={cand['similarity']:.3f} spatial={cand['spatial_score']:.3f}"
                )
            
            logger.info(f"[{self.task_id}] Iteration {iteration}: Completed {merged_count} merges")
        
        # If we have too many clusters, keep only the best ones
        if len(nodes) > self.max_clusters:
            logger.info(f"[{self.task_id}] Limiting {len(nodes)} clusters to top {self.max_clusters} by score")
            # Sort by creation_cost (similarity score) descending
            nodes.sort(key=lambda n: n.creation_cost, reverse=True)
            nodes = nodes[:self.max_clusters]
        
        # Match clusters to prompts and assign IDs
        if prompt_embeddings is not None and self._clip_vision:
            from ..image.formats import ImageFormat
            import numpy as np
            
            logger.info(f"[{self.task_id}] Computing cluster-prompt similarities for {len(nodes)} clusters")
            
            # Encode each cluster's crop
            cluster_crops = []
            for node in nodes:
                crop = node.detection.materialize(format=ImageFormat.NUMPY)
                cluster_crops.append(crop)
            
            cluster_embeddings = self._clip_vision.encode(cluster_crops, use_cache=False)
            
            # Compute all cluster-prompt similarities (raw cosine)
            cluster_prompt_sims = []  # List of (cluster_idx, prompt_idx, raw_sim)
            for cluster_idx, cluster_emb in enumerate(cluster_embeddings):
                for prompt_idx, prompt_emb in enumerate(prompt_embeddings):
                    sim = (cluster_emb * prompt_emb).sum().item()
                    cluster_prompt_sims.append((cluster_idx, prompt_idx, sim))
            
            # Find best match for each cluster
            cluster_matches = {}  # cluster_idx -> (prompt_idx, raw_sim)
            for cluster_idx in range(len(nodes)):
                best_prompt_idx = 0
                best_sim = -1.0
                for cidx, pidx, sim in cluster_prompt_sims:
                    if cidx == cluster_idx and sim > best_sim:
                        best_sim = sim
                        best_prompt_idx = pidx
                cluster_matches[cluster_idx] = (best_prompt_idx, best_sim)
            
            # Log cluster similarities
            logger.info(f"[{self.task_id}] Cluster-Prompt Raw Similarities:")
            logger.info(f"  {'Cluster':<10} {'Category':<15} {'Prompt':<30} {'Raw Sim':<10}")
            for cluster_idx, node in enumerate(nodes):
                prompt_idx, raw_sim = cluster_matches[cluster_idx]
                category = node.detection.category.label
                prompt_text = prompts[prompt_idx][:28] + "..." if len(prompts[prompt_idx]) > 28 else prompts[prompt_idx]
                logger.info(f"  C{cluster_idx:<9} {category:<15} {prompt_text:<30} {raw_sim:<10.3f}")
            
            # Compute median of raw similarities for logging
            all_sims = [sim for _, _, sim in cluster_prompt_sims]
            median_sim = float(np.median(all_sims)) if all_sims else 0.25
            logger.info(f"[{self.task_id}] Cluster similarity median: {median_sim:.3f}")
            
            # Assign cluster IDs and matched prompts with calibrated probabilities
            # Based on CLIP similarity distributions:
            #   - Random pairs peak at ~0.15-0.20 (should map to low probability)
            #   - Corresponding pairs peak at ~0.25-0.30 (should map to high probability)
            # Use sigmoid centered at 0.25 (good match threshold) with appropriate gain
            calibration_center = 0.25  # Empirically determined from CLIP distributions
            calibration_gain = 15.0    # Steeper curve for better separation
            
            for cluster_id, node in enumerate(nodes):
                prompt_idx, raw_sim = cluster_matches[cluster_id]
                
                # Apply calibrated sigmoid transformation
                # raw_sim < 0.15: prob → 0-10% (random)
                # raw_sim ~ 0.25: prob → 50% (decision boundary)
                # raw_sim > 0.30: prob → 80-95% (strong match)
                prob = 1.0 / (1.0 + np.exp(-calibration_gain * (raw_sim - calibration_center)))
                
                # Store cluster ID and matched prompt
                node.detection.cluster_id = cluster_id
                node.detection.matched_prompts = [prompts[prompt_idx]]
                node.detection.match_probabilities = [float(prob)]
                
                num_children = len(node.detection.children) if node.detection.children else 0
                logger.info(
                    f"[{self.task_id}] Cluster {cluster_id}: '{prompts[prompt_idx]}' "
                    f"raw={raw_sim:.3f} prob={prob:.3f} children={num_children}"
                )
        else:
            # No prompts - just assign IDs
            for cluster_id, node in enumerate(nodes):
                node.detection.cluster_id = cluster_id
                logger.info(f"[{self.task_id}] Cluster {cluster_id}: {node.detection.category.label} with {len(node.detection.children) if node.detection.children else 0} children")
        
        # Return Detection objects
        return [n.detection for n in nodes]
    
    def _calculate_similarity(
        self,
        a: '_ClusterNode',
        b: '_ClusterNode',
        semantic_probs: Dict,
        visual_embeddings: Dict,
        detection_prompt_similarities: Dict
    ) -> tuple[float, Dict[str, float]]:
        """
        Calculate weighted average similarity for merging two nodes.
        
        Args:
            a: First cluster node
            b: Second cluster node
            semantic_probs: Precomputed semantic probabilities (category-level fallback)
            visual_embeddings: Precomputed CLIP vision embeddings (instance-level)
            detection_prompt_similarities: Detection-prompt cosine similarities
            
        Returns:
            Tuple of (similarity score, component dict) where components contains
            {'prox': proximity_sim, 'size': size_sim, 'sem': semantic_sim, 'vis': visual_sim}
        """
        import math
        from ...object_detector.object_clusterer import _box_area, _union_box
        
        # Extract boxes
        a_box = a.detection.bbox if hasattr(a.detection, 'bbox') else a.detection.box
        b_box = b.detection.bbox if hasattr(b.detection, 'bbox') else b.detection.box
        
        # 1. Proximity similarity (packing efficiency)
        merged_box = _union_box(a_box, b_box)
        merged_box_area = _box_area(merged_box)
        total_pixel_area = a.pixel_area + b.pixel_area
        
        # Clamp to [0, 1] - can exceed 1.0 if boxes overlap
        sim_prox = min(1.0, total_pixel_area / merged_box_area) if merged_box_area > 0 else 0.0
        
        # 2. Size similarity
        sim_size = 1.0
        shared_categories = a.categories.intersection(b.categories)
        if shared_categories:
            ratios = []
            for cat in shared_categories:
                areas_a = a.category_areas.get(cat, [])
                areas_b = b.category_areas.get(cat, [])
                if areas_a and areas_b:
                    avg_a = sum(areas_a) / len(areas_a)
                    avg_b = sum(areas_b) / len(areas_b)
                    if avg_a > 0 and avg_b > 0:
                        ratio = min(avg_a, avg_b) / max(avg_a, avg_b)
                        ratios.append(ratio)
            if ratios:
                sim_size = sum(ratios) / len(ratios)
        else:
            # Different categories - less compatible
            min_area = min(a.pixel_area, b.pixel_area)
            max_area = max(a.pixel_area, b.pixel_area)
            sim_size = min_area / max_area if max_area > 0 else 0.0
        
        # 3. Semantic similarity (prompt-based or category-based fallback)
        sim_sem = 0.5  # Default neutral
        
        if self.enable_semantic:
            # Use detection-prompt similarities if available
            a_prompt_sim = detection_prompt_similarities.get(id(a.detection))
            b_prompt_sim = detection_prompt_similarities.get(id(b.detection))
            
            if a_prompt_sim is not None and b_prompt_sim is not None:
                # Geometric mean of detection-prompt similarities
                import math
                sim_sem = math.sqrt(a_prompt_sim * b_prompt_sim)
            elif semantic_probs:
                # Fall back to category-based if no prompt similarities
                a_label = a.detection.category.label if hasattr(a.detection, 'category') else str(a.detection.object_category)
                b_label = b.detection.category.label if hasattr(b.detection, 'category') else str(b.detection.object_category)
                
                a_prob = semantic_probs.get(a_label, 0.5)
                b_prob = semantic_probs.get(b_label, 0.5)
                
                import math
                sim_sem = math.sqrt(a_prob * b_prob)
        
        # 4. Visual similarity (instance-level CLIP vision)
        sim_vis = 0.5  # Default neutral
        if self.enable_visual and visual_embeddings:
            a_id = id(a.detection)
            b_id = id(b.detection)
            
            if a_id in visual_embeddings and b_id in visual_embeddings:
                emb_a = visual_embeddings[a_id]
                emb_b = visual_embeddings[b_id]
                
                # Cosine similarity (embeddings are already normalized)
                sim_vis = (emb_a * emb_b).sum().item()
        
        # Weighted average
        total_weight = self.proximity_weight + self.size_weight + self.semantic_weight + self.visual_weight
        if total_weight == 0:
            return 0.0, {'prox': sim_prox, 'size': sim_size, 'sem': sim_sem, 'vis': sim_vis}
        
        weighted_sum = (
            sim_prox * self.proximity_weight +
            sim_size * self.size_weight +
            sim_sem * self.semantic_weight +
            sim_vis * self.visual_weight
        )
        
        total_sim = weighted_sum / total_weight
        
        return total_sim, {'prox': sim_prox, 'size': sim_size, 'sem': sim_sem, 'vis': sim_vis}
    
    def _merge_nodes(
        self,
        a: '_ClusterNode',
        b: '_ClusterNode',
        similarity: float
    ) -> '_ClusterNode':
        """
        Merge two cluster nodes into a new one.
        
        Args:
            a: First cluster node
            b: Second cluster node
            similarity: Merge similarity score
            
        Returns:
            New merged cluster node
        """
        from ...object_detector.object_clusterer import (
            _ClusterNode, _box_area, _union_box
        )
        
        # Extract boxes and attributes
        a_box = a.detection.bbox if hasattr(a.detection, 'bbox') else a.detection.box
        b_box = b.detection.bbox if hasattr(b.detection, 'bbox') else b.detection.box
        
        a_conf = a.detection.confidence if hasattr(a.detection, 'confidence') else a.detection.conf
        b_conf = b.detection.confidence if hasattr(b.detection, 'confidence') else b.detection.conf
        
        # Merge spatial and category info
        new_box = _union_box(a_box, b_box)
        new_pixel_area = a.pixel_area + b.pixel_area
        new_categories = a.categories.union(b.categories)
        
        new_category_areas = a.category_areas.copy()
        for cat, areas in b.category_areas.items():
            new_category_areas.setdefault(cat, []).extend(areas)
        
        # Choose category from larger detection
        new_category = a.detection.category if a.pixel_area > b.pixel_area else b.detection.category
        new_conf = max(a_conf, b_conf)
        
        # Get base image
        base_image = a.detection.get_base_image()
        
        # Create new Detection with children
        merged_detection = Detection(
            bbox=new_box,
            confidence=new_conf,
            category=new_category,
            source_image=base_image,
            cache_key=f"cluster_{id(a)}_{id(b)}"
        )
        merged_detection.children = [a.detection, b.detection]
        
        return _ClusterNode(
            detection=merged_detection,
            pixel_area=new_pixel_area,
            categories=new_categories,
            category_areas=new_category_areas,
            creation_cost=similarity
        )
    
    def __str__(self) -> str:
        """String representation."""
        return (f"ClustererTask(in={self.input_label}, out={self.output_label}, "
                f"max_clusters={self.max_clusters}, threshold={self.merge_threshold})")
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()
