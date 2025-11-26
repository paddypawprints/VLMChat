#!/usr/bin/env python3
"""Quick script to show the similarity report output."""

import sys
sys.path.insert(0, 'src')

from pipeline.dsl_parser import DSLParser
from pipeline.pipeline_runner import PipelineRunner
from pipeline.task_base import ContextDataType

# Parse and run pipeline
parser = DSLParser()
pipeline = parser.parse_file('pipelines/camera_yolo_clusterer_clip.dsl')
runner = PipelineRunner(pipeline.tasks, pipeline.entry_points)
ctx = runner.run()

# Print the similarity report
if ContextDataType.TEXT in ctx.data:
    print("\n" + "="*70)
    print("SIMILARITY REPORT OUTPUT:")
    print("="*70)
    print(ctx.data[ContextDataType.TEXT][0])
else:
    print("No TEXT output found in context")
