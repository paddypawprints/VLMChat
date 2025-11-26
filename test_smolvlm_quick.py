"""Quick test for SmolVLM task loading."""

import sys
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).parent
PROJECT_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.dsl_parser import DSLParser, create_task_registry
from src.pipeline.pipeline_runner import PipelineRunner

# Load the DSL file
dsl_file = REPO_ROOT / "pipelines" / "smolvlm_chat.dsl"
with open(dsl_file) as f:
    dsl = f.read()

print("DSL:")
print(dsl)
print("\n" + "="*70)

# Create registry and parse
registry = create_task_registry()
parser = DSLParser(registry)

try:
    pipeline = parser.parse(dsl)
    print("✅ DSL parsed successfully")
    
    # Create runner (don't run - would need model)
    runner = PipelineRunner(pipeline, max_workers=2)
    print("✅ Runner created successfully")
    
    # Check queues
    print(f"✅ Input queue: {runner.input_queue}")
    print(f"✅ Output queue: {runner.output_queue}")
    print(f"✅ Running state: {runner.is_running()}")
    
    print("\n✅ All checks passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
