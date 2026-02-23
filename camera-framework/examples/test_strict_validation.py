"""Example showing strict buffer validation in action."""

from camera_framework.buffer import Buffer, blocking_policy
from camera_framework.task import BaseTask


class DetectionTracker(BaseTask):
    """Simulates a tracker that publishes alerts."""
    def process(self):
        pass


class VLMWorker(BaseTask):
    """Simulates a VLM worker that also publishes alerts."""
    def process(self):
        pass


if __name__ == "__main__":
    # Create a strict buffer (blocking policy)
    alert_buffer = Buffer(
        size=10, 
        policy=blocking_policy, 
        strict=True,  # STRICT MODE
        name="alert_buffer"
    )
    
    # First task connects - OK
    tracker = DetectionTracker(name="DetectionTracker")
    tracker.add_output("alerts", alert_buffer)
    print("✅ DetectionTracker connected to alert_buffer")
    
    # Second task tries to connect - CRASH!
    vlm_worker = VLMWorker(name="VLMWorker")
    try:
        vlm_worker.add_output("verified_alerts", alert_buffer)
        print("❌ This shouldn't happen - validation failed!")
    except ValueError as e:
        print(f"\n🚨 Buffer validation error (as expected):\n")
        print(f"{e}\n")
        print("✅ Validation working - pipeline construction failed fast!")
