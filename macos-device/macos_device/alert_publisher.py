"""Alert publisher task - publishes confirmed detections to MQTT."""

import base64
import logging
import numpy as np
from datetime import datetime
from typing import Optional, Any
from camera_framework import BaseTask


logger = logging.getLogger(__name__)


def _convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert (dict, list, numpy type, or other)
        
    Returns:
        Object with all numpy types converted to Python types
    """
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class AlertPublisher(BaseTask):
    """
    Publishes confirmed detection alerts to MQTT.
    
    Reads alert events from DetectionTracker and publishes to
    devices/{device_id}/alerts topic.
    
    Alert format follows alerts-v1.0.0.json schema:
    - type: "detection"
    - timestamp: ISO 8601
    - watchlist_item_id: filter_id
    - description: Generated from attributes/colors
    - confidence: Best detection confidence
    - image: Base64 encoded crop
    - metadata: bbox, attributes, colors, etc.
    
    Example:
        publisher = AlertPublisher(
            device_id="mac-dev-01",
            mqtt_client=mqtt_client
        )
        publisher.add_input("alerts", alert_buffer)
    """
    
    def __init__(
        self,
        device_id: str,
        mqtt_client,
        name: str = "alert_publisher",
    ):
        """Initialize alert publisher.
        
        Args:
            device_id: Device identifier for MQTT topic
            mqtt_client: MQTTClient instance for publishing
            name: Task name
        """
        super().__init__(name=name)
        self.device_id = device_id
        self.mqtt_client = mqtt_client
        self.alerts_published = 0
    
    def is_ready(self) -> bool:
        """Ready when input buffer has data."""
        if not self.inputs:
            return False
        input_buffer = list(self.inputs.values())[0]
        return input_buffer.has_data()
    
    def process(self) -> None:
        """Read alerts from buffer and publish to MQTT."""
        if not self.inputs:
            return
        
        input_buffer = list(self.inputs.values())[0]
        message = input_buffer.get()
        if not message:
            return
        
        # Extract confirmation events from message
        confirmations = message.get("confirmations", [])
        for event in confirmations:
            if isinstance(event, dict):
                self._publish_alert(event)
    
    def _publish_alert(self, event: dict) -> None:
        """Publish single alert to MQTT.
        
        Args:
            event: Alert event from VLMQueue or SmolVLMWorker
        """
        try:
            # Build alert payload
            alert = {
                "type": "detection",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "watchlist_item_id": event.get("filter_id", "unknown"),
                "description": self._generate_description(event),
                "confidence": event.get("confidence", 0.0),
            }
            
            # Add VLM-specific fields if present (using schema field names)
            if "vlm_required" in event:
                alert["vlm_required"] = event["vlm_required"]
            if "vlm_status" in event:
                alert["vlm_status"] = event["vlm_status"]
            if "vlm_response" in event:
                alert["vlm_response"] = event["vlm_response"]
            if "vlm_inference_time" in event:
                alert["vlm_inference_time"] = event["vlm_inference_time"]
            
            # Add base64 image if available
            if event.get("crop_jpeg"):
                alert["image"] = base64.b64encode(event["crop_jpeg"]).decode('utf-8')
            
            # Add metadata
            alert["metadata"] = {
                "track_id": event.get("track_id"),
                "bounding_box": self._format_bbox(event.get("bbox")),
                "attributes": _convert_numpy_types(event.get("attributes", {})),
                "colors": _convert_numpy_types(event.get("colors", {})),
                "confirmation_count": event.get("confirmation_count", 0),
                "first_seen": event.get("first_seen"),
            }
            
            # Publish to MQTT
            topic = f"devices/{self.device_id}/alerts"
            self.mqtt_client.publish(topic, alert, qos=1)
            
            self.alerts_published += 1
            logger.info(f"Published alert for filter {event.get('filter_id')} (track {event.get('track_id')})")
            
        except Exception as e:
            logger.error(f"Failed to publish alert: {e}", exc_info=True)
            raise
    
    def _generate_description(self, event: dict) -> str:
        """Generate human-readable alert description.
        
        Args:
            event: Alert event
            
        Returns:
            Description string
        """
        parts = []
        
        # Get attributes
        attributes = event.get("attributes", {})
        colors = event.get("colors", {})
        
        # Build description from attributes
        if attributes:
            # Person attributes
            attr_parts = []
            
            # Gender
            if attributes.get("Female", {}).get("value"):
                attr_parts.append("female")
            
            # Age
            if attributes.get("AgeOver60", {}).get("value"):
                attr_parts.append("elderly")
            elif attributes.get("AgeLess18", {}).get("value"):
                attr_parts.append("young")
            
            # Clothing/accessories
            clothing = []
            if attributes.get("Hat", {}).get("value"):
                hat_color = colors.get("Hat", "")
                clothing.append(f"{hat_color} hat" if hat_color else "hat")
            
            if attributes.get("Glasses", {}).get("value"):
                clothing.append("glasses")
            
            if attributes.get("Backpack", {}).get("value"):
                clothing.append("backpack")
            
            if attributes.get("ShoulderBag", {}).get("value"):
                clothing.append("shoulder bag")
            
            if attributes.get("HandBag", {}).get("value"):
                clothing.append("handbag")
            
            # Upper clothing
            upper = []
            if attributes.get("LongSleeve", {}).get("value"):
                upper_color = colors.get("UpperStride", "") or colors.get("LongSleeve", "")
                upper.append(f"{upper_color} long sleeve" if upper_color else "long sleeve")
            elif attributes.get("ShortSleeve", {}).get("value"):
                upper_color = colors.get("ShortSleeve", "")
                upper.append(f"{upper_color} short sleeve" if upper_color else "short sleeve")
            
            if attributes.get("LongCoat", {}).get("value"):
                coat_color = colors.get("LongCoat", "")
                upper.append(f"{coat_color} coat" if coat_color else "coat")
            
            # Lower clothing
            lower = []
            if attributes.get("Trousers", {}).get("value"):
                lower_color = colors.get("LowerStripe", "") or colors.get("Trousers", "")
                lower.append(f"{lower_color} trousers" if lower_color else "trousers")
            elif attributes.get("Shorts", {}).get("value"):
                lower_color = colors.get("Shorts", "")
                lower.append(f"{lower_color} shorts" if lower_color else "shorts")
            elif attributes.get("Skirt&Dress", {}).get("value"):
                dress_color = colors.get("Skirt&Dress", "")
                lower.append(f"{dress_color} dress/skirt" if dress_color else "dress/skirt")
            
            # Combine parts
            if attr_parts:
                parts.extend(attr_parts)
            if clothing:
                parts.extend(clothing)
            if upper:
                parts.extend(upper)
            if lower:
                parts.extend(lower)
        
        if parts:
            return f"Person detected: {', '.join(parts)}"
        else:
            return "Detection alert"
    
    def _format_bbox(self, bbox: Optional[tuple]) -> Optional[dict]:
        """Convert bbox tuple to schema format.
        
        Args:
            bbox: (x1, y1, x2, y2) tuple
            
        Returns:
            {x, y, width, height} dict
        """
        if not bbox or len(bbox) != 4:
            return None
        
        x1, y1, x2, y2 = bbox
        return {
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
        }
    
    def get_stats(self) -> dict:
        """Get publisher statistics."""
        return {
            "alerts_published": self.alerts_published,
        }
