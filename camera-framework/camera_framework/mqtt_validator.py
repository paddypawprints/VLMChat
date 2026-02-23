"""MQTT message validator for device incoming commands."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import jsonschema
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)


class MQTTValidator:
    """
    Validates incoming MQTT messages against JSON schemas.
    
    Security model:
    - Whitelist-based: Only known topics are allowed
    - Fail-fast: Device terminates on unknown topic or validation failure
    - Schema enforcement: All messages must conform to their schema
    """
    
    # Topic whitelist with schema mappings
    # Format: topic_pattern -> (schema_name, schema_version)
    TOPIC_SCHEMA_MAP = {
        "devices/{device_id}/commands/metrics": ("metrics-command", "v1.0.0"),
        "devices/{device_id}/commands/snapshot": ("command-simple", "v1.0.0"),
        "devices/{device_id}/commands/logs": ("log-command", "v1.0.0"),
        "devices/{device_id}/commands/filter": ("filter-list", "v1.0.0"),
    }
    
    def __init__(self, device_id: str, schemas_path: Optional[str] = None):
        """
        Initialize MQTT validator.
        
        Args:
            device_id: Device ID for topic matching
            schemas_path: Path to schemas directory 
                         (default: SCHEMAS_PATH env var or ../../shared/schemas)
            
        Raises:
            SystemExit: If schemas cannot be loaded or are invalid
        """
        self.device_id = device_id
        
        # Determine schemas path: CLI arg > Environment variable > Relative path
        if schemas_path is None:
            schemas_path = os.environ.get('SCHEMAS_PATH')
            if schemas_path is None:
                # Default to ../../shared/schemas relative to this file
                # (from camera-framework/camera_framework/ up to VLMChat/ then to shared/)
                current_dir = Path(__file__).parent
                schemas_path = str(current_dir / ".." / ".." / "shared" / "schemas")
        
        self.schemas_path = Path(schemas_path).resolve()
        
        # Load and compile schemas
        self.schemas: Dict[str, Any] = {}
        self._load_schemas()
        
        logger.info(f"✓ MQTT validator initialized with {len(self.schemas)} schemas")
    
    def _load_schemas(self):
        """
        Load all required schemas from disk.
        
        Raises:
            SystemExit: If schemas directory doesn't exist or schemas are invalid
        """
        if not self.schemas_path.exists():
            logger.critical(f"FATAL: Schemas directory not found: {self.schemas_path}")
            logger.critical("Set SCHEMAS_PATH environment variable or use --schemas-path")
            sys.exit(1)
        
        # Extract unique schema requirements from topic map
        required_schemas = set(
            (name, version) for name, version in self.TOPIC_SCHEMA_MAP.values()
        )
        
        for schema_name, version in required_schemas:
            # Use flat schema structure: schema-name-v1.0.0.json
            schema_file = self.schemas_path / f"{schema_name}-{version}.json"
            
            if not schema_file.exists():
                logger.critical(f"FATAL: Required schema not found: {schema_file}")
                sys.exit(1)
            
            try:
                with open(schema_file, 'r') as f:
                    schema = json.load(f)
                
                # Validate schema itself is valid JSON Schema
                # This will raise if the schema is malformed
                jsonschema.Draft7Validator.check_schema(schema)
                
                self.schemas[f"{schema_name}/{version}"] = schema
                logger.debug(f"Loaded schema: {schema_name}/{version}")
                
            except json.JSONDecodeError as e:
                logger.critical(f"FATAL: Invalid JSON in schema {schema_file}: {e}")
                sys.exit(1)
            except jsonschema.SchemaError as e:
                logger.critical(f"FATAL: Invalid JSON Schema {schema_file}: {e}")
                sys.exit(1)
            except Exception as e:
                logger.critical(f"FATAL: Failed to load schema {schema_file}: {e}")
                sys.exit(1)
    
    def validate_message(self, topic: str, payload: Dict[str, Any]):
        """
        Validate incoming MQTT message.
        
        Args:
            topic: MQTT topic the message was received on
            payload: Parsed JSON payload
            
        Raises:
            SystemExit: If topic is unknown or message fails validation
        """
        # ============================================================
        # DO NOT REMOVE OR MODIFY - SECURITY CRITICAL
        # This validation prevents unauthorized commands and ensures
        # data integrity. Device must terminate on validation failure.
        # ============================================================
        
        # Check if topic is in whitelist
        schema_key = self._get_schema_for_topic(topic)
        
        if schema_key is None:
            logger.critical(f"FATAL: Unknown MQTT topic received: {topic}")
            logger.critical("This is a security violation - device terminating")
            sys.exit(1)
        
        # Get schema
        schema = self.schemas.get(schema_key)
        if schema is None:
            logger.critical(f"FATAL: Schema not loaded for topic {topic}: {schema_key}")
            sys.exit(1)
        
        # Validate message against schema
        try:
            validate(instance=payload, schema=schema)
            logger.debug(f"✓ Message validated: {topic}")
            
        except ValidationError as e:
            logger.critical(f"FATAL: Invalid message received on {topic}")
            logger.critical(f"Validation error: {e.message}")
            logger.critical(f"Failed at path: {' -> '.join(str(p) for p in e.path)}")
            logger.critical(f"Payload: {json.dumps(payload, indent=2)}")
            logger.critical("Device terminating due to validation failure")
            sys.exit(1)
        
        except Exception as e:
            logger.critical(f"FATAL: Unexpected validation error for {topic}: {e}")
            sys.exit(1)
        
        # ============================================================
        # END SECURITY CRITICAL SECTION
        # ============================================================
    
    def _get_schema_for_topic(self, topic: str) -> Optional[str]:
        """
        Find schema for topic using whitelist.
        
        Args:
            topic: Actual MQTT topic received
            
        Returns:
            Schema key (name/version) or None if topic not in whitelist
        """
        # Try to match against patterns in whitelist
        for pattern, (schema_name, version) in self.TOPIC_SCHEMA_MAP.items():
            # Replace {device_id} with actual device ID
            expected_topic = pattern.replace("{device_id}", self.device_id)
            
            if topic == expected_topic:
                return f"{schema_name}/{version}"
        
        return None
    
    def get_allowed_topics(self) -> list[str]:
        """
        Get list of allowed topics for this device.
        
        Returns:
            List of MQTT topics device should subscribe to
        """
        return [
            pattern.replace("{device_id}", self.device_id)
            for pattern in self.TOPIC_SCHEMA_MAP.keys()
        ]
