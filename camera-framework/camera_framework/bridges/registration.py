"""Device registration utilities."""

import time
import platform
from pathlib import Path
from typing import Optional
from datetime import datetime
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

try:
    import psutil
except ImportError:
    psutil = None


def ensure_keypair(device_id: str) -> tuple:
    """Generate or load Ed25519 keypair for device.
    
    Returns:
        (private_key, public_key) tuple
    """
    key_path = Path.home() / f".vlmchat_device_key_{device_id}.pem"
    
    if key_path.exists():
        # Load existing key
        with open(key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
            )
    else:
        # Generate new keypair
        private_key = ed25519.Ed25519PrivateKey.generate()
        
        # Save to file
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        with open(key_path, "wb") as f:
            f.write(pem)
        key_path.chmod(0o600)
    
    public_key = private_key.public_key()
    return private_key, public_key


def get_device_specs() -> dict:
    """Collect device specifications."""
    specs = {
        "cpu_model": platform.processor() or platform.machine(),
        "cpu_count": psutil.cpu_count() if psutil else None,
        "memory_total_mb": psutil.virtual_memory().total // (1024 * 1024) if psutil else None,
        "python_version": platform.python_version(),
    }
    
    # Try to get temperature if available
    if psutil:
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get first available temperature
                for name, entries in temps.items():
                    if entries:
                        specs["temperature"] = entries[0].current
                        break
        except (AttributeError, OSError):
            pass
    
    return specs


def register_device(mqtt_client, device_id: str, device_type: str) -> None:
    """Register device with backend (one-time operation).
    
    Args:
        mqtt_client: MQTT client to publish with
        device_id: Unique device identifier
        device_type: Device type (e.g., "macos", "jetson")
    """
    # Ensure we have keypair
    private_key, public_key = ensure_keypair(device_id)
    
    # Get public key bytes
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    
    # Map device types to schema enum values
    type_mapping = {
        "macos": "other",
        "jetson": "jetson",
        "pi": "raspberry-pi",
        "raspberry-pi": "raspberry-pi",
    }
    
    # Build registration message matching the schema
    message = {
        "device_id": device_id,
        "type": type_mapping.get(device_type, "other"),  # Changed from device_type to type
        "jwt": "dev-token",  # TODO: Generate proper JWT
        "capabilities": ["camera", "metrics"],  # Basic capabilities
        "schema_versions": {
            "metrics": "v1.0.0",
            "alerts": "v1.0.0"
        },
        "public_key": public_key_bytes.decode("utf-8"),
        "specs": get_device_specs(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    
    # Publish registration
    topic = f"devices/{device_id}/register"
    mqtt_client.publish(topic, message)
