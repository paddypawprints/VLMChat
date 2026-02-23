"""Device registration and heartbeat management."""

import logging
import socket
import platform
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from .mqtt_client import MQTTClient

logger = logging.getLogger(__name__)


class Registration:
    """Handles device registration, PKI auth, and heartbeat."""
    
    def __init__(self, device_id: str, device_type: str, mqtt_client: MQTTClient,
                 device_name: Optional[str] = None):
        """
        Initialize registration handler.
        
        Args:
            device_id: Unique device identifier  
            device_type: Device type (mac, jetson, pi)
            mqtt_client: MQTT client instance
            device_name: Human-readable name (optional)
        """
        self.device_id = device_id
        self.device_type = device_type
        self.device_name = device_name or f"{device_type}-{device_id}"
        self.mqtt = mqtt_client
        
        self.private_key = None
        self.public_key = None
        self._load_or_generate_keypair()
        
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False
    
    def _load_or_generate_keypair(self):
        """Load existing or generate new Ed25519 keypair."""
        key_file = Path.home() / f".vlmchat_device_key_{self.device_id}.pem"
        
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519
            from cryptography.hazmat.primitives import serialization
            
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.private_key = serialization.load_pem_private_key(
                        f.read(), password=None
                    )
                logger.info(f"Loaded keypair from {key_file}")
            else:
                self.private_key = ed25519.Ed25519PrivateKey.generate()
                pem = self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                with open(key_file, 'wb') as f:
                    f.write(pem)
                key_file.chmod(0o600)
                logger.info(f"Generated new keypair at {key_file}")
            
            self.public_key = self.private_key.public_key()
            
        except ImportError:
            logger.warning("cryptography not available, PKI auth disabled")
    
    def get_public_key_pem(self) -> Optional[str]:
        """Get public key in PEM format."""
        if not self.public_key:
            return None
        
        try:
            from cryptography.hazmat.primitives import serialization
            pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return pem.decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to export public key: {e}")
            return None
    
    def get_device_specs(self) -> Dict[str, Any]:
        """Get device hardware specifications."""
        specs = {}
        
        try:
            import psutil
            
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                specs["cpu"] = f"{cpu_count} cores @ {cpu_freq.current:.0f}MHz"
            else:
                specs["cpu"] = f"{cpu_count} cores"
            
            mem = psutil.virtual_memory()
            specs["memory"] = f"{mem.total / (1024**3):.1f}GB"
            specs["usage"] = round(psutil.cpu_percent(interval=0.1), 1)
            
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            specs["temperature"] = round(entries[0].current, 1)
                            break
            except (AttributeError, OSError):
                pass
                
        except ImportError:
            specs["cpu"] = platform.processor() or platform.machine()
            specs["memory"] = "unknown"
        
        return specs
    
    def get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "unknown"
    
    def register(self, instruments: Optional[list] = None):
        """
        Send device registration message.
        
        Args:
            instruments: Optional list of available metric instruments
        """
        registration = {
            "name": self.device_name,
            "type": self.device_type,
            "ip": self.get_local_ip(),
            "specs": self.get_device_specs(),
            "timestamp": datetime.now().isoformat(),
        }
        
        public_key_pem = self.get_public_key_pem()
        if public_key_pem:
            registration["publicKey"] = public_key_pem
            registration["keyAlgorithm"] = "Ed25519"
        
        if instruments:
            registration["instruments"] = instruments
            logger.info(f"Including {len(instruments)} instruments in registration")
        
        topic = f"devices/{self.device_id}/register"
        self.mqtt.publish(topic, registration, qos=1)
        logger.info(f"✓ Sent registration to {topic}")
    
    def send_heartbeat(self):
        """Send heartbeat message."""
        heartbeat = {
            "timestamp": datetime.now().isoformat(),
            "status": "online",
        }
        topic = f"devices/{self.device_id}/heartbeat"
        self.mqtt.publish(topic, heartbeat, qos=0)
    
    def start_heartbeat(self, interval: int = 30):
        """
        Start heartbeat thread.
        
        Args:
            interval: Heartbeat interval in seconds
        """
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            logger.warning("Heartbeat already running")
            return
        
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(interval,),
            daemon=True
        )
        self._heartbeat_thread.start()
        logger.info(f"Heartbeat started (every {interval}s)")
    
    def stop_heartbeat(self):
        """Stop heartbeat thread."""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)
    
    def _heartbeat_loop(self, interval: int):
        """Background heartbeat loop."""
        while self._running:
            if self.mqtt.connected:
                self.send_heartbeat()
            time.sleep(interval)
