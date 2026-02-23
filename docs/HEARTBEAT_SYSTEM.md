# Device Heartbeat System

## Overview
The heartbeat system keeps track of device connectivity status by having devices send periodic "alive" messages to the server. This allows the web platform to accurately show which devices are online and when they were last seen.

## Architecture

### Device Side (Python)
Location: [`device_app.py`](device_app.py)

**Implementation:**
- Background thread sends heartbeat every 30 seconds
- Uses MQTT topic: `devices/{device_id}/heartbeat`
- Payload includes timestamp and status
- Starts automatically when device connects
- Uses QoS 0 (fire and forget) for efficiency

**Key Functions:**
- `_send_heartbeat()` - Sends a single heartbeat message
- `_heartbeat_loop()` - Background thread that sends heartbeats every 30 seconds
- Automatically starts when device runs `start()`

### Server Side (TypeScript)
Location: [`web-platform/server/mqtt.ts`](web-platform/server/mqtt.ts)

**Implementation:**
- Subscribes to `devices/+/heartbeat` topic
- Updates device `lastSeen` timestamp in database
- Lightweight update (only timestamp, no heavy processing)

**Key Functions:**
- `handleDeviceHeartbeat(deviceId)` - Updates lastSeen in database

### Database Schema
Location: [`web-platform/shared/schema.ts`](web-platform/shared/schema.ts)

**Device Table:**
- `lastSeen: timestamp("last_seen")` - Stores timestamp of last heartbeat

### Frontend (React/TypeScript)
Location: [`web-platform/client/src/components/DeviceConnection.tsx`](web-platform/client/src/components/DeviceConnection.tsx)

**Features:**
- Visual status indicator (Online/Offline badge)
- Animated pulse icon for online devices
- "Last heartbeat" timestamp display
- Auto-refresh every 10 seconds to update UI

**Status Logic:**
- Device is "Online" if heartbeat received within last 60 seconds
- Device is "Offline" if no heartbeat for >60 seconds
- Shows relative time: "Just now", "2 minutes ago", etc.

## Timing

| Component | Interval | Purpose |
|-----------|----------|---------|
| Device heartbeat | 30 seconds | Send alive signal |
| UI refresh | 10 seconds | Update display |
| Online threshold | 60 seconds | Consider device alive |

## Testing

### Run the Heartbeat Monitor
Monitor all device heartbeats in real-time:

```bash
python test_heartbeat.py
```

This will show:
- ✅ Heartbeat messages every 30 seconds
- 📝 Device registration messages
- 🔴/🟢 Status changes

### Expected Output
```
🔍 Device Heartbeat Monitor
==================================================

✓ Connected to MQTT broker at localhost:1883
✓ Subscribed to device heartbeat, register, and status topics
Monitoring heartbeats... (Ctrl+C to stop)

[14:23:45] 💚 HEARTBEAT from mac-mini-001
  └─ Status: online
  └─ Timestamp: 2025-12-29T14:23:45.123456

[14:24:15] 💚 HEARTBEAT from mac-mini-001
  └─ Status: online
  └─ Timestamp: 2025-12-29T14:24:15.234567
```

## Troubleshooting

### Device shows as "Offline" when it should be online

**Check:**
1. Device is connected to MQTT broker
2. Heartbeat thread is running:
   ```python
   # In device_app.py logs, you should see:
   # "✓ Device application started"
   ```
3. MQTT broker is receiving messages:
   ```bash
   mosquitto_sub -t "devices/+/heartbeat" -v
   ```

### Heartbeats not appearing in UI

**Check:**
1. Server is subscribed to heartbeat topic (check server logs)
2. Database `lastSeen` field is being updated:
   ```sql
   SELECT id, name, last_seen FROM devices;
   ```
3. Frontend is polling for updates (check Network tab, should refresh every 10s)

### Performance Concerns

**Current Load:**
- Each device: 1 message every 30 seconds
- 10 devices = ~20 messages/minute
- 100 devices = ~200 messages/minute

**Optimizations if needed:**
- Increase heartbeat interval (currently 30s, could go to 60s)
- Use Redis instead of database for lastSeen
- Batch database updates

## Implementation Details

### Device Heartbeat Message Format
```json
{
  "timestamp": "2025-12-29T14:23:45.123456",
  "status": "online"
}
```

### MQTT Topic Structure
```
devices/{device_id}/heartbeat
```

Example:
```
devices/mac-mini-001/heartbeat
devices/jetson-nano-042/heartbeat
```

## Future Enhancements

- [ ] Add heartbeat history/uptime metrics
- [ ] Alert when device goes offline unexpectedly
- [ ] Show network latency/jitter in heartbeats
- [ ] Device health metrics in heartbeat (CPU, memory, temp)
- [ ] Configurable heartbeat intervals per device type
