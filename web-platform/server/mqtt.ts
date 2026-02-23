import mqtt from 'mqtt';
import { db } from './db';
import { deviceMessages, devices, searchTerms, deviceConfigs } from '@shared/schema';
import { eq, desc, and } from 'drizzle-orm';
import { getRedisClient } from './redis';
import { createDeviceSession, deleteDeviceSession, isDeviceConnected } from './sessions';
import { validateMQTTMessage } from './mqtt-validator';
import { broadcastToClients } from './websocket';

const MQTT_URL = process.env.MQTT_URL || 'mqtt://localhost:1883';

export function setupMQTT() {
  const client = mqtt.connect(MQTT_URL);

  // Store client globally for API routes to use
  (global as any).mqttClient = client;

  client.on('connect', () => {
    console.log('[MQTT] Connected to broker');
    
    // Subscribe to all device topics
    client.subscribe('devices/+/register', { qos: 1 });
    client.subscribe('devices/+/status', { qos: 1 });
    client.subscribe('devices/+/heartbeat', { qos: 1 });
    client.subscribe('devices/+/alerts', { qos: 1 });
    client.subscribe('devices/+/snapshot', { qos: 1 });
    client.subscribe('devices/+/metrics', { qos: 0 });
    client.subscribe('devices/+/logs', { qos: 0 });
    client.subscribe('devices/+/webrtc/+', { qos: 1 });
    
    console.log('[MQTT] Subscribed to device topics');
  });

  client.on('message', async (topic, message, packet) => {
    try {
      // Parse topic
      const parts = topic.split('/');
      if (parts.length < 3 || parts[0] !== 'devices') {
        console.warn(`[MQTT] Invalid topic format: ${topic}`);
        return;
      }

      const deviceId = parts[1];
      const messageType = parts[2];
      const subType = parts[3]; // For metrics/data/{session} or webrtc/{type}
      
      let payload: any;
      try {
        payload = JSON.parse(message.toString());
      } catch {
        payload = { raw: message.toString() };
      }
      
      // ============================================================
      // DO NOT REMOVE OR MODIFY - SECURITY CRITICAL
      // Validate all incoming MQTT messages against schemas
      // Rejects unknown topics and invalid payloads
      // ============================================================
      try {
        validateMQTTMessage(topic, payload);  // Throws on validation failure
      } catch (validationError: any) {
        console.error('═'.repeat(80));
        console.error('[MQTT] 🚨 VALIDATION FAILED - UNRECOVERABLE ERROR');
        console.error('[MQTT] Topic:', topic);
        console.error('[MQTT] Payload (full):', JSON.stringify(payload, null, 2));
        console.error('[MQTT] Validation Error:', validationError?.message || validationError);
        console.error('═'.repeat(80));
        console.error('[MQTT] 💀 Schema mismatch detected - exiting server');
        console.error('[MQTT] 💀 This indicates device schemas are out of sync with server');
        console.error('[MQTT] 💀 Run: just generate && docker-compose up --build');
        console.error('═'.repeat(80));
        // UNRECOVERABLE ERROR - Schema mismatch will cause data corruption
        process.exit(1);
      }
      // ============================================================
      // END SECURITY CRITICAL SECTION
      // ============================================================

      // Handle device registration FIRST (before storing message)
      if (messageType === 'register') {
        await handleDeviceRegister(deviceId, payload);
      }

      // Handle status updates
      if (messageType === 'status') {
        await handleDeviceStatus(deviceId, payload);
      }

      // Handle heartbeat
      if (messageType === 'heartbeat') {
        await handleDeviceHeartbeat(deviceId);
      }

      // Handle metrics data
      if (messageType === 'metrics') {
        await handleDeviceMetrics(deviceId, payload);
        // Don't store metrics in deviceMessages - they go to Redis
        return;
      }

      // Handle log entries
      if (messageType === 'logs') {
        await handleDeviceLogs(deviceId, payload);
        // Don't store logs in deviceMessages - broadcast directly
        return;
      }

      // Handle snapshot data
      if (messageType === 'snapshot') {
        await handleDeviceSnapshot(deviceId, payload);
        // Don't store snapshots in deviceMessages - they go to Redis
        return;
      }

      // Handle alerts
      if (messageType === 'alerts') {
        await handleDeviceAlert(deviceId, payload);
        // Store alert in deviceMessages for history
        // (continue to db.insert below)
      }

      // Store all messages in deviceMessages table (after device exists)
      try {
        await db.insert(deviceMessages).values({
          deviceId,
          topic,
          payload,
          qos: packet.qos?.toString(),
          retained: packet.retain || false,
        });
        console.log(`[MQTT] Message stored: ${topic}`);
      } catch (dbError: any) {
        // If device doesn't exist yet, skip storing message (it's a race condition)
        if (dbError.code === '23503') {
          console.log(`[MQTT] Skipping message storage for unregistered device: ${deviceId}`);
        } else {
          throw dbError;
        }
      }
    } catch (error) {
      console.error('[MQTT] Error processing message:', error);
    }
  });

  client.on('error', (error) => {
    console.error('[MQTT] Connection error:', error);
  });

  client.on('offline', () => {
    console.log('[MQTT] Client offline');
  });

  client.on('reconnect', () => {
    console.log('[MQTT] Reconnecting...');
  });

  return client;
}

// Handler functions for different message types
async function handleDeviceRegister(deviceId: string, payload: any) {
  try {
    console.log(`[MQTT] Registration payload for ${deviceId}:`, JSON.stringify(payload));
    
    // Check if device already has an active session
    const alreadyConnected = await isDeviceConnected(deviceId);
    if (alreadyConnected) {
      console.warn(`[MQTT] Device ${deviceId} already connected - sending filter anyway`);
      // Device might have reconnected and missed the initial filter publish
      // Always send filters to ensure device has current configuration
      await publishCurrentFilterToDevice(deviceId);
      return;
    }
    
    // Create device session
    const clientId = payload.clientId || deviceId;
    const sessionCreated = await createDeviceSession(deviceId, clientId);
    if (!sessionCreated) {
      console.warn(`[MQTT] Failed to create session for device ${deviceId}`);
      return;
    }
    
    // Create or update device record
    const existingDevice = await db.select().from(devices).where(eq(devices.id, deviceId)).limit(1);
    
    if (existingDevice.length > 0) {
      // Update existing device
      const updateData: any = {
        name: payload.name || deviceId,
        type: payload.type || 'unknown',
        ip: payload.ip || '',
        status: 'connected',
        lastSeen: new Date(),
        specs: payload.specs || {}
      };
      
      // Update public key if provided
      if (payload.publicKey) {
        updateData.publicKey = payload.publicKey;
        updateData.keyAlgorithm = payload.keyAlgorithm || 'Ed25519';
      }
      
      await db.update(devices)
        .set(updateData)
        .where(eq(devices.id, deviceId));
      
      console.log(`[MQTT] Device ${deviceId} re-registered`);
    } else {
      // Create new device
      await db.insert(devices).values({
        id: deviceId,
        name: payload.name || deviceId,
        type: payload.type || 'unknown',
        ip: payload.ip || '',
        publicKey: payload.publicKey || null,
        keyAlgorithm: payload.keyAlgorithm || 'Ed25519',
        status: 'connected',
        userId: null, // TODO: Associate with user
        lastSeen: new Date(),
        specs: payload.specs || {}
      });
      
      console.log(`[MQTT] New device registered: ${deviceId}`);
    }
    
    // Publish current filter configuration to the new device
    await publishCurrentFilterToDevice(deviceId);
    
    // Publish current device configuration if one exists
    await publishCurrentConfigToDevice(deviceId);
    
  } catch (error) {
    console.error(`[MQTT] Error registering device ${deviceId}:`, error);
  }
}

async function handleDeviceStatus(deviceId: string, payload: any) {
  try {
    const status = payload.status || 'connected';
    
    // If status is offline/disconnected, remove device session
    if (status === 'offline' || status === 'disconnected') {
      await deleteDeviceSession(deviceId);
    }
    
    await db.update(devices)
      .set({ 
        status,
        lastSeen: new Date()
      })
      .where(eq(devices.id, deviceId));
    
    console.log(`[MQTT] Device ${deviceId} status: ${status}`);
  } catch (error) {
    console.error(`[MQTT] Error updating device status:`, error);
  }
}

async function handleDeviceHeartbeat(deviceId: string) {
  try {
    console.log(`[MQTT] Updating heartbeat for device ${deviceId}`);
    const result = await db.update(devices)
      .set({ lastSeen: new Date() })
      .where(eq(devices.id, deviceId));
    console.log(`[MQTT] Heartbeat updated for device ${deviceId}`, result);
  } catch (error) {
    console.error(`[MQTT] Error updating device heartbeat:`, error);
  }
}

async function handleDeviceMetrics(deviceId: string, payload: any) {
  try {
    const redis = getRedisClient();
    
    // Store metrics in Redis with 1 hour TTL
    const key = `device:${deviceId}:metrics:latest`;
    await redis.setex(key, 3600, JSON.stringify(payload));
    
    // Also store timestamped version for history (keep last 100)
    const historyKey = `device:${deviceId}:metrics:history`;
    const timestamp = payload.timestamp || new Date().toISOString();
    await redis.lpush(historyKey, JSON.stringify({ ...payload, _stored: new Date().toISOString() }));
    await redis.ltrim(historyKey, 0, 99); // Keep last 100 entries
    await redis.expire(historyKey, 3600); // 1 hour TTL
    
    console.log(`[MQTT] Stored metrics for device ${deviceId}`);
    
    // Broadcast metrics to connected WebSocket clients matching schema
    // payload already has { session, timestamp, instruments }
    console.log(`[MQTT] 📊 Preparing to broadcast metrics for device ${deviceId}`);
    const { broadcastToClients } = await import('./websocket');
    broadcastToClients(deviceId, {
      type: 'metrics',
      message: payload  // Send the metrics data directly (matches metrics-data-v1.0.0.json)
    });
    console.log(`[MQTT] ✅ Metrics broadcast completed for device ${deviceId}`);
  } catch (error) {
    console.error(`[MQTT] Error storing device metrics:`, error);
  }
}

// Helper function to publish messages to devices
export async function publishToDevice(deviceId: string, type: string, payload: any) {
  const client = (global as any).mqttClient;
  if (!client) {
    console.error('[MQTT] ❌ MQTT client not initialized!');
    throw new Error('MQTT client not initialized');
  }
  
  const topic = `devices/${deviceId}/${type}`;
  console.log(`[MQTT] 📤 Publishing to ${topic}:`, JSON.stringify(payload));
  
  return new Promise<void>((resolve, reject) => {
    client.publish(topic, JSON.stringify(payload), { qos: 1 }, (err: Error | undefined) => {
      if (err) {
        console.error(`[MQTT] ❌ Failed to publish to ${topic}:`, err);
        reject(err);
      } else {
        console.log(`[MQTT] ✓ Published to ${topic}`);
        resolve();
      }
    });
  });
}

async function handleDeviceSnapshot(deviceId: string, payload: any) {
  try {
    const redis = getRedisClient();
    const snapshotKey = `device:${deviceId}:snapshot:latest`;
    
    // Build snapshot data matching snapshot-v1.0.0.json schema
    const snapshotData: any = {
      device_id: deviceId,
      timestamp: payload.timestamp || new Date().toISOString(),
      image: payload.image
    };
    
    // Add optional fields only if present
    if (payload.format) snapshotData.format = payload.format;
    if (payload.width) snapshotData.width = payload.width;
    if (payload.height) snapshotData.height = payload.height;
    if (payload.detections) snapshotData.detections = payload.detections;
    
    // Store latest snapshot in Redis
    await redis.set(snapshotKey, JSON.stringify(snapshotData));
    
    // Set expiry (snapshots expire after 5 minutes)
    await redis.expire(snapshotKey, 300);
    
    // Broadcast to WebSocket clients
    broadcastToClients(deviceId, {
      type: 'snapshot',
      message: snapshotData
    });
    
    console.log(`[MQTT] Stored and broadcasted snapshot for device ${deviceId}`);
  } catch (error) {
    console.error(`[MQTT] Error storing device snapshot:`, error);
  }
}
async function handleDeviceLogs(deviceId: string, payload: any) {
  try {
    // Broadcast log entry directly to WebSocket clients
    // No storage needed - logs are ephemeral and kept in browser only
    broadcastToClients(deviceId, {
      type: 'logs',
      message: payload
    });
    
    console.log(`[MQTT] Broadcasted log entry for device ${deviceId}: ${payload.level} - ${payload.message}`);
  } catch (error) {
    console.error(`[MQTT] Error broadcasting device logs:`, error);
  }
}

async function handleDeviceAlert(deviceId: string, payload: any) {
  try {
    const redis = getRedisClient();
    
    // Enrich alert with search string from watchlist item
    let enrichedPayload = { ...payload };
    if (payload.watchlist_item_id) {
      try {
        const [searchTerm] = await db.select()
          .from(searchTerms)
          .where(eq(searchTerms.id, payload.watchlist_item_id))
          .limit(1);
        
        if (searchTerm) {
          enrichedPayload.search_string = searchTerm.searchString;
          console.log(`[MQTT] Enriched alert with search string: "${searchTerm.searchString}"`);
        }
      } catch (err) {
        console.warn(`[MQTT] Failed to fetch search string for watchlist_item_id ${payload.watchlist_item_id}:`, err);
      }
    }
    
    // Store latest alert in Redis (for API queries)
    const alertKey = `device:${deviceId}:alert:latest`;
    await redis.set(alertKey, JSON.stringify(enrichedPayload));
    await redis.expire(alertKey, 3600); // 1 hour TTL
    
    // Store in alert history (keep last 50 alerts)
    const historyKey = `device:${deviceId}:alerts:history`;
    await redis.lpush(historyKey, JSON.stringify({
      ...enrichedPayload,
      _stored: new Date().toISOString()
    }));
    await redis.ltrim(historyKey, 0, 49); // Keep last 50
    await redis.expire(historyKey, 86400); // 24 hour TTL
    
    console.log(`[MQTT] Stored alert for device ${deviceId}: ${enrichedPayload.description || enrichedPayload.type}`);
    
    // Broadcast to WebSocket clients (with search string)
    console.log(`[MQTT] 🚨 Preparing to broadcast alert for device ${deviceId}`);
    broadcastToClients(deviceId, {
      type: 'alert',
      message: enrichedPayload
    });
    
    console.log(`[MQTT] ✅ Alert broadcast completed for device ${deviceId}`);
  } catch (error) {
    console.error(`[MQTT] Error handling device alert:`, error);
  }
}

/**
 * Publish current filter configuration to a device
 * Called when device registers to sync initial state
 */
async function publishCurrentFilterToDevice(deviceId: string) {
  try {
    // Get MQTT client
    const mqtt = (global as any).mqttClient;
    if (!mqtt) {
      console.warn(`[MQTT] MQTT client not available, cannot sync filter to ${deviceId}`);
      return;
    }

    // Devices are currently unowned (userId: null) until a claim flow is implemented.
    // Send all search terms on connect so the device has the current filter set.
    // TODO: scope to device.userId once device claiming is wired up.
    const allTerms = await db
      .select()
      .from(searchTerms)
      .orderBy(desc(searchTerms.createdAt));

    if (allTerms.length === 0) {
      console.log(`[MQTT] No search terms found, skipping filter sync for ${deviceId}`);
      return;
    }

    const filters = allTerms.map(term => {
      const groq = (term.groqResponse as any) || {};
      const strategy = groq.strategy || {};
      return {
        id: term.id,
        name: term.searchString,
        category_mask: term.categoryMask,
        category_colors: term.categoryColors,
        attribute_mask: term.attributeMask,
        attribute_colors: term.attributeColors,
        color_requirements: term.colorRequirements || {},
        vlm_required: strategy.requires_vlm ?? false,
        vlm_reasoning: strategy.vlm_reasoning ?? '',
      };
    });

    const filterList = { filters };

    console.log(`[MQTT] Publishing ${filters.length} filters to device ${deviceId}:`);
    filters.forEach(f => {
      const catCount = (f.category_mask as boolean[]).filter(x => x).length;
      const attrCount = (f.attribute_mask as boolean[]).filter(x => x).length;
      const colorReqKeys = Object.keys(f.color_requirements || {});
      console.log(`  - "${f.name}": ${catCount}/80 cats, ${attrCount}/26 attrs, colors: ${colorReqKeys.join(',')}`);
    });

    mqtt.publish(
      `devices/${deviceId}/commands/filter`,
      JSON.stringify(filterList),
      { qos: 1 }
    );

    console.log(`[MQTT] Filter message published to devices/${deviceId}/commands/filter`);
  } catch (error) {
    console.error(`[MQTT] Error publishing filters to device ${deviceId}:`, error);
  }
}

/**
 * Publish current device configuration to a device
 * Called when device registers to sync configuration state
 */
async function publishCurrentConfigToDevice(deviceId: string) {
  try {
    // Get active configuration for this device
    const [activeConfig] = await db
      .select()
      .from(deviceConfigs)
      .where(
        and(
          eq(deviceConfigs.deviceId, deviceId),
          eq(deviceConfigs.isActive, true)
        )
      )
      .orderBy(desc(deviceConfigs.updatedAt))
      .limit(1);
    
    if (!activeConfig) {
      console.log(`[MQTT] No configuration found for device ${deviceId}, skipping config sync`);
      return;
    }
    
    // Get MQTT client
    const mqtt = (global as any).mqttClient;
    if (!mqtt) {
      console.warn(`[MQTT] MQTT client not available, cannot sync config to ${deviceId}`);
      return;
    }
    
    const configCommand = {
      version: activeConfig.version,
      config: activeConfig.config,
    };
    
    console.log(`[MQTT] Publishing config v${activeConfig.version} to device ${deviceId}`);
    
    mqtt.publish(
      `devices/${deviceId}/commands/config`,
      JSON.stringify(configCommand),
      { qos: 1 }
    );
    
    console.log(`[MQTT] Config message published to devices/${deviceId}/commands/config`);
  } catch (error) {
    console.error(`[MQTT] Error publishing config to device ${deviceId}:`, error);
  }
}
