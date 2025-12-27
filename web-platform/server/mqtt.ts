import mqtt from 'mqtt';
import { db } from './db';
import { deviceMessages, devices } from '@shared/schema';
import { eq } from 'drizzle-orm';

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
    client.subscribe('devices/+/metrics/config', { qos: 1 });
    client.subscribe('devices/+/metrics/data/+', { qos: 1 });
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

      // Handle device registration
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

      // Store all messages in deviceMessages table
      await db.insert(deviceMessages).values({
        deviceId,
        topic,
        payload,
        qos: packet.qos?.toString(),
        retained: packet.retain || false,
      });

      console.log(`[MQTT] Message stored: ${topic}`);
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
    // Create or update device record
    const existingDevice = await db.select().from(devices).where(eq(devices.id, deviceId)).limit(1);
    
    if (existingDevice.length > 0) {
      // Update existing device
      await db.update(devices)
        .set({
          name: payload.name || deviceId,
          type: payload.type || 'unknown',
          ip: payload.ip || '',
          status: 'connected',
          lastSeen: new Date(),
          specs: payload.specs || {}
        })
        .where(eq(devices.id, deviceId));
      
      console.log(`[MQTT] Device ${deviceId} re-registered`);
    } else {
      // Create new device
      await db.insert(devices).values({
        id: deviceId,
        name: payload.name || deviceId,
        type: payload.type || 'unknown',
        ip: payload.ip || '',
        status: 'connected',
        userId: null, // TODO: Associate with user
        lastSeen: new Date(),
        specs: payload.specs || {}
      });
      
      console.log(`[MQTT] New device registered: ${deviceId}`);
    }
  } catch (error) {
    console.error(`[MQTT] Error registering device ${deviceId}:`, error);
  }
}

async function handleDeviceStatus(deviceId: string, payload: any) {
  try {
    const status = payload.status || 'connected';
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
    await db.update(devices)
      .set({ lastSeen: new Date() })
      .where(eq(devices.id, deviceId));
  } catch (error) {
    console.error(`[MQTT] Error updating device heartbeat:`, error);
  }
}

// Helper function to publish messages to devices
export async function publishToDevice(deviceId: string, type: string, payload: any) {
  const client = (global as any).mqttClient;
  if (!client) {
    throw new Error('MQTT client not initialized');
  }
  
  const topic = `devices/${deviceId}/${type}`;
  return new Promise<void>((resolve, reject) => {
    client.publish(topic, JSON.stringify(payload), { qos: 1 }, (err: Error | undefined) => {
      if (err) {
        console.error(`[MQTT] Failed to publish to ${topic}:`, err);
        reject(err);
      } else {
        console.log(`[MQTT] Published to ${topic}`);
        resolve();
      }
    });
  });
}
