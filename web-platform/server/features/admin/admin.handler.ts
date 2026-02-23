import { Request, Response } from 'express';
import { db } from '../../db';
import { devices } from '@shared/schema';
import { eq, desc } from 'drizzle-orm';

/**
 * Admin Device Management Handlers
 */

export async function listDevices(req: Request, res: Response) {
  try {
    const allDevices = await db.select().from(devices).orderBy(desc(devices.lastSeen));
    
    // Map devices and normalize status field to commissioning states only
    // Also serialize Date objects to ISO strings
    const mappedDevices = allDevices.map(device => ({
      ...device,
      // If status is invalid (like 'offline'), map to 'disconnected'
      status: ['manufactured', 'registered', 'connected', 'disconnected'].includes(device.status)
        ? device.status
        : 'disconnected',
      // Serialize dates to ISO strings
      manufacturedAt: device.manufacturedAt?.toISOString() || null,
      lastSeen: device.lastSeen?.toISOString() || null,
      claimedAt: device.claimedAt?.toISOString() || null
    }));
    
    console.log('[Admin] Returning devices:', JSON.stringify(mappedDevices, null, 2));
    res.json(mappedDevices);
  } catch (error) {
    console.error('Error fetching devices:', error);
    res.status(500).json({ error: 'Failed to fetch devices' });
  }
}

export async function createDevice(req: Request, res: Response) {
  try {
    const { id, name, type, ip } = req.body;
    
    // Validate required fields
    if (!id || !name || !type || !ip) {
      return res.status(400).json({ error: 'Missing required fields: id, name, type, ip' });
    }
    
    const [newDevice] = await db.insert(devices).values({
      id,
      name,
      type,
      ip,
      status: 'manufactured'
    }).returning();
    
    res.status(201).json(newDevice);
  } catch (error) {
    console.error('Error creating device:', error);
    res.status(500).json({ error: 'Failed to create device' });
  }
}

export async function updateDevice(req: Request, res: Response) {
  try {
    const { deviceId } = req.params;
    const updates = req.body;
    
    // Get existing device
    const [existingDevice] = await db
      .select()
      .from(devices)
      .where(eq(devices.id, deviceId))
      .limit(1);
    
    if (!existingDevice) {
      return res.status(404).json({ error: 'Device not found' });
    }
    
    const [updated] = await db
      .update(devices)
      .set({
        ...(updates.name && { name: updates.name }),
        ...(updates.type && { type: updates.type }),
        ...(updates.ip && { ip: updates.ip }),
        ...(updates.status && { status: updates.status })
      })
      .where(eq(devices.id, deviceId))
      .returning();
    
    res.json(updated);
  } catch (error) {
    console.error('Error updating device:', error);
    res.status(500).json({ error: 'Failed to update device' });
  }
}

export async function deleteDevice(req: Request, res: Response) {
  try {
    const { deviceId } = req.params;
    
    const [deleted] = await db
      .delete(devices)
      .where(eq(devices.id, deviceId))
      .returning();
    
    if (!deleted) {
      return res.status(404).json({ error: 'Device not found' });
    }
    
    res.json({ success: true, deviceId });
  } catch (error) {
    console.error('Error deleting device:', error);
    res.status(500).json({ error: 'Failed to delete device' });
  }
}

/**
 * Admin Service Management Handlers
 * 
 * Shows infrastructure service status (MQTT, PostgreSQL, Redis, WebSocket)
 */

interface AdminService {
  id: string;
  name: string;
  type: string;
  status: 'connected' | 'disconnected' | 'running' | 'stopped';
  metadata?: Record<string, any>;
}

export async function listServices(req: Request, res: Response) {
  try {
    const services: AdminService[] = [];
    
    // MQTT Service
    const mqttClient = (global as any).mqttClient;
    services.push({
      id: 'mqtt',
      name: 'MQTT Broker',
      type: 'mqtt',
      status: mqttClient?.connected ? 'connected' : 'disconnected',
      metadata: {
        url: process.env.MQTT_URL || 'mqtt://localhost:1883',
        topics: [
          'devices/+/register',
          'devices/+/status',
          'devices/+/heartbeat',
          'devices/+/alerts',
          'devices/+/snapshot',
          'devices/+/metrics',
          'devices/+/logs',
          'devices/+/webrtc/+'
        ]
      }
    });
    
    // PostgreSQL Service
    try {
      await db.select().from(devices).limit(1);
      services.push({
        id: 'postgres',
        name: 'PostgreSQL',
        type: 'database',
        status: 'connected',
        metadata: {
          url: process.env.DATABASE_URL?.replace(/:[^:]*@/, ':****@') || 'postgres://localhost:5432'
        }
      });
    } catch {
      services.push({
        id: 'postgres',
        name: 'PostgreSQL',
        type: 'database',
        status: 'disconnected'
      });
    }
    
    // Redis Service
    try {
      const { getRedisClient } = await import('../../redis.js');
      const redis = getRedisClient();
      await redis.ping();
      services.push({
        id: 'redis',
        name: 'Redis',
        type: 'cache',
        status: 'connected',
        metadata: {
          url: process.env.REDIS_URL || 'redis://localhost:6379'
        }
      });
    } catch {
      services.push({
        id: 'redis',
        name: 'Redis',
        type: 'cache',
        status: 'disconnected'
      });
    }
    
    // WebSocket Service
    const wsServer = (global as any).wsServer;
    services.push({
      id: 'websocket',
      name: 'WebSocket',
      type: 'websocket',
      status: wsServer ? 'running' : 'stopped',
      metadata: {
        clients: wsServer?.clients?.size || 0
      }
    });
    
    console.log('[Admin] Returning services:', JSON.stringify(services, null, 2));
    res.json(services);
  } catch (error) {
    console.error('Error fetching services:', error);
    res.status(500).json({ error: 'Failed to fetch services' });
  }
}

export async function createService(req: Request, res: Response) {
  res.status(501).json({ error: 'Service creation not supported - services are managed by infrastructure' });
}

export async function updateService(req: Request, res: Response) {
  res.status(501).json({ error: 'Service updates not supported - services are managed by infrastructure' });
}

export async function deleteService(req: Request, res: Response) {
  res.status(501).json({ error: 'Service deletion not supported - services are managed by infrastructure' });
}
