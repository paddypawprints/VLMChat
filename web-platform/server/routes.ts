import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { db } from "./db";
import { users, devices, chatMessages, deviceMessages, metricsCommandSchema } from "@shared/schema";
import { eq, and, desc, or, isNull } from "drizzle-orm";
import { readdir, readFile } from "fs/promises";
import { join } from "path";
import { getRedisClient } from "./redis";
import { publishToDevice } from "./mqtt";
import { createUserSession, getUserSession, deleteUserSession } from "./sessions";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
import { setupWebSocket } from "./websocket";
import { requireAuth } from "./core/middleware";
import { searchRoutes } from "./features/search";
import { configRoutes } from "./features/config";
import { adminRoutes } from "./features/admin";
// import { setupVideoProxy } from "./video-proxy";
// import { setupMediasoupSignaling } from "./mediasoup-signaling";

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

export async function registerRoutes(app: Express): Promise<Server> {
  // Log all incoming requests
  app.use((req, res, next) => {
    const start = Date.now();
    res.on('finish', () => {
      const duration = Date.now() - start;
      const status = res.statusCode;
      if (status >= 400) {
        console.log(`[HTTP ${status}] ${req.method} ${req.path} - ${duration}ms`);
      }
    });
    next();
  });

  // Register feature routes
  console.log('[Routes] Registering feature routes...');
  
  // Log admin routes in detail
  if (adminRoutes && adminRoutes.stack) {
    console.log('[Routes] adminRoutes has', adminRoutes.stack.length, 'routes:');
    adminRoutes.stack.forEach((layer: any, i: number) => {
      console.log(`  [${i}] layer.route exists:`, !!layer.route, 'layer.name:', layer.name);
      if (layer.route) {
        const methods = Object.keys(layer.route.methods).join(',').toUpperCase();
        console.log(`       ${methods} ${layer.route.path}`);
      } else if (layer.name === 'router') {
        console.log(`       (nested router, regexp: ${layer.regexp})`);
      }
    });
  }
  
  app.use('/api', searchRoutes);
  app.use('/api', configRoutes);
  app.use('/api', adminRoutes);
  
  // Log what routes are actually mounted on /api
  console.log('[Routes] Checking all /api mounts...');
  const apiLayers = (app as any)._router?.stack?.filter((l: any) => 
    l.regexp?.toString().includes('api') && l.name === 'router'
  );
  console.log('[Routes] Found', apiLayers?.length || 0, 'router mounts on /api');
  apiLayers?.forEach((layer: any, i: number) => {
    if (layer.handle && layer.handle.stack) {
      console.log(`[Routes] Mount ${i}: ${layer.handle.stack.length} routes`);
    }
  });
  
  console.log('[Routes] Feature routes registered');
  
  // Auth endpoints
  app.post("/api/auth/login", async (req, res) => {
    try {
      const { email, name } = req.body;
      
      // Use email as name if name not provided (demo mode)
      const userName = name || email.split('@')[0];
      
      // Find or create user
      let [user] = await db.select().from(users).where(eq(users.email, email));
      
      if (!user) {
        [user] = await db.insert(users).values({
          email,
          name: userName,
          provider: 'email'
        }).returning();
      }
      
      // Create session and return as bearer token
      const sessionId = await createUserSession(user.id);
      res.json({ user, sessionId });
    } catch (error) {
      console.error('Login error:', error);
      res.status(500).json({ error: 'Login failed', details: error.message });
    }
  });

  app.get("/api/auth/me", requireAuth, async (req, res) => {
    try {
      const userId = (req as any).user.userId;
      const [user] = await db.select().from(users).where(eq(users.id, userId));
      
      if (!user) {
        return res.status(404).json({ error: 'User not found' });
      }
      
      res.json({ user });
    } catch (error) {
      console.error('Get user error:', error);
      res.status(500).json({ error: 'Failed to get user' });
    }
  });

  app.post("/api/auth/logout", async (req, res) => {
    try {
      const authHeader = req.headers.authorization;
      if (authHeader && authHeader.startsWith('Bearer ')) {
        const sessionId = authHeader.substring(7);
        await deleteUserSession(sessionId);
      }
      res.json({ success: true });
    } catch (error) {
      console.error('Logout error:', error);
      res.status(500).json({ error: 'Logout failed' });
    }
  });

  // Device endpoints
  app.get("/api/devices", requireAuth, async (req, res) => {
    try {
      const userId = (req as any).user.userId;
      const { status } = req.query;
      
      // Show devices owned by user OR unassigned devices (userId is null)
      let conditions = [eq(devices.userId, userId), isNull(devices.userId)];
      
      if (status === 'active' || status === 'connected') {
        conditions.push(eq(devices.status, 'connected'));
      }
      
      const userDevices = await db.select().from(devices).where(or(...conditions));
      
      // Enrich with real-time status from Redis
      const enrichedDevices = await Promise.all(
        userDevices.map(async (device) => {
          const { isDeviceConnected } = await import('./sessions.js');
          const isOnline = await isDeviceConnected(device.id);
          return {
            deviceId: device.id,
            name: device.name,
            status: isOnline ? 'connected' : 'disconnected',
            userId: device.userId,
            lastSeen: device.lastSeen?.toISOString(),
            createdAt: device.manufacturedAt?.toISOString(),
          };
        })
      );
      
      res.json(enrichedDevices);
    } catch (error) {
      console.error('Get devices error:', error);
      res.status(500).json({ error: 'Failed to fetch devices' });
    }
  });

  app.post("/api/devices", async (req, res) => {
    try {
      const { id, name, type, ip, specs } = req.body;
      
      const [device] = await db.insert(devices).values({
        id,
        name,
        type,
        ip,
        specs,
        status: 'disconnected'
      }).returning();
      
      res.json(device);
    } catch (error) {
      res.status(500).json({ error: 'Failed to create device' });
    }
  });

  app.delete("/api/devices/:deviceId", requireAuth, async (req, res) => {
    try {
      const userId = (req as any).user.userId;
      const deviceId = req.params.deviceId;
      
      // Verify device ownership
      const [device] = await db.select().from(devices).where(eq(devices.id, deviceId));
      if (!device || device.userId !== userId) {
        return res.status(403).json({ error: 'Access denied' });
      }
      
      await db.delete(devices).where(eq(devices.id, deviceId));
      res.json({ success: true });
    } catch (error) {
      console.error('Delete device error:', error);
      res.status(500).json({ error: 'Failed to delete device' });
    }
  });

  // Device metrics endpoints
  app.post("/api/devices/:deviceId/metrics/configure", requireAuth, async (req, res) => {
    try {
      const userId = (req as any).user.userId;
      const deviceId = req.params.deviceId;
      
      // Verify device ownership
      const [device] = await db.select().from(devices).where(eq(devices.id, deviceId));
      if (!device || device.userId !== userId) {
        return res.status(403).json({ error: 'Access denied' });
      }
      
      // Validate payload
      const command = metricsCommandSchema.parse(req.body);
      
      // Publish command to device via MQTT
      await publishToDevice(deviceId, 'commands/metrics', command);
      
      res.json({ success: true, command });
    } catch (error) {
      console.error('Metrics configure error:', error);
      res.status(500).json({ error: 'Failed to configure metrics', details: error.message });
    }
  });

  app.get("/api/devices/:deviceId/metrics", requireAuth, async (req, res) => {
    try {
      const userId = (req as any).user.userId;
      const deviceId = req.params.deviceId;
      const redis = getRedisClient();
      
      // Verify device ownership
      const [device] = await db.select().from(devices).where(eq(devices.id, deviceId));
      if (!device || device.userId !== userId) {
        return res.status(403).json({ error: 'Access denied' });
      }
      
      // Disable caching for Safari and other browsers
      res.set('Cache-Control', 'no-store, no-cache, must-revalidate, private');
      res.set('Pragma', 'no-cache');
      res.set('Expires', '0');
      
      // Get latest metrics
      const latestKey = `device:${deviceId}:metrics:latest`;
      const latest = await redis.get(latestKey);
      
      let transformedLatest = null;
      if (latest) {
        const metricsData = JSON.parse(latest);
        
        // Transform schema format to frontend format
        transformedLatest = {
          timestamp: metricsData.timestamp
        };
        
        // Convert instruments array to flat object
        if (metricsData.instruments) {
          for (const instrument of metricsData.instruments) {
            // Map instrument names to frontend keys
            if (instrument.name === 'pipeline.fps') {
              transformedLatest.fps = instrument.value;
            } else if (instrument.name === 'pipeline.duration') {
              transformedLatest.avgDuration = instrument.value;
            } else if (instrument.name === 'cache.objects') {
              transformedLatest.totalObjects = instrument.value;
            } else if (instrument.name === 'cache.references') {
              transformedLatest.totalReferences = instrument.value;
            }
          }
        }
      }
      
      // Get history (optional, via query param)
      let history = null;
      if (req.query.includeHistory === 'true') {
        const historyKey = `device:${deviceId}:metrics:history`;
        const historyData = await redis.lrange(historyKey, 0, 99);
        history = historyData.map(item => JSON.parse(item));
      }
      
      res.json({
        latest: transformedLatest,
        history
      });
    } catch (error) {
      console.error('Metrics fetch error:', error);
      res.status(500).json({ error: 'Failed to fetch metrics', details: error.message });
    }
  });

  // Device messages endpoints
  app.get("/api/devices/:deviceId/messages", async (req, res) => {
    try {
      const messages = await db.select()
        .from(deviceMessages)
        .where(eq(deviceMessages.deviceId, req.params.deviceId))
        .orderBy(desc(deviceMessages.createdAt))
        .limit(100);
      
      res.json(messages);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch device messages' });
    }
  });

  // Device snapshot endpoints
  app.post("/api/devices/:deviceId/snapshot", async (req, res) => {
    try {
      const deviceId = req.params.deviceId;
      
      console.log(`[API] ⚡ Snapshot requested for device: ${deviceId}`);
      console.log('[API] ⚡ Sending MQTT command...');
      
      // Send MQTT command to device to capture snapshot
      await publishToDevice(deviceId, 'commands/snapshot', { 
        type: 'snapshot'
      });
      
      console.log(`[API] ✓ Snapshot command sent successfully to ${deviceId}`);
      res.status(202).json({ accepted: true });
    } catch (error) {
      console.error('[API] ❌ Snapshot request error:', error);
      res.status(500).json({ error: 'Failed to request snapshot', details: error.message });
    }
  });

  app.get("/api/devices/:deviceId/snapshot", async (req, res) => {
    try {
      const deviceId = req.params.deviceId;
      const redis = getRedisClient();
      
      // Get latest snapshot from Redis
      const snapshotKey = `device:${deviceId}:snapshot:latest`;
      const snapshot = await redis.get(snapshotKey);
      
      if (snapshot) {
        const data = JSON.parse(snapshot);
        res.json(data);
      } else {
        res.json({ image: null, timestamp: null });
      }
    } catch (error) {
      console.error('Snapshot fetch error:', error);
      res.status(500).json({ error: 'Failed to fetch snapshot', details: error.message });
    }
  });

  app.get("/api/devices/:deviceId", async (req, res) => {
    try {
      const [device] = await db.select()
        .from(devices)
        .where(eq(devices.id, req.params.deviceId))
        .limit(1);
      
      if (!device) {
        res.status(404).json({ error: 'Device not found' });
        return;
      }
      
      // Get real-time status from Redis
      const { isDeviceConnected } = await import('./sessions.js');
      const isOnline = await isDeviceConnected(device.id);
      
      // Map database fields to API response format
      res.json({
        deviceId: device.id,
        name: device.name,
        status: isOnline ? 'connected' : 'disconnected',
        userId: device.userId,
        lastSeen: device.lastSeen?.toISOString(),
        createdAt: device.manufacturedAt?.toISOString(),
      });
    } catch (error) {
      console.error('Device fetch error:', error);
      res.status(500).json({ error: 'Failed to fetch device' });
    }
  });

  // Chat endpoints
  app.get("/api/chat/messages", async (req, res) => {
    try {
      const { deviceId } = req.query;
      // TODO: Filter by userId from session
      
      let query = db.select().from(chatMessages);
      if (deviceId) {
        query = query.where(eq(chatMessages.deviceId, deviceId as string));
      }
      
      const messages = await query;
      res.json(messages);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch messages' });
    }
  });

  app.post("/api/chat/message", async (req, res) => {
    try {
      const { message, deviceId, debug } = req.body;
      // TODO: Get userId from session
      const userId = '00000000-0000-0000-0000-000000000000'; // Temporary
      
      // Save user message
      const [userMessage] = await db.insert(chatMessages).values({
        userId,
        deviceId: deviceId || null,
        role: 'user',
        content: message,
        images: [],
      }).returning();
      
      // Call AI service
      const aiResponse = await fetch(`${AI_SERVICE_URL}/api/infer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          temperature: 0.7,
          max_tokens: 150
        })
      });
      
      if (!aiResponse.ok) {
        throw new Error('AI service unavailable');
      }
      
      const aiData = await aiResponse.json();
      
      // Save AI response
      const [aiMessage] = await db.insert(chatMessages).values({
        userId,
        deviceId: deviceId || null,
        role: 'assistant',
        content: aiData.content,
        images: [],
        debug: debug ? aiData.debug : null
      }).returning();
      
      res.json({
        userMessage,
        aiMessage
      });
    } catch (error) {
      res.status(500).json({ error: 'Failed to send message' });
    }
  });

  // Metrics query endpoints
  app.get("/api/devices/:deviceId/metrics", async (req, res) => {
    try {
      const { deviceId } = req.params;
      
      // Get metrics config messages for this device
      const configMessages = await db.select()
        .from(deviceMessages)
        .where(
          and(
            eq(deviceMessages.deviceId, deviceId),
            eq(deviceMessages.topic, `devices/${deviceId}/metrics/config`)
          )
        )
        .orderBy(desc(deviceMessages.createdAt))
        .limit(1);
      
      if (configMessages.length === 0) {
        return res.json({ sessions: [] });
      }
      
      const config = configMessages[0].payload as any;
      res.json(config);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch metrics config' });
    }
  });

  app.get("/api/devices/:deviceId/metrics/:session", async (req, res) => {
    try {
      const { deviceId, session } = req.params;
      const limit = parseInt(req.query.limit as string) || 100;
      
      // Get latest metrics data for this session
      const metricsMessages = await db.select()
        .from(deviceMessages)
        .where(
          and(
            eq(deviceMessages.deviceId, deviceId),
            eq(deviceMessages.topic, `devices/${deviceId}/metrics/data/${session}`)
          )
        )
        .orderBy(desc(deviceMessages.createdAt))
        .limit(limit);
      
      const data = metricsMessages.map(msg => ({
        timestamp: msg.createdAt,
        ...msg.payload
      }));
      
      res.json({ session, data });
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch metrics data' });
    }
  });

  app.post("/api/devices/:deviceId/metrics/subscribe", async (req, res) => {
    try {
      const { deviceId } = req.params;
      const { session, interval_seconds = 5 } = req.body;
      
      // Publish subscribe command via MQTT
      const mqtt = (global as any).mqttClient;
      if (!mqtt) {
        return res.status(503).json({ error: 'MQTT not available' });
      }
      
      mqtt.publish(
        `devices/${deviceId}/metrics/subscribe`,
        JSON.stringify({ session, interval_seconds }),
        { qos: 1 }
      );
      
      res.json({ success: true, session, interval_seconds });
    } catch (error) {
      res.status(500).json({ error: 'Failed to subscribe to metrics' });
    }
  });

  app.post("/api/devices/:deviceId/metrics/unsubscribe", async (req, res) => {
    try {
      const { deviceId } = req.params;
      const { session } = req.body;
      
      // Publish unsubscribe command via MQTT
      const mqtt = (global as any).mqttClient;
      if (!mqtt) {
        return res.status(503).json({ error: 'MQTT not available' });
      }
      
      mqtt.publish(
        `devices/${deviceId}/metrics/unsubscribe`,
        JSON.stringify({ session }),
        { qos: 1 }
      );
      
      res.json({ success: true, session });
    } catch (error) {
      res.status(500).json({ error: 'Failed to unsubscribe from metrics' });
    }
  });

  const httpServer = createServer(app);
  
  // Setup WebSocket servers
  setupWebSocket(httpServer);
  // setupVideoProxy(httpServer);
  // setupMediasoupSignaling(httpServer);
  
  return httpServer;
}
