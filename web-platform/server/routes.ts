import type { Express } from "express";
import { createServer, type Server } from "http";
import { db } from "./db";
import { users, devices, chatMessages, deviceMessages } from "@shared/schema";
import { eq, and, desc } from "drizzle-orm";
import { readdir, readFile } from "fs/promises";
import { join } from "path";

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

export async function registerRoutes(app: Express): Promise<Server> {
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
      
      res.json({ user });
    } catch (error) {
      console.error('Login error:', error);
      res.status(500).json({ error: 'Login failed', details: error.message });
    }
  });

  app.get("/api/auth/me", async (req, res) => {
    // TODO: Implement session management
    res.json({ user: null });
  });

  app.post("/api/auth/logout", async (req, res) => {
    // TODO: Implement session management
    res.json({ success: true });
  });

  // Device endpoints
  app.get("/api/devices", async (req, res) => {
    try {
      // TODO: Filter by userId from session
      const allDevices = await db.select().from(devices);
      res.json(allDevices);
    } catch (error) {
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

  app.delete("/api/devices/:id", async (req, res) => {
    try {
      await db.delete(devices).where(eq(devices.id, req.params.id));
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: 'Failed to delete device' });
    }
  });

  // Device messages endpoints
  app.get("/api/devices/:id/messages", async (req, res) => {
    try {
      const messages = await db.select()
        .from(deviceMessages)
        .where(eq(deviceMessages.deviceId, req.params.id))
        .orderBy(desc(deviceMessages.createdAt))
        .limit(100);
      
      res.json(messages);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch device messages' });
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

  // Schema endpoints
  app.get("/api/schemas", async (req, res) => {
    try {
      const schemasDir = join(process.cwd(), 'shared', 'schemas');
      const topics = await readdir(schemasDir);
      
      const schemas = [];
      for (const topic of topics) {
        const topicPath = join(schemasDir, topic);
        const versions = await readdir(topicPath);
        
        for (const version of versions) {
          schemas.push({
            topic,
            version,
            url: `/api/schemas/${topic}/${version}/schema.json`
          });
        }
      }
      
      res.json({ schemas });
    } catch (error) {
      res.status(500).json({ error: 'Failed to list schemas' });
    }
  });

  app.get("/api/schemas/:topic/:version/schema.json", async (req, res) => {
    try {
      const { topic, version } = req.params;
      const schemaPath = join(process.cwd(), 'shared', 'schemas', topic, version, 'schema.json');
      const schemaContent = await readFile(schemaPath, 'utf-8');
      
      res.json(JSON.parse(schemaContent));
    } catch (error) {
      res.status(404).json({ error: 'Schema not found' });
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
  return httpServer;
}
