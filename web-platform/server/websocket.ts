import { WebSocketServer, WebSocket } from 'ws';
import type { Server } from 'http';
import { parse } from 'url';
import { validateClientMessage, validateServerMessage } from './validation';

// Store browser client connections by deviceId
const clientConnections = new Map<string, Set<WebSocket>>();

// Ping interval for keepalive
const PING_INTERVAL = 30000; // 30 seconds
const PONG_TIMEOUT = 5000; // 5 seconds to respond

export function setupWebSocket(httpServer: Server) {
  const wss = new WebSocketServer({ 
    server: httpServer,
    path: '/ws'
  });

  // Ping all clients periodically to detect dead connections
  const pingInterval = setInterval(() => {
    clientConnections.forEach((clients, deviceId) => {
      clients.forEach((ws) => {
        // Check if alive from previous ping
        if ((ws as any).isAlive === false) {
          console.log(`[WebSocket] Client for device ${deviceId} did not respond to ping, terminating`);
          ws.terminate();
          return;
        }
        
        // Mark as waiting for pong, will be set true when pong received
        (ws as any).isAlive = false;
        ws.ping();
      });
    });
  }, PING_INTERVAL);

  wss.on('close', () => {
    clearInterval(pingInterval);
  });

  wss.on('connection', (ws: WebSocket, request) => {
    const { query } = parse(request.url || '', true);
    const deviceId = query.deviceId as string;

    if (!deviceId) {
      console.warn('[WebSocket] Connection rejected - no deviceId specified');
      ws.close(1008, 'deviceId required');
      return;
    }

    // Mark as alive initially
    (ws as any).isAlive = true;

    // Handle pong responses
    ws.on('pong', () => {
      (ws as any).isAlive = true;
    });

    // Register browser client connection
    if (!clientConnections.has(deviceId)) {
      clientConnections.set(deviceId, new Set());
    }
    clientConnections.get(deviceId)!.add(ws);
    console.log(`[WebSocket] Client connected for device: ${deviceId}`);

    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data.toString());
        
        // ============================================================
        // DO NOT REMOVE OR MODIFY - SECURITY CRITICAL
        // Validate all incoming WebSocket messages before processing
        // ============================================================
        const validation = validateClientMessage(message);
        if (!validation.valid) {
          console.error(`[WebSocket] Invalid message from client ${deviceId}:`, validation.errors);
          ws.send(JSON.stringify({
            type: 'error',
            error: 'Invalid message format',
            details: validation.errors
          }));
          return;
        }
        // ============================================================
        // END SECURITY CRITICAL SECTION
        // ============================================================
        
        console.log(`[WebSocket] Message from client for ${deviceId}:`, message.type);
        
        // Handle metrics control messages
        if (message.type === 'metrics_start') {
          console.log(`[WebSocket] Starting metrics for device: ${deviceId}`);
          const mqtt = (global as any).mqttClient;
          mqtt.publish(
            `devices/${deviceId}/commands/metrics`,
            JSON.stringify({ enabled: true }),
            { qos: 1 }
          );
        } else if (message.type === 'metrics_stop') {
          console.log(`[WebSocket] Stopping metrics for device: ${deviceId}`);
          const mqtt = (global as any).mqttClient;
          mqtt.publish(
            `devices/${deviceId}/commands/metrics`,
            JSON.stringify({ enabled: false }),
            { qos: 1 }
          );
        } else if (message.type === 'logs_start') {
          const logLevel = message.message?.level || 'WARNING';
          console.log(`[WebSocket] Starting log streaming for device: ${deviceId} at level ${logLevel}`);
          const mqtt = (global as any).mqttClient;
          mqtt.publish(
            `devices/${deviceId}/commands/logs`,
            JSON.stringify({ enabled: true, level: logLevel }),
            { qos: 1 }
          );
        } else if (message.type === 'logs_stop') {
          console.log(`[WebSocket] Stopping log streaming for device: ${deviceId}`);
          const mqtt = (global as any).mqttClient;
          mqtt.publish(
            `devices/${deviceId}/commands/logs`,
            JSON.stringify({ enabled: false }),
            { qos: 1 }
          );
        }
      } catch (error) {
        console.error('[WebSocket] Failed to parse message:', error);
        ws.send(JSON.stringify({
          type: 'error',
          error: 'Failed to parse message'
        }));
      }
    });

    ws.on('close', () => {
      const clients = clientConnections.get(deviceId);
      if (clients) {
        clients.delete(ws);
        if (clients.size === 0) {
          clientConnections.delete(deviceId);
          // Last client disconnected, stop metrics
          console.log(`[WebSocket] Last client disconnected for ${deviceId}, stopping metrics`);
          const mqtt = (global as any).mqttClient;
          mqtt.publish(
            `devices/${deviceId}/commands/metrics`,
            JSON.stringify({ enabled: false }),
            { qos: 1 }
          );
        }
      }
      console.log(`[WebSocket] Client disconnected for device: ${deviceId}`);
    });

    ws.on('error', (error) => {
      console.error('[WebSocket] Error:', error);
    });

    // Send initial connection confirmation
    ws.send(JSON.stringify({
      type: 'connected',
      message: {
        deviceId,
        timestamp: new Date().toISOString()
      }
    }));
  });

  console.log('[WebSocket] Server initialized on path /ws');
  
  return wss;
}

/**
 * Broadcast message to all browser clients watching a specific device
 */
export function broadcastToClients(deviceId: string, message: any) {
  const clients = clientConnections.get(deviceId);
  if (!clients || clients.size === 0) {
    console.log(`[WebSocket] No clients connected for device ${deviceId}`);
    return;
  }

  // ============================================================
  // DO NOT REMOVE OR MODIFY - SECURITY CRITICAL
  // Validate all outgoing WebSocket messages before broadcasting
  // ============================================================
  const validation = validateServerMessage(message);
  if (!validation.valid) {
    console.error(`[WebSocket] Invalid broadcast message for ${deviceId}:`, validation.errors);
    console.error('[WebSocket] Message:', JSON.stringify(message, null, 2));
    return;
  }
  // ============================================================
  // END SECURITY CRITICAL SECTION
  // ============================================================

  console.log(`[WebSocket] 📡 Broadcasting message type '${message.type}' to device ${deviceId}`);
  if (message.type === 'snapshot') {
    console.log(`[WebSocket] Snapshot details: ${message.message?.width}x${message.message?.height}, ${message.message?.image?.length || 0} bytes`);
  } else if (message.type === 'metrics') {
    console.log(`[WebSocket] Metrics details:`, JSON.stringify(message.message).substring(0, 200));
  } else if (message.type === 'alert') {
    console.log(`[WebSocket] Alert details: ${message.message?.description || 'unknown'}`);
  }

  const payload = JSON.stringify(message);
  let sentCount = 0;

  clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(payload);
      sentCount++;
    }
  });

  console.log(`[WebSocket] Broadcasted to ${sentCount} client(s) for device ${deviceId}`);
}
