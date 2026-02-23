import { WebSocketServer, WebSocket } from 'ws';
import type { Server } from 'http';
import { parse } from 'url';

// Store video streaming connections
const deviceStreamers = new Map<string, WebSocket>();
const clientViewers = new Map<string, Set<WebSocket>>();

interface VideoFrame {
  type: 'video-frame';
  deviceId: string;
  frame: string; // Base64 encoded JPEG
  timestamp: string;
  format?: string;
  width?: number;
  height?: number;
  frameNumber?: number;
}

interface StreamControl {
  type: 'start-stream' | 'stop-stream';
  deviceId: string;
}

export function setupVideoProxy(httpServer: Server) {
  const wss = new WebSocketServer({ 
    server: httpServer,
    path: '/ws/video',
    // Increase max payload for video frames (5MB)
    maxPayload: 5 * 1024 * 1024
  });

  wss.on('connection', (ws: WebSocket, request) => {
    const { query } = parse(request.url || '', true);
    const deviceId = query.deviceId as string;
    const role = query.role as string; // 'device' or 'client'

    if (!deviceId || !role) {
      console.warn('[VideoProxy] Connection rejected - missing deviceId or role');
      ws.close(1008, 'deviceId and role required');
      return;
    }

    if (role === 'device') {
      handleDeviceConnection(deviceId, ws);
    } else if (role === 'client') {
      handleClientConnection(deviceId, ws);
    } else {
      console.warn('[VideoProxy] Invalid role:', role);
      ws.close(1008, 'Invalid role');
    }
  });

  console.log('[VideoProxy] Video streaming server initialized on path /ws/video');
  
  return wss;
}

function handleDeviceConnection(deviceId: string, ws: WebSocket) {
  // Only one device can stream at a time
  if (deviceStreamers.has(deviceId)) {
    console.warn(`[VideoProxy] Device ${deviceId} already streaming - closing old connection`);
    deviceStreamers.get(deviceId)?.close();
  }

  deviceStreamers.set(deviceId, ws);
  console.log(`[VideoProxy] Device ${deviceId} connected for streaming`);

  let frameCount = 0;
  let lastLogTime = Date.now();
  let framesPerSecond = 0;

  ws.on('message', (data) => {
    try {
      const message = JSON.parse(data.toString()) as VideoFrame;

      if (message.type === 'video-frame') {
        frameCount++;
        framesPerSecond++;

        // Broadcast frame to all clients watching this device
        const viewers = clientViewers.get(deviceId);
        if (viewers && viewers.size > 0) {
          const payload = JSON.stringify(message);
          let sentCount = 0;

          viewers.forEach((viewer) => {
            if (viewer.readyState === WebSocket.OPEN) {
              viewer.send(payload);
              sentCount++;
            }
          });

          // Log stats every 5 seconds
          const now = Date.now();
          if (now - lastLogTime > 5000) {
            const fps = framesPerSecond / 5;
            console.log(`[VideoProxy] ${deviceId}: ${fps.toFixed(1)} fps → ${sentCount} viewer(s)`);
            framesPerSecond = 0;
            lastLogTime = now;
          }
        }
      }
    } catch (error) {
      console.error('[VideoProxy] Failed to parse device message:', error);
    }
  });

  ws.on('close', () => {
    deviceStreamers.delete(deviceId);
    console.log(`[VideoProxy] Device ${deviceId} disconnected (sent ${frameCount} frames)`);
    
    // Notify clients that stream ended
    const viewers = clientViewers.get(deviceId);
    if (viewers) {
      viewers.forEach((viewer) => {
        if (viewer.readyState === WebSocket.OPEN) {
          viewer.send(JSON.stringify({ type: 'stream-ended', deviceId }));
        }
      });
    }
  });

  ws.on('error', (error) => {
    console.error(`[VideoProxy] Device ${deviceId} error:`, error);
  });

  // Send confirmation
  ws.send(JSON.stringify({
    type: 'stream-ready',
    deviceId,
    timestamp: new Date().toISOString()
  }));
}

function handleClientConnection(deviceId: string, ws: WebSocket) {
  // Add client to viewers for this device
  if (!clientViewers.has(deviceId)) {
    clientViewers.set(deviceId, new Set());
  }
  clientViewers.get(deviceId)!.add(ws);
  
  const viewerCount = clientViewers.get(deviceId)!.size;
  console.log(`[VideoProxy] Client connected to watch ${deviceId} (${viewerCount} total viewers)`);

  ws.on('message', (data) => {
    try {
      const message = JSON.parse(data.toString()) as StreamControl;

      if (message.type === 'start-stream') {
        // Request device to start streaming
        const deviceWs = deviceStreamers.get(deviceId);
        if (deviceWs && deviceWs.readyState === WebSocket.OPEN) {
          deviceWs.send(JSON.stringify({ type: 'start-streaming' }));
          console.log(`[VideoProxy] Requested ${deviceId} to start streaming`);
        } else {
          ws.send(JSON.stringify({ 
            type: 'error', 
            message: 'Device not connected' 
          }));
        }
      } else if (message.type === 'stop-stream') {
        const deviceWs = deviceStreamers.get(deviceId);
        if (deviceWs && deviceWs.readyState === WebSocket.OPEN) {
          deviceWs.send(JSON.stringify({ type: 'stop-streaming' }));
          console.log(`[VideoProxy] Requested ${deviceId} to stop streaming`);
        }
      }
    } catch (error) {
      console.error('[VideoProxy] Failed to parse client message:', error);
    }
  });

  ws.on('close', () => {
    const viewers = clientViewers.get(deviceId);
    if (viewers) {
      viewers.delete(ws);
      if (viewers.size === 0) {
        clientViewers.delete(deviceId);
        
        // No more viewers - tell device to stop streaming
        const deviceWs = deviceStreamers.get(deviceId);
        if (deviceWs && deviceWs.readyState === WebSocket.OPEN) {
          deviceWs.send(JSON.stringify({ type: 'stop-streaming' }));
          console.log(`[VideoProxy] No more viewers for ${deviceId}, stopping stream`);
        }
      }
      console.log(`[VideoProxy] Client disconnected from ${deviceId} (${viewers.size} viewers remain)`);
    }
  });

  ws.on('error', (error) => {
    console.error('[VideoProxy] Client error:', error);
  });

  // Send connection confirmation
  ws.send(JSON.stringify({
    type: 'viewer-connected',
    deviceId,
    isStreaming: deviceStreamers.has(deviceId),
    timestamp: new Date().toISOString()
  }));
}

/**
 * Get streaming stats for monitoring
 */
export function getStreamingStats() {
  const stats = {
    activeStreamers: deviceStreamers.size,
    devices: [] as any[]
  };

  deviceStreamers.forEach((ws, deviceId) => {
    const viewerCount = clientViewers.get(deviceId)?.size || 0;
    stats.devices.push({
      deviceId,
      connected: ws.readyState === WebSocket.OPEN,
      viewers: viewerCount
    });
  });

  return stats;
}
