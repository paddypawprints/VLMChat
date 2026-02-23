import { WebSocketServer, WebSocket } from 'ws';
import type { Server } from 'http';
import { parse } from 'url';
import * as mediasoup from './mediasoup-server';

// Store WebSocket connections by deviceId and role
const deviceConnections = new Map<string, WebSocket>();
const clientConnections = new Map<string, Set<WebSocket>>();

interface MediasoupMessage {
  type: string;
  [key: string]: any;
}

export function setupMediasoupSignaling(httpServer: Server) {
  const wss = new WebSocketServer({ 
    server: httpServer,
    path: '/ws/media'
  });

  wss.on('connection', (ws: WebSocket, request) => {
    const { query } = parse(request.url || '', true);
    const deviceId = query.deviceId as string;
    const role = query.role as string; // 'device' or 'client'

    if (!deviceId || !role) {
      console.warn('[MediasoupSignaling] Connection rejected - missing deviceId or role');
      ws.close(1008, 'deviceId and role required');
      return;
    }

    if (role === 'device') {
      handleDeviceConnection(deviceId, ws);
    } else if (role === 'client') {
      handleClientConnection(deviceId, ws);
    } else {
      console.warn('[MediasoupSignaling] Invalid role:', role);
      ws.close(1008, 'Invalid role');
    }
  });

  console.log('[MediasoupSignaling] Signaling server initialized on path /ws/media');
  
  return wss;
}

function handleDeviceConnection(deviceId: string, ws: WebSocket) {
  if (deviceConnections.has(deviceId)) {
    console.warn(`[MediasoupSignaling] Device ${deviceId} already connected - closing old connection`);
    deviceConnections.get(deviceId)?.close();
  }

  deviceConnections.set(deviceId, ws);
  console.log(`[MediasoupSignaling] Device ${deviceId} connected`);

  ws.on('message', async (data) => {
    try {
      const message = JSON.parse(data.toString()) as MediasoupMessage;
      await handleDeviceMessage(deviceId, message, ws);
    } catch (error) {
      console.error('[MediasoupSignaling] Device message error:', error);
      ws.send(JSON.stringify({ 
        type: 'error', 
        message: error.message 
      }));
    }
  });

  ws.on('close', () => {
    deviceConnections.delete(deviceId);
    console.log(`[MediasoupSignaling] Device ${deviceId} disconnected`);
  });

  ws.send(JSON.stringify({
    type: 'device-connected',
    deviceId,
    timestamp: new Date().toISOString()
  }));
}

function handleClientConnection(deviceId: string, ws: WebSocket) {
  if (!clientConnections.has(deviceId)) {
    clientConnections.set(deviceId, new Set());
  }
  clientConnections.get(deviceId)!.add(ws);
  
  console.log(`[MediasoupSignaling] Client connected for device ${deviceId}`);

  ws.on('message', async (data) => {
    try {
      const message = JSON.parse(data.toString()) as MediasoupMessage;
      await handleClientMessage(deviceId, message, ws);
    } catch (error) {
      console.error('[MediasoupSignaling] Client message error:', error);
      ws.send(JSON.stringify({ 
        type: 'error', 
        message: error.message 
      }));
    }
  });

  ws.on('close', () => {
    const clients = clientConnections.get(deviceId);
    if (clients) {
      clients.delete(ws);
      if (clients.size === 0) {
        clientConnections.delete(deviceId);
      }
    }
    console.log(`[MediasoupSignaling] Client disconnected from device ${deviceId}`);
  });

  ws.send(JSON.stringify({
    type: 'client-connected',
    deviceId,
    timestamp: new Date().toISOString()
  }));
}

async function handleDeviceMessage(deviceId: string, message: MediasoupMessage, ws: WebSocket) {
  console.log(`[MediasoupSignaling] Device ${deviceId} message:`, message.type);

  switch (message.type) {
    case 'getRouterRtpCapabilities': {
      const rtpCapabilities = mediasoup.getRouterRtpCapabilities(deviceId);
      if (!rtpCapabilities) {
        // Create router if it doesn't exist
        await mediasoup.getOrCreateRouter(deviceId);
        const capabilities = mediasoup.getRouterRtpCapabilities(deviceId);
        ws.send(JSON.stringify({
          type: 'routerRtpCapabilities',
          rtpCapabilities: capabilities
        }));
      } else {
        ws.send(JSON.stringify({
          type: 'routerRtpCapabilities',
          rtpCapabilities
        }));
      }
      break;
    }

    case 'createProducerTransport': {
      const { transport, params } = await mediasoup.createWebRtcTransport(deviceId, 'send');
      ws.send(JSON.stringify({
        type: 'producerTransportCreated',
        transportParams: params
      }));
      break;
    }

    case 'connectProducerTransport': {
      await mediasoup.connectTransport(message.transportId, message.dtlsParameters);
      ws.send(JSON.stringify({
        type: 'producerTransportConnected'
      }));
      break;
    }

    case 'produce': {
      const { id } = await mediasoup.createProducer(
        message.transportId,
        message.kind,
        message.rtpParameters
      );
      ws.send(JSON.stringify({
        type: 'produced',
        producerId: id
      }));
      
      // Notify all clients that a new producer is available
      const clients = clientConnections.get(deviceId);
      if (clients) {
        const notification = JSON.stringify({
          type: 'newProducer',
          producerId: id,
          kind: message.kind
        });
        clients.forEach(client => {
          if (client.readyState === WebSocket.OPEN) {
            client.send(notification);
          }
        });
      }
      break;
    }

    default:
      console.warn(`[MediasoupSignaling] Unknown device message type: ${message.type}`);
  }
}

async function handleClientMessage(deviceId: string, message: MediasoupMessage, ws: WebSocket) {
  console.log(`[MediasoupSignaling] Client for ${deviceId} message:`, message.type);

  switch (message.type) {
    case 'getRouterRtpCapabilities': {
      const rtpCapabilities = mediasoup.getRouterRtpCapabilities(deviceId);
      if (!rtpCapabilities) {
        await mediasoup.getOrCreateRouter(deviceId);
        const capabilities = mediasoup.getRouterRtpCapabilities(deviceId);
        ws.send(JSON.stringify({
          type: 'routerRtpCapabilities',
          rtpCapabilities: capabilities
        }));
      } else {
        ws.send(JSON.stringify({
          type: 'routerRtpCapabilities',
          rtpCapabilities
        }));
      }
      break;
    }

    case 'createConsumerTransport': {
      const { transport, params } = await mediasoup.createWebRtcTransport(deviceId, 'recv');
      ws.send(JSON.stringify({
        type: 'consumerTransportCreated',
        transportParams: params
      }));
      break;
    }

    case 'connectConsumerTransport': {
      await mediasoup.connectTransport(message.transportId, message.dtlsParameters);
      ws.send(JSON.stringify({
        type: 'consumerTransportConnected'
      }));
      break;
    }

    case 'consume': {
      const consumer = await mediasoup.createConsumer(
        message.transportId,
        message.producerId,
        message.rtpCapabilities,
        deviceId
      );
      ws.send(JSON.stringify({
        type: 'consumed',
        consumer
      }));
      break;
    }

    default:
      console.warn(`[MediasoupSignaling] Unknown client message type: ${message.type}`);
  }
}
