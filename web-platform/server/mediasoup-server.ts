import * as mediasoup from 'mediasoup';
import { types as mediasoupTypes } from 'mediasoup';

let worker: mediasoupTypes.Worker;
const routers = new Map<string, mediasoupTypes.Router>();
const transports = new Map<string, mediasoupTypes.WebRtcTransport>();
const producers = new Map<string, mediasoupTypes.Producer>();
const consumers = new Map<string, mediasoupTypes.Consumer>();

// RTP capabilities for the router
const mediaCodecs: mediasoupTypes.RtpCodecCapability[] = [
  {
    kind: 'video',
    mimeType: 'video/VP8',
    clockRate: 90000,
    parameters: {
      'x-google-start-bitrate': 1000
    }
  },
  {
    kind: 'video',
    mimeType: 'video/H264',
    clockRate: 90000,
    parameters: {
      'packetization-mode': 1,
      'profile-level-id': '42e01f',
      'level-asymmetry-allowed': 1
    }
  }
];

export async function initMediasoup() {
  try {
    // Create mediasoup worker
    worker = await mediasoup.createWorker({
      logLevel: 'warn',
      rtcMinPort: 40000,
      rtcMaxPort: 49999,
    });

    worker.on('died', () => {
      console.error('[Mediasoup] Worker died, exiting in 2s...');
      setTimeout(() => process.exit(1), 2000);
    });

    console.log('[Mediasoup] Worker created (PID:', worker.pid, ')');
    
    return worker;
  } catch (error) {
    console.error('[Mediasoup] Failed to create worker:', error);
    throw error;
  }
}

export async function getOrCreateRouter(deviceId: string): Promise<mediasoupTypes.Router> {
  if (routers.has(deviceId)) {
    return routers.get(deviceId)!;
  }

  if (!worker) {
    throw new Error('Mediasoup worker not initialized');
  }

  const router = await worker.createRouter({ mediaCodecs });
  routers.set(deviceId, router);
  
  console.log('[Mediasoup] Created router for device:', deviceId);
  return router;
}

export async function createWebRtcTransport(
  deviceId: string,
  direction: 'send' | 'recv'
): Promise<{
  transport: mediasoupTypes.WebRtcTransport;
  params: {
    id: string;
    iceParameters: mediasoupTypes.IceParameters;
    iceCandidates: mediasoupTypes.IceCandidate[];
    dtlsParameters: mediasoupTypes.DtlsParameters;
  };
}> {
  const router = await getOrCreateRouter(deviceId);

  // Use environment variables or defaults for WebRTC transport
  const webRtcServerOptions: mediasoupTypes.WebRtcTransportOptions = {
    listenIps: [
      {
        ip: process.env.MEDIASOUP_LISTEN_IP || '0.0.0.0',
        announcedIp: process.env.MEDIASOUP_ANNOUNCED_IP || undefined,
      }
    ],
    enableUdp: true,
    enableTcp: true,
    preferUdp: true,
  };

  const transport = await router.createWebRtcTransport(webRtcServerOptions);

  const transportId = `${deviceId}-${direction}-${Date.now()}`;
  transports.set(transportId, transport);

  console.log(`[Mediasoup] Created ${direction} transport for ${deviceId}:`, transportId);

  return {
    transport,
    params: {
      id: transport.id,
      iceParameters: transport.iceParameters,
      iceCandidates: transport.iceCandidates,
      dtlsParameters: transport.dtlsParameters,
    },
  };
}

export async function connectTransport(
  transportId: string,
  dtlsParameters: mediasoupTypes.DtlsParameters
): Promise<void> {
  const transport = transports.get(transportId);
  if (!transport) {
    throw new Error(`Transport ${transportId} not found`);
  }

  await transport.connect({ dtlsParameters });
  console.log('[Mediasoup] Transport connected:', transportId);
}

export async function createProducer(
  transportId: string,
  kind: mediasoupTypes.MediaKind,
  rtpParameters: mediasoupTypes.RtpParameters
): Promise<{ id: string }> {
  const transport = transports.get(transportId);
  if (!transport) {
    throw new Error(`Transport ${transportId} not found`);
  }

  const producer = await transport.produce({
    kind,
    rtpParameters,
  });

  const producerId = `${transportId}-producer-${Date.now()}`;
  producers.set(producerId, producer);

  console.log('[Mediasoup] Created producer:', producerId, kind);

  return { id: producer.id };
}

export async function createConsumer(
  transportId: string,
  producerId: string,
  rtpCapabilities: mediasoupTypes.RtpCapabilities,
  deviceId: string
): Promise<{
  id: string;
  kind: mediasoupTypes.MediaKind;
  rtpParameters: mediasoupTypes.RtpParameters;
  producerId: string;
}> {
  const transport = transports.get(transportId);
  if (!transport) {
    throw new Error(`Transport ${transportId} not found`);
  }

  const producer = producers.get(producerId);
  if (!producer) {
    throw new Error(`Producer ${producerId} not found`);
  }

  const router = await getOrCreateRouter(deviceId);

  const consumer = await transport.consume({
    producerId: producer.id,
    rtpCapabilities,
    paused: false,
  });

  const consumerId = `${transportId}-consumer-${Date.now()}`;
  consumers.set(consumerId, consumer);

  console.log('[Mediasoup] Created consumer:', consumerId);

  return {
    id: consumer.id,
    kind: consumer.kind,
    rtpParameters: consumer.rtpParameters,
    producerId: producer.id,
  };
}

export function getRouterRtpCapabilities(deviceId: string): mediasoupTypes.RtpCapabilities | null {
  const router = routers.get(deviceId);
  return router ? router.rtpCapabilities : null;
}

export function cleanup() {
  transports.forEach(t => t.close());
  routers.forEach(r => r.close());
  transports.clear();
  routers.clear();
  producers.clear();
  consumers.clear();
}
