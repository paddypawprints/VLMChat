import { useRef, useState, useCallback, useEffect } from 'react';

interface UseWebRTCOptions {
  onTrack?: (stream: MediaStream) => void;
  onConnectionStateChange?: (state: RTCPeerConnectionState) => void;
  send: (message: any) => void;
  onMessage: (handler: (message: any) => void) => void;
}

export function useWebRTC({ onTrack, onConnectionStateChange, send, onMessage }: UseWebRTCOptions) {
  const [connectionState, setConnectionState] = useState<RTCPeerConnectionState>('new');
  const [isStarting, setIsStarting] = useState(false);
  const peerConnection = useRef<RTCPeerConnection | null>(null);

  const startLiveStream = useCallback(async () => {
    if (peerConnection.current) {
      console.log('[WebRTC] Connection already exists');
      return;
    }

    setIsStarting(true);

    try {
      // Create peer connection
      const pc = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' }
        ]
      });

      peerConnection.current = pc;

      // Handle connection state changes
      pc.onconnectionstatechange = () => {
        console.log('[WebRTC] Connection state:', pc.connectionState);
        setConnectionState(pc.connectionState);
        onConnectionStateChange?.(pc.connectionState);

        if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
          cleanup();
        }
      };

      // Handle ICE candidates
      pc.onicecandidate = (event) => {
        if (event.candidate) {
          console.log('[WebRTC] Sending ICE candidate');
          send({
            type: 'ice-candidate',
            candidate: event.candidate.toJSON()
          });
        }
      };

      // Handle incoming tracks (video stream from device)
      pc.ontrack = (event) => {
        console.log('[WebRTC] Received track:', event.track.kind);
        onTrack?.(event.streams[0]);
      };

      // Add transceiver to receive video
      pc.addTransceiver('video', { direction: 'recvonly' });

      // Create and send offer
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      console.log('[WebRTC] Sending offer');
      send({
        type: 'webrtc-offer',
        sdp: offer.sdp
      });

      setIsStarting(false);
    } catch (error) {
      console.error('[WebRTC] Failed to start:', error);
      setIsStarting(false);
      cleanup();
    }
  }, [send, onTrack, onConnectionStateChange]);

  const stopLiveStream = useCallback(() => {
    cleanup();
  }, []);

  const cleanup = useCallback(() => {
    if (peerConnection.current) {
      peerConnection.current.close();
      peerConnection.current = null;
      setConnectionState('closed');
    }
  }, []);

  // Handle incoming WebRTC messages
  useEffect(() => {
    const handleMessage = async (message: any) => {
      if (!peerConnection.current) return;

      try {
        if (message.type === 'webrtc-answer') {
          console.log('[WebRTC] Received answer');
          await peerConnection.current.setRemoteDescription({
            type: 'answer',
            sdp: message.sdp
          });
        } else if (message.type === 'ice-candidate' && message.candidate) {
          console.log('[WebRTC] Received ICE candidate');
          await peerConnection.current.addIceCandidate(new RTCIceCandidate(message.candidate));
        }
      } catch (error) {
        console.error('[WebRTC] Error handling message:', error);
      }
    };

    onMessage(handleMessage);
  }, [onMessage]);

  return {
    startLiveStream,
    stopLiveStream,
    connectionState,
    isStarting
  };
}
