import { useEffect, useRef, useState, useCallback } from 'react';
import { validateServerMessage } from '@/lib/websocket-validator';

interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

interface LogEntry {
  timestamp: string;
  level: string;
  logger: string;
  message: string;
  module?: string;
  line?: number;
  thread?: string;
}

interface Alert {
  type: string;
  timestamp: string;
  watchlist_item_id: string;
  description: string;
  confidence: number;
  image?: string;
  image_url?: string;
  metadata?: any;
}

interface UseWebSocketOptions {
  deviceId: string;
  onSnapshot?: (data: { image: string; timestamp: string; width?: number; height?: number; format?: string }) => void;
  onMetrics?: (data: any) => void;
  onLogs?: (data: LogEntry) => void;
  onAlert?: (data: Alert) => void;
  onMessage?: (message: WebSocketMessage) => void;
  enabled?: boolean;
}

export function useWebSocket({ deviceId, onSnapshot, onMetrics, onLogs, onAlert, onMessage, enabled = true }: UseWebSocketOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout>();

  const send = useCallback((message: any) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
      console.log('[WebSocket] Sent:', message.type);
    } else {
      console.warn('[WebSocket] Cannot send, not connected');
    }
  }, []);

  useEffect(() => {
    if (!enabled || !deviceId) return;

    console.log('[WebSocket] useEffect running - deviceId:', deviceId, 'enabled:', enabled);

    const connect = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws?deviceId=${deviceId}`;

      console.log('[WebSocket] Connecting to:', wsUrl);
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        console.log('[WebSocket] Connected');
        setIsConnected(true);
        
        // Request metrics to start when connected
        if (ws.current) {
          ws.current.send(JSON.stringify({ type: 'metrics_start', message: {} }));
          console.log('[WebSocket] Sent metrics_start command');
        }
      };

      ws.current.onmessage = (event) => {
        console.log('[WebSocket] 📨 Raw message received:', event.data.substring(0, 200));
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;
          console.log('[WebSocket] 📦 Parsed message type:', message.type);
          
          // ============================================================
          // DO NOT REMOVE OR MODIFY - SECURITY CRITICAL
          // Validate all incoming WebSocket messages before processing
          // ============================================================
          const validation = validateServerMessage(message);
          if (!validation.valid) {
            console.error('[WebSocket] ❌ Invalid message from server:', {
              type: message.type,
              errors: validation.errors
            });
            console.error('[WebSocket] Validation errors details:', JSON.stringify(validation.errors, null, 2));
            console.error('[WebSocket] Message:', JSON.stringify(message, null, 2));
            // Don't process invalid messages - security violation
            return;
          }
          // ============================================================
          // END SECURITY CRITICAL SECTION
          // ============================================================
          
          console.log('[WebSocket] ✅ Valid message received:', message.type);

          if (message.type === 'snapshot') {
            console.log('[WebSocket] 📸 Snapshot message - calling onSnapshot callback');
            if (onSnapshot) {
              onSnapshot(message.message);
            }
          } else if (message.type === 'metrics') {
            console.log('[WebSocket] 📊 Metrics message received:', JSON.stringify(message.message).substring(0, 200));
            if (onMetrics) {
              console.log('[WebSocket] Calling onMetrics callback');
              onMetrics(message.message);
            } else {
              console.warn('[WebSocket] onMetrics callback not defined!');
            }
          } else if (message.type === 'logs') {
            console.log('[WebSocket] 📝 Logs message received');
            if (onLogs) {
              onLogs(message.message);
            }
          } else if (message.type === 'alert') {
            console.log('[WebSocket] 🚨 Alert message received:', JSON.stringify(message.message).substring(0, 200));
            if (onAlert) {
              console.log('[WebSocket] Calling onAlert callback');
              onAlert(message.message);
            } else {
              console.warn('[WebSocket] onAlert callback not defined!');
            }
          }

          if (onMessage) {
            onMessage(message);
          }
        } catch (error) {
          console.error('[WebSocket] Failed to parse message:', error);
        }
      };

      ws.current.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
      };

      ws.current.onclose = () => {
        console.log('[WebSocket] Disconnected');
        setIsConnected(false);

        // Attempt to reconnect after 3 seconds
        reconnectTimeout.current = setTimeout(() => {
          console.log('[WebSocket] Attempting to reconnect...');
          connect();
        }, 3000);
      };
    };

    connect();

    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      if (ws.current) {
        console.log('[WebSocket] Closing connection on cleanup');
        // Send metrics_stop before closing
        if (ws.current.readyState === WebSocket.OPEN) {
          ws.current.send(JSON.stringify({ type: 'metrics_stop', message: {} }));
          console.log('[WebSocket] Sent metrics_stop command');
        }
        ws.current.close();
        ws.current = null;
      }
    };
    // Only reconnect when deviceId or enabled changes, not when callbacks change
  }, [deviceId, enabled]);

  return { isConnected, send };
}
