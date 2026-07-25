import { useEffect, useState, useCallback, useRef } from 'react';

type WebSocketHookResult = {
  socket: WebSocket | null;
  isConnected: boolean;
  lastMessage: any;
  sendMessage: (msg: any) => void;
};

export function useWebSocket(url: string, autoReconnect: boolean = true): WebSocketHookResult {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setIsConnected(true);
        console.log(`[WebSocket] Connected to ${url}`);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
        } catch {
          setLastMessage(event.data);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log(`[WebSocket] Disconnected from ${url}`);
        if (autoReconnect) {
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log('[WebSocket] Attempting to reconnect...');
            connect();
          }, 3000);
        }
      };

      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        ws.close();
      };

      setSocket(ws);
    } catch (e) {
      console.error('[WebSocket] Connection failed:', e);
    }
  }, [url, autoReconnect]);

  useEffect(() => {
    connect();
    return () => {
      if (socket) {
        socket.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connect]);

  const sendMessage = useCallback((msg: any) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(typeof msg === 'string' ? msg : JSON.stringify(msg));
    } else {
      console.warn('[WebSocket] Cannot send message, socket is not open');
    }
  }, [socket]);

  return { socket, isConnected, lastMessage, sendMessage };
}
