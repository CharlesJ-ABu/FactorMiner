import { useEffect, useState, useCallback, useRef } from 'react';

type WebSocketHookResult = {
  socket: WebSocket | null;
  isConnected: boolean;
  lastMessage: any;
  sendMessage: (msg: any) => void;
};

export function useWebSocket(
  url: string, autoReconnect: boolean = true, onMessage?: (message: any) => void,
): WebSocketHookResult {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  useEffect(() => {
    let disposed = false;
    let ownedSocket: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
    const isCurrent = (ws: WebSocket) => !disposed && ownedSocket === ws;

    const connect = () => {
      if (disposed) return;
      setIsConnected(false);
      try {
        const ws = new WebSocket(url);
        ownedSocket = ws;
        socketRef.current = ws;
        setSocket(ws);

        ws.onopen = () => {
          if (!isCurrent(ws)) return;
          setIsConnected(true);
          console.log(`[WebSocket] Connected to ${url}`);
        };

        ws.onmessage = (event) => {
          if (!isCurrent(ws)) return;
          let data: any;
          try {
            data = JSON.parse(event.data);
          } catch {
            data = event.data;
          }
          // Deliver every event even when React batches lastMessage renders.
          setLastMessage(data);
          onMessageRef.current?.(data);
        };

        ws.onclose = () => {
          if (!isCurrent(ws)) return;
          ownedSocket = null;
          socketRef.current = null;
          setSocket(null);
          setIsConnected(false);
          console.log(`[WebSocket] Disconnected from ${url}`);
          if (autoReconnect) {
            reconnectTimeout = setTimeout(() => {
              reconnectTimeout = null;
              console.log('[WebSocket] Attempting to reconnect...');
              connect();
            }, 3000);
          }
        };

        ws.onerror = (error) => {
          if (!isCurrent(ws)) return;
          console.error('[WebSocket] Error:', error);
          ws.close();
        };
      } catch (e) {
        setSocket(null);
        console.error('[WebSocket] Connection failed:', e);
      }
    };

    connect();
    return () => {
      disposed = true;
      if (reconnectTimeout !== null) clearTimeout(reconnectTimeout);
      const ws = ownedSocket;
      ownedSocket = null;
      if (socketRef.current === ws) socketRef.current = null;
      if (ws) {
        ws.onopen = ws.onmessage = ws.onclose = ws.onerror = null;
        ws.close();
      }
    };
  }, [url, autoReconnect]);

  const sendMessage = useCallback((msg: any) => {
    const ws = socketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(typeof msg === 'string' ? msg : JSON.stringify(msg));
    } else {
      console.warn('[WebSocket] Cannot send message, socket is not open');
    }
  }, []);

  return { socket, isConnected, lastMessage, sendMessage };
}
