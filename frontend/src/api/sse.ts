import { authHeaders } from "./client";

export interface SseMessage { id: string | null; event: string; data: unknown }

/**
 * fetch-based SSE client (EventSource cannot send auth headers). Reconnects with backoff and sends Last-Event-ID so the
 * server resumes from the last delivered cursor; the server sends a `resync` event if that cursor was pruned.
 * Returns a stop function.
 */
export function openStream(path: string, onMessage: (m: SseMessage) => void, onState: (s: "connected" | "reconnecting") => void,
                           startId: string | null = null): () => void {
  const ctl = new AbortController();
  let lastId = startId;
  let delay = 500;
  (async () => {
    while (!ctl.signal.aborted) {
      try {
        const res = await fetch(path, { signal: ctl.signal, headers: { ...authHeaders(path), Accept: "text/event-stream", ...(lastId ? { "Last-Event-ID": lastId } : {}) } });
        if (!res.ok || !res.body) throw new Error(String(res.status));
        onState("connected");
        delay = 500;
        const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
        let buf = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += value;
          let i: number;
          while ((i = buf.indexOf("\n\n")) >= 0) {
            const block = buf.slice(0, i);
            buf = buf.slice(i + 2);
            const f: Record<string, string> = {};
            for (const line of block.split("\n")) {
              const k = line.indexOf(": ");
              if (k > 0 && !line.startsWith(":")) f[line.slice(0, k)] = line.slice(k + 2);
            }
            if (f.event && f.data) {
              if (f.id) lastId = f.id;
              onMessage({ id: f.id ?? null, event: f.event, data: JSON.parse(f.data) });
            }
          }
        }
      } catch {
        if (ctl.signal.aborted) return;
      }
      onState("reconnecting");
      await new Promise((r) => setTimeout(r, delay));
      delay = Math.min(delay * 2, 8000);
    }
  })();
  return () => ctl.abort();
}
