import type { Perspective } from "./types";

let evaluatorToken = "";
let operatorToken = "";
try {
  evaluatorToken = sessionStorage.getItem("evaluatorToken") ?? "";
  operatorToken = sessionStorage.getItem("operatorToken") ?? "";
} catch { /* storage unavailable */ }

export function setToken(kind: "evaluator" | "operator", value: string): void {
  if (kind === "evaluator") evaluatorToken = value; else operatorToken = value;
  try { sessionStorage.setItem(`${kind}Token`, value); } catch { /* ignore */ }
}
export const getToken = (kind: "evaluator" | "operator"): string => (kind === "evaluator" ? evaluatorToken : operatorToken);

export class ApiError extends Error {
  constructor(public status: number, message: string, public body?: unknown) { super(message); }
}

export function authHeaders(path: string): Record<string, string> {
  const tok = path.startsWith("/api/world/evaluator") ? evaluatorToken : operatorToken;
  return tok ? { Authorization: `Bearer ${tok}` } : {};
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, { ...init, headers: { ...authHeaders(path), ...(init.headers as Record<string, string>) } });
  const text = await res.text();
  let body: unknown = text;
  try { body = JSON.parse(text); } catch { /* not json */ }
  if (!res.ok) {
    const detail = typeof body === "object" && body && "detail" in body ? String((body as { detail: unknown }).detail) : res.statusText;
    throw new ApiError(res.status, detail, body);
  }
  return body as T;
}

export const mutate = <T,>(path: string): Promise<T> =>
  api<T>(path, { method: "POST", headers: { "Idempotency-Key": crypto.randomUUID() } });

export const worldPath = (p: Perspective, episode: string, leaf: string): string =>
  `/api/world/${p}/episodes/${encodeURIComponent(episode)}/${leaf}`;
