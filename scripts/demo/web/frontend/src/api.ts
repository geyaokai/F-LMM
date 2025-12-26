import { ApiEnvelope } from "./types";

const envBase = import.meta.env.VITE_API_BASE as string | undefined;

export function defaultApiBase(): string {
  if (envBase) {
    return envBase;
  }
  const port = window.location.port;
  if (port === "5173" || port === "4173") {
    return "http://127.0.0.1:9000";
  }
  return window.location.origin;
}

function normalizeBase(base: string): string {
  if (!base) return "";
  return base.endsWith("/") ? base.slice(0, -1) : base;
}

function buildUrl(base: string, path: string): string {
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  if (base.startsWith("http")) {
    return new URL(cleanPath, normalizeBase(base) + "/").toString();
  }
  return `${normalizeBase(base)}${cleanPath}`;
}

export function withResultUrl(apiBase: string, path?: string | null): string | undefined {
  if (!path) return undefined;
  if (/^https?:\/\//i.test(path)) {
    return path;
  }
  const cleanBase = normalizeBase(apiBase);
  if (!cleanBase) {
    return path;
  }
  return `${cleanBase}${path.startsWith("/") ? "" : "/"}${path}`;
}

export async function apiFetch<T>(
  apiBase: string,
  path: string,
  init: RequestInit & { json?: unknown } = {},
): Promise<ApiEnvelope<T>> {
  const headers: Record<string, string> = {
    ...(init.headers as Record<string, string> | undefined),
  };
  let body = init.body;
  if (init.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(init.json);
  }
  const resp = await fetch(buildUrl(apiBase, path), { ...init, body, headers });
  const text = await resp.text();
  let parsed: unknown = null;
  try {
    parsed = text ? JSON.parse(text) : null;
  } catch {
    parsed = text;
  }
  if (!resp.ok) {
    const message = (parsed as any)?.message || resp.statusText;
    throw new Error(`HTTP ${resp.status}: ${message}`);
  }
  return parsed as ApiEnvelope<T>;
}
