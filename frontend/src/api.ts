/** In development, Vite proxy: /api -> backend. In production, use VITE_API_BASE_URL. */
export function apiBase(): string {
  const env = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "");
  if (env) return env;
  if (import.meta.env.DEV) return "/api";
  return "http://127.0.0.1:8001";
}

/** C# gateway ClientApiKey compatibility. Automatically uses dev key in development. */
function gatewayHeaders(): Record<string, string> {
  const key =
    import.meta.env.VITE_GATEWAY_API_KEY?.trim() ||
    (import.meta.env.DEV ? "dev-client-key" : "");
  if (!key) return {};
  return { "x-api-key": key };
}

async function parseError(res: Response): Promise<string> {
  const t = await res.text();
  try {
    const j = JSON.parse(t) as { detail?: unknown };
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) return JSON.stringify(j.detail);
  } catch {
    /* response is not JSON */
  }
  return t || res.statusText;
}

export async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${apiBase()}${path.startsWith("/") ? path : `/${path}`}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      ...gatewayHeaders(),
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<T>;
}

export async function apiPostBlob(path: string, body: unknown): Promise<Blob> {
  const url = `${apiBase()}${path.startsWith("/") ? path : `/${path}`}`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      ...gatewayHeaders(),
      "Content-Type": "application/json",
      Accept: "image/png",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.blob();
}

export async function apiUploadBatchCsv(
  file: File,
  opts?: { topicTitle?: string; keywordsSubtitle?: string },
): Promise<BatchUploadResponse> {
  const url = `${apiBase()}${"/predict/batch/upload"}`;
  const form = new FormData();
  form.append("file", file);
  if (opts?.topicTitle?.trim()) form.append("topic_title", opts.topicTitle.trim());
  if (opts?.keywordsSubtitle?.trim()) form.append("keywords_subtitle", opts.keywordsSubtitle.trim());
  const res = await fetch(url, { method: "POST", headers: gatewayHeaders(), body: form });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<BatchUploadResponse>;
}

export type HealthResponse = {
  status: string;
  model_loaded: boolean;
  detail?: string | null;
};

export type MetaResponse = {
  model_name: string;
  class_names: string[];
  confidence_fallback_enabled: boolean;
  confidence_threshold: number;
  confidence_fallback_label: string;
  max_batch_items: number;
  max_visualize_texts: number;
  max_batch_upload_bytes: number;
};

export type PredictResponse = {
  sentiment: string;
  raw_sentiment: string;
  confidence: number;
  fallback_applied: boolean;
  probabilities: Record<string, number>;
};

export type BatchPredictionRow = {
  id?: string | number;
  text: string;
  sentiment: string;
  raw_sentiment: string;
  fallback_applied: boolean;
  confidence: number;
};

export type DistributionStatsResponse = {
  counts: Record<string, number>;
  total: number;
  topic_title: string;
  keywords_subtitle: string;
  source: string;
};

export type BatchUploadResponse = {
  predictions: BatchPredictionRow[];
  counts: Record<string, number>;
  total: number;
  topic_title: string;
  keywords_subtitle: string;
  source: string;
};

