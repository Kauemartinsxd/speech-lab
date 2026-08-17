import type { EngineMatrixEntry, EngineResult, Sample } from "./types";

const BASE = "/api";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchEngines(): Promise<EngineMatrixEntry[]> {
  const data = await json<{ engines: EngineMatrixEntry[] }>(await fetch(`${BASE}/engines`));
  return data.engines;
}

export async function fetchSamples(): Promise<Sample[]> {
  const data = await json<{ samples: Sample[] }>(await fetch(`${BASE}/samples`));
  return data.samples;
}

export async function uploadSample(params: {
  blob: Blob;
  filename: string;
  label?: string;
  referenceText?: string;
  source?: string;
}): Promise<Sample> {
  const form = new FormData();
  form.append("file", params.blob, params.filename);
  form.append("label", params.label ?? "");
  if (params.referenceText) form.append("reference_text", params.referenceText);
  form.append("source", params.source ?? "upload");
  return json<Sample>(await fetch(`${BASE}/samples`, { method: "POST", body: form }));
}

export async function createRun(params: {
  sampleId: number;
  engines?: string[];
  referenceText?: string | null;
  options?: Record<string, unknown>;
}): Promise<{ run_id: number; engines: string[] }> {
  return json(
    await fetch(`${BASE}/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_id: params.sampleId,
        engines: params.engines,
        reference_text: params.referenceText,
        options: params.options ?? {},
      }),
    }),
  );
}

export function audioUrl(sampleId: number): string {
  return `${BASE}/samples/${sampleId}/audio`;
}

export async function unloadModels(): Promise<void> {
  await fetch(`${BASE}/engines/unload`, { method: "POST" });
}

/**
 * Consome o SSE da execução. Cada engine chega assim que termina — a UI não
 * espera a mais lenta, que é justamente o ponto do comparativo.
 */
export function streamRun(
  runId: number,
  handlers: {
    onStart?: (data: { engines: string[] }) => void;
    onResult: (result: EngineResult) => void;
    onDone?: () => void;
    onError?: (message: string) => void;
  },
): () => void {
  const source = new EventSource(`${BASE}/runs/${runId}/stream`);

  source.addEventListener("start", (e) => {
    handlers.onStart?.(JSON.parse((e as MessageEvent).data));
  });
  source.addEventListener("result", (e) => {
    handlers.onResult(JSON.parse((e as MessageEvent).data) as EngineResult);
  });
  source.addEventListener("error", (e) => {
    const data = (e as MessageEvent).data;
    if (data) handlers.onError?.(JSON.parse(data).detail);
  });
  source.addEventListener("done", () => {
    handlers.onDone?.();
    source.close();
  });

  return () => source.close();
}
