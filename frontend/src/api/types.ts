// Espelha backend/app/engines/base.py. Se um lado mudar, o outro muda junto.

export type EngineStatus = "ok" | "unavailable" | "error" | "timeout";

export type DeviationType = "substituicao" | "elisao" | "epentese" | "metatese";

export type DeviationCategory =
  | "divergencia_de_decodificacao"
  | "variante_vernacular"
  | "desvio_atipico";

export interface EngineCapabilities {
  produces_orthographic: boolean;
  produces_phonemic: boolean;
  produces_word_timings: boolean;
  produces_phone_timings: boolean;
  produces_confidence: boolean;
  requires_reference_text: boolean;
  sends_audio_externally: boolean;
  is_reproducible: boolean;
  normalizes_to_standard: boolean;
}

export interface EngineMatrixEntry {
  name: string;
  label: string;
  enabled: boolean;
  implemented: boolean;
  available: boolean;
  reason: string | null;
  remedy: string | null;
  model_version?: string | null;
  capabilities: EngineCapabilities | null;
}

export interface Token {
  text: string;
  start_s: number | null;
  end_s: number | null;
  confidence: number | null;
  raw_text: string | null;
}

export interface GopScore {
  phoneme: string;
  index: number;
  start_s: number;
  end_s: number;
  gop_posterior: number;
  gop_ratio: number;
  below_threshold: boolean;
}

export interface Deviation {
  expected: string | null;
  observed: string | null;
  index_canonical: number | null;
  index_observed: number | null;
  type: DeviationType;
  category: DeviationCategory;
  process: string | null;
  word: string | null;
  start_s: number | null;
  end_s: number | null;
  gop: number | null;
  evidence: string | null;
}

export interface EngineResult {
  engine: string;
  status: EngineStatus;
  unavailable_reason: string | null;
  transcript: string | null;
  words: Token[];
  phonemes: Token[];
  canonical_phonemes: Token[];
  gop: GopScore[];
  deviations: Deviation[];
  metrics: Record<string, number>;
  raw: Record<string, unknown>;
  latency_ms: number;
  cost_estimate_usd: number | null;
  model_version: string | null;
  device: string | null;
}

export interface Sample {
  id: number;
  label: string;
  source: "recording" | "upload" | "seed_tts";
  audio_path: string | null;
  sha256: string;
  duration_s: number;
  sample_rate: number;
  reference_text: string | null;
  sintetico: boolean;
  tts_voice: string | null;
  audio_discarded_at: string | null;
  notes: string | null;
  created_at: string;
}

export const CATEGORIA_LABEL: Record<DeviationCategory, string> = {
  divergencia_de_decodificacao: "divergência de decodificação",
  variante_vernacular: "variante vernacular",
  desvio_atipico: "desvio atípico",
};

// Nenhuma destas descrições usa a palavra "erro". Variante não é erro.
export const CATEGORIA_DESCRICAO: Record<DeviationCategory, string> = {
  divergencia_de_decodificacao: "leu outra palavra, pulou ou inseriu — relevante em leitura em voz alta",
  variante_vernacular: "traço documentado do pt-BR; é variação, não desvio",
  desvio_atipico: "não bate com nenhuma variante conhecida; candidato a revisão humana",
};
