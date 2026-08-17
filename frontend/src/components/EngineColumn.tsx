import { useState } from "react";
import type { EngineMatrixEntry, EngineResult } from "../api/types";
import { contarDivergencias, diffPalavras } from "../lib/diff";

interface Props {
  engine: EngineMatrixEntry;
  result: EngineResult | undefined;
  rodando: boolean;
  referenceText: string;
  onSeek?: (start: number, end: number) => void;
}

const CORES_OP: Record<string, string> = {
  igual: "text-bench-text",
  diferente: "bg-amber-500/25 text-amber-200 rounded px-0.5",
  extra: "bg-emerald-500/20 text-emerald-200 rounded px-0.5",
  ausente: "bg-rose-500/20 text-rose-300 line-through rounded px-0.5",
};

function Metrica({ nome, valor }: { nome: string; valor: number }) {
  const formatado =
    Math.abs(valor) >= 100 || Number.isInteger(valor) ? valor.toFixed(0) : valor.toFixed(3);
  return (
    <div className="flex justify-between gap-2 font-mono text-[11px]">
      <span className="truncate text-bench-muted">{nome}</span>
      <span className="tabular-nums">{formatado}</span>
    </div>
  );
}

export default function EngineColumn({
  engine,
  result,
  rodando,
  referenceText,
  onSeek,
}: Props) {
  const [mostrarBruto, setMostrarBruto] = useState(false);

  const normaliza = engine.capabilities?.normalizes_to_standard ?? false;
  const diff =
    result?.transcript && referenceText.trim()
      ? diffPalavras(referenceText, result.transcript)
      : null;
  const divergencias = diff ? contarDivergencias(diff) : null;

  return (
    <div className="flex min-w-[300px] flex-1 flex-col painel">
      <header className="border-b border-bench-border px-3 py-2">
        <div className="flex items-center justify-between gap-2">
          <span className="font-mono text-xs">{engine.name}</span>
          {result && (
            <span className="font-mono text-[10px] text-bench-muted">
              {result.latency_ms} ms
              {result.device ? ` · ${result.device}` : ""}
            </span>
          )}
        </div>
        <div className="mt-0.5 text-[11px] text-bench-muted">{engine.label}</div>
        {normaliza && (
          <div className="mt-1 rounded bg-amber-900/30 px-1.5 py-1 text-[10px] leading-snug text-amber-300">
            Esta engine normaliza para a forma padrão. Ela existe para demonstrar o
            problema, não para resolvê-lo.
          </div>
        )}
      </header>

      <div className="flex-1 space-y-3 p-3">
        {!result && rodando && (
          <div className="animate-pulse font-mono text-xs text-bench-muted">
            executando…
          </div>
        )}
        {!result && !rodando && (
          <div className="font-mono text-xs text-bench-muted">aguardando execução</div>
        )}

        {result?.status === "unavailable" && (
          <div className="rounded bg-amber-950/40 p-2 text-xs text-amber-300">
            indisponível — {result.unavailable_reason}
          </div>
        )}
        {(result?.status === "error" || result?.status === "timeout") && (
          <div className="rounded bg-rose-950/40 p-2 text-xs text-rose-300">
            {result.status} — {result.unavailable_reason}
          </div>
        )}

        {result?.status === "ok" && (
          <>
            <div>
              <div className="rotulo mb-1">transcrição crua</div>
              <p className="font-mono text-sm leading-relaxed">
                {diff ? (
                  diff.map((d, i) => (
                    <span key={i} className={CORES_OP[d.op]}>
                      {d.text}{" "}
                    </span>
                  ))
                ) : (
                  <span>{result.transcript || <em className="text-bench-muted">vazia</em>}</span>
                )}
              </p>
              {divergencias !== null && (
                <div className="mt-1.5 text-[11px] text-bench-muted">
                  {divergencias === 0 ? (
                    <span className="text-amber-400">
                      bate 100% com a referência — não capturou nenhum desvio
                    </span>
                  ) : (
                    <span>
                      {divergencias} palavra(s) divergindo da referência
                    </span>
                  )}
                </div>
              )}
            </div>

            {result.words.length > 0 && (
              <div>
                <div className="rotulo mb-1">palavras · confiança</div>
                <div className="flex flex-wrap gap-1">
                  {result.words.map((w, i) => (
                    <button
                      key={i}
                      onClick={() =>
                        w.start_s != null && w.end_s != null && onSeek?.(w.start_s, w.end_s)
                      }
                      title={
                        w.confidence != null
                          ? `confiança ${w.confidence.toFixed(3)}`
                          : undefined
                      }
                      className="rounded border border-bench-border px-1 py-0.5 font-mono text-[11px]
                                 hover:border-sky-600"
                      style={
                        w.confidence != null
                          ? {
                              // opacidade proporcional à confiança: palavra incerta
                              // aparece apagada
                              backgroundColor: `rgba(56,189,248,${(w.confidence * 0.28).toFixed(3)})`,
                            }
                          : undefined
                      }
                    >
                      {w.text}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {Object.keys(result.metrics).length > 0 && (
              <div>
                <div className="rotulo mb-1">métricas</div>
                <div className="space-y-0.5">
                  {Object.entries(result.metrics).map(([k, v]) => (
                    <Metrica key={k} nome={k} valor={v} />
                  ))}
                </div>
              </div>
            )}

            <div>
              <button
                className="text-[11px] text-bench-muted underline hover:text-bench-text"
                onClick={() => setMostrarBruto((v) => !v)}
              >
                {mostrarBruto ? "ocultar" : "ver"} payload bruto
              </button>
              {mostrarBruto && (
                <pre className="mt-1 max-h-64 overflow-auto rounded bg-black/40 p-2 text-[10px] leading-tight">
                  {JSON.stringify(result.raw, null, 2)}
                </pre>
              )}
            </div>
          </>
        )}
      </div>

      {result?.model_version && (
        <footer className="border-t border-bench-border px-3 py-1.5 font-mono text-[10px] text-bench-muted">
          {result.model_version}
        </footer>
      )}
    </div>
  );
}
