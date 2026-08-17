import type { EngineMatrixEntry } from "../api/types";

interface Props {
  engines: EngineMatrixEntry[];
  selected: Set<string>;
  onToggle: (name: string) => void;
}

/**
 * Matriz de engines. Tudo aqui vem de `capabilities` do backend — nenhuma regra
 * por engine está codificada neste arquivo.
 */
export default function EngineMatrix({ engines, selected, onToggle }: Props) {
  return (
    <div className="space-y-1">
      {engines.map((e) => {
        const selecionavel = e.implemented && e.available;
        const externa = e.capabilities?.sends_audio_externally ?? false;
        const normaliza = e.capabilities?.normalizes_to_standard ?? false;
        const qualitativa = e.capabilities ? !e.capabilities.is_reproducible : false;

        return (
          <label
            key={e.name}
            className={`flex items-start gap-2 rounded border px-2 py-1.5 text-sm ${
              selecionavel
                ? "cursor-pointer border-bench-border hover:border-slate-500"
                : "cursor-not-allowed border-transparent opacity-45"
            }`}
          >
            <input
              type="checkbox"
              className="mt-1 accent-sky-600"
              disabled={!selecionavel}
              checked={selected.has(e.name)}
              onChange={() => onToggle(e.name)}
            />
            <span className="min-w-0 flex-1">
              <span className="flex flex-wrap items-center gap-1.5">
                <span className="font-mono text-xs">{e.name}</span>
                {normaliza && (
                  <span
                    className="rounded bg-amber-900/50 px-1 text-[10px] text-amber-300"
                    title="Decodificação autorregressiva: 'conserta' a fala para a forma padrão"
                  >
                    normaliza
                  </span>
                )}
                {qualitativa && (
                  <span
                    className="rounded bg-purple-900/50 px-1 text-[10px] text-purple-300"
                    title="Saída de LLM: opina, não produz métrica reproduzível"
                  >
                    qualitativa
                  </span>
                )}
                {externa && (
                  <span
                    className="rounded bg-red-900/60 px-1 text-[10px] text-red-300"
                    title="Esta engine ENVIA O ÁUDIO para um serviço externo"
                  >
                    áudio sai da máquina
                  </span>
                )}
              </span>
              <span className="block text-[11px] text-bench-muted">{e.label}</span>
              {e.reason && (
                <span className="block text-[11px] text-amber-500/80">
                  {e.reason}
                  {e.remedy ? ` — ${e.remedy}` : ""}
                </span>
              )}
            </span>
          </label>
        );
      })}
    </div>
  );
}
