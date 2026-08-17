import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  audioUrl,
  createRun,
  fetchEngines,
  fetchSamples,
  streamRun,
  unloadModels,
  uploadSample,
} from "../api/client";
import type { EngineMatrixEntry, EngineResult, Sample } from "../api/types";
import EngineColumn from "./EngineColumn";
import EngineMatrix from "./EngineMatrix";
import RecorderPanel from "./RecorderPanel";

export default function Bench() {
  const [engines, setEngines] = useState<EngineMatrixEntry[]>([]);
  const [selecionadas, setSelecionadas] = useState<Set<string>>(new Set());
  const [samples, setSamples] = useState<Sample[]>([]);
  const [sampleAtual, setSampleAtual] = useState<Sample | null>(null);
  const [referenceText, setReferenceText] = useState("");
  const [resultados, setResultados] = useState<Record<string, EngineResult>>({});
  const [rodando, setRodando] = useState(false);
  const [erro, setErro] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const pararStreamRef = useRef<(() => void) | null>(null);

  const carregar = useCallback(async () => {
    try {
      const [es, ss] = await Promise.all([fetchEngines(), fetchSamples()]);
      setEngines(es);
      setSamples(ss);
      setSelecionadas(
        new Set(es.filter((e) => e.enabled && e.implemented && e.available).map((e) => e.name)),
      );
    } catch (e) {
      setErro(e instanceof Error ? e.message : "falha ao carregar o backend");
    }
  }, []);

  useEffect(() => {
    void carregar();
    return () => pararStreamRef.current?.();
  }, [carregar]);

  const colunas = useMemo(
    () => engines.filter((e) => selecionadas.has(e.name)),
    [engines, selecionadas],
  );

  const externasSelecionadas = colunas.filter(
    (e) => e.capabilities?.sends_audio_externally,
  );

  function alternar(nome: string) {
    setSelecionadas((prev) => {
      const proximo = new Set(prev);
      if (proximo.has(nome)) proximo.delete(nome);
      else proximo.add(nome);
      return proximo;
    });
  }

  function selecionarSample(s: Sample) {
    setSampleAtual(s);
    setReferenceText(s.reference_text ?? "");
    setResultados({});
  }

  async function receberAudio(blob: Blob, filename: string) {
    setErro(null);
    try {
      const sample = await uploadSample({ blob, filename, referenceText });
      setSamples((prev) => [sample, ...prev]);
      setSampleAtual(sample);
      setResultados({});
    } catch (e) {
      setErro(e instanceof Error ? e.message : "falha ao enviar o áudio");
    }
  }

  async function executar() {
    if (!sampleAtual || colunas.length === 0) return;
    setErro(null);
    setResultados({});
    setRodando(true);

    try {
      const { run_id } = await createRun({
        sampleId: sampleAtual.id,
        engines: colunas.map((e) => e.name),
        referenceText: referenceText || null,
      });

      pararStreamRef.current = streamRun(run_id, {
        onResult: (r) => setResultados((prev) => ({ ...prev, [r.engine]: r })),
        onDone: () => setRodando(false),
        onError: (msg) => {
          setErro(msg);
          setRodando(false);
        },
      });
    } catch (e) {
      setErro(e instanceof Error ? e.message : "falha ao iniciar a execução");
      setRodando(false);
    }
  }

  /** Toca só o intervalo pedido — base do player sincronizado por fonema. */
  function tocarIntervalo(start: number, end: number) {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = start;
    void audio.play();
    const parar = () => {
      if (audio.currentTime >= end) {
        audio.pause();
        audio.removeEventListener("timeupdate", parar);
      }
    };
    audio.addEventListener("timeupdate", parar);
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-bench-border px-4 py-2">
        <div className="flex items-baseline gap-3">
          <h1 className="font-mono text-sm font-semibold">speech-lab</h1>
          <span className="text-[11px] text-bench-muted">
            qual pipeline transcreve o que a pessoa falou, em vez de consertar
          </span>
        </div>
        <button
          className="botao text-[11px]"
          onClick={() => unloadModels().then(carregar)}
          title="Libera os modelos da RAM. Útil em 16 GB entre experimentos."
        >
          liberar modelos
        </button>
      </header>

      <div className="flex min-h-0 flex-1">
        <aside className="flex w-80 shrink-0 flex-col gap-3 overflow-y-auto border-r border-bench-border p-3">
          <section className="space-y-2">
            <div className="rotulo">entrada</div>
            <RecorderPanel onAudio={receberAudio} desabilitado={rodando} />
          </section>

          <section className="space-y-1">
            <div className="rotulo">texto de referência (opcional)</div>
            <textarea
              className="campo h-20 resize-none font-mono text-xs"
              placeholder="Quando presente, habilita alinhamento forçado, GOP e métricas de leitura."
              value={referenceText}
              onChange={(e) => setReferenceText(e.target.value)}
            />
          </section>

          <section className="space-y-1">
            <div className="rotulo">amostras ({samples.length})</div>
            <div className="max-h-56 space-y-0.5 overflow-y-auto">
              {samples.map((s) => (
                <button
                  key={s.id}
                  onClick={() => selecionarSample(s)}
                  className={`w-full rounded border px-2 py-1 text-left text-[11px] ${
                    sampleAtual?.id === s.id
                      ? "border-sky-600 bg-sky-950/30"
                      : "border-bench-border hover:border-slate-500"
                  }`}
                >
                  <div className="flex items-center justify-between gap-1">
                    <span className="truncate font-mono">{s.label}</span>
                    <span className="shrink-0 text-bench-muted">
                      {s.duration_s.toFixed(1)}s
                    </span>
                  </div>
                  {s.sintetico && (
                    <span
                      className="text-[10px] text-purple-400"
                      title="Voz sintética. Valida o pipeline, não responde à pergunta sobre fala infantil."
                    >
                      sintético · {s.tts_voice}
                    </span>
                  )}
                </button>
              ))}
              {samples.length === 0 && (
                <div className="text-[11px] text-bench-muted">
                  nenhuma amostra ainda — grave, envie um arquivo ou rode{" "}
                  <code className="font-mono">make seed</code>
                </div>
              )}
            </div>
          </section>

          <section className="space-y-1">
            <div className="rotulo">engines</div>
            <EngineMatrix
              engines={engines}
              selected={selecionadas}
              onToggle={alternar}
            />
          </section>

          {externasSelecionadas.length > 0 && (
            <div className="rounded border border-red-900 bg-red-950/40 p-2 text-[11px] text-red-300">
              <strong>O áudio sairá desta máquina.</strong> As engines{" "}
              {externasSelecionadas.map((e) => e.name).join(", ")} enviam o áudio para
              um serviço externo. Áudio de criança é dado biométrico sob a LGPD.
            </div>
          )}

          <button
            className="botao-primario"
            onClick={executar}
            disabled={!sampleAtual || colunas.length === 0 || rodando}
          >
            {rodando ? "executando…" : `executar ${colunas.length} engine(s)`}
          </button>

          {erro && (
            <div className="rounded bg-rose-950/50 p-2 text-[11px] text-rose-300">{erro}</div>
          )}
        </aside>

        <main className="flex min-w-0 flex-1 flex-col">
          {sampleAtual && (
            <div className="flex items-center gap-3 border-b border-bench-border px-4 py-2">
              <audio
                ref={audioRef}
                src={audioUrl(sampleAtual.id)}
                controls
                className="h-8"
              />
              <div className="font-mono text-[11px] text-bench-muted">
                {sampleAtual.duration_s.toFixed(2)}s · {sampleAtual.sample_rate} Hz · mono ·
                sha {sampleAtual.sha256.slice(0, 12)}
              </div>
            </div>
          )}

          <div className="min-h-0 flex-1 overflow-auto p-3">
            {colunas.length === 0 ? (
              <div className="text-sm text-bench-muted">
                selecione ao menos uma engine
              </div>
            ) : (
              <>
                {referenceText.trim() && (
                  <div className="mb-3 rounded border border-bench-border bg-black/20 px-3 py-2 text-[11px] leading-relaxed text-bench-muted">
                    <span className="text-amber-400">Como ler o destaque:</span> bater
                    100% com a referência não é acerto. Se a fala trouxe um desvio e a
                    engine escreveu a forma padrão, ela concorda com o texto porque
                    apagou o dado. Divergir da referência é o comportamento esperado de
                    quem preservou a fala.
                  </div>
                )}
                <div className="flex flex-wrap gap-3">
                  {colunas.map((e) => (
                    <EngineColumn
                      key={e.name}
                      engine={e}
                      result={resultados[e.name]}
                      rodando={rodando}
                      referenceText={referenceText}
                      onSeek={tocarIntervalo}
                    />
                  ))}
                </div>
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
