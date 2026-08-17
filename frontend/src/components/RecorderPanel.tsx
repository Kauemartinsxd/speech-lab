import { useRef, useState } from "react";

interface Props {
  onAudio: (blob: Blob, filename: string) => void;
  desabilitado?: boolean;
}

/**
 * Gravação pelo MediaRecorder ou upload de arquivo. Qualquer que seja a
 * origem, o backend normaliza para WAV mono 16 kHz — as engines nunca veem
 * formatos diferentes.
 */
export default function RecorderPanel({ onAudio, desabilitado }: Props) {
  const [gravando, setGravando] = useState(false);
  const [erro, setErro] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  async function iniciar() {
    setErro(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType });
        const ext = recorder.mimeType.includes("ogg") ? "ogg" : "webm";
        onAudio(blob, `gravacao.${ext}`);
        stream.getTracks().forEach((t) => t.stop());
      };

      recorder.start();
      recorderRef.current = recorder;
      setGravando(true);
    } catch (e) {
      setErro(e instanceof Error ? e.message : "falha ao acessar o microfone");
    }
  }

  function parar() {
    recorderRef.current?.stop();
    recorderRef.current = null;
    setGravando(false);
  }

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <button
          className={gravando ? "botao border-rose-600 text-rose-300" : "botao"}
          onClick={gravando ? parar : iniciar}
          disabled={desabilitado}
        >
          {gravando ? "■ parar" : "● gravar"}
        </button>

        <label className="botao cursor-pointer">
          arquivo…
          <input
            type="file"
            accept="audio/*,video/*"
            className="hidden"
            disabled={desabilitado}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onAudio(file, file.name);
              e.target.value = "";
            }}
          />
        </label>
      </div>

      {gravando && (
        <div className="flex items-center gap-2 text-[11px] text-rose-400">
          <span className="h-2 w-2 animate-pulse rounded-full bg-rose-500" />
          gravando…
        </div>
      )}
      {erro && <div className="text-[11px] text-rose-400">{erro}</div>}
    </div>
  );
}
