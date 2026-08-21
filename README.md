# Leitura em voz alta — análise de qualidade da fala (alfabetização)

Ferramenta local (roda 100% no seu Mac, sem enviar áudio para fora) para:

1. mostrar uma **frase** para ser lida;
2. **gravar** a voz pelo microfone (ou enviar um arquivo);
3. transcrever de forma **fiel** — sem o modelo "consertar" a fala;
4. **comparar** com o texto certo, palavra a palavra, letra a letra e som a som;
5. dar uma **nota** e listar os desvios (ex.: *problema → probrema*: "Troca de L por R (rotacismo)").

## Rodar

```bash
./run.sh            # sobe em http://localhost:8765
```

Na primeira vez o script cria o ambiente em `~/.venvs/tts` (Python 3.11 via `uv`), instala as
dependências e o `espeak-ng` (Homebrew). Os dois modelos (~1,2 GB cada) são baixados do
Hugging Face na primeira execução e ficam em `~/.cache/huggingface`. Depois disso funciona offline.

> A pasta `Desktop` fica no iCloud Drive; por isso a venv fica **fora** dela (`~/.venvs/tts`).
> Para mudar: `TTS_VENV=/outro/lugar ./run.sh`. Para forçar CPU: `ASR_DEVICE=cpu ./run.sh`.

## Expor na internet

```bash
./expor.sh          # servidor + Cloudflare Tunnel, com senha gerada na hora
```

Imprime a URL `https://….trycloudflare.com`, o usuário (`prof`) e a senha. Precisa de
`brew install cloudflared`. Para escolher a senha: `TTS_PASSWORD=minhasenha ./expor.sh`.

Detalhes que importam:

- **HTTPS é obrigatório para o microfone** — navegador só libera `getUserMedia` em contexto
  seguro. Por isso o túnel funciona e o IP da LAN (`http://…`) não: por lá só dá para *enviar
  arquivo*, não gravar.
- O servidor continua escutando só em `127.0.0.1`; quem fala com o mundo é o túnel.
- **A senha é obrigatória ao expor.** Sem `TTS_PASSWORD` não há autenticação (modo local).
- **A URL muda** a cada reinício do túnel (é um túnel efêmero, sem conta na Cloudflare).
- O áudio passa pelos servidores da Cloudflare em trânsito — deixa de ser 100 % local.
- Seu Mac é o servidor: precisa ficar ligado e acordado, e cada análise usa a GPU dele.

Derrubar tudo: `pkill -f cloudflared; pkill -f 'uvicorn server.app'`

## Como funciona (duas camadas)

| Camada | Modelo | O que faz |
|---|---|---|
| **Letras** | `jonatasgrosman/wav2vec2-large-xlsr-53-portuguese` (CTC, decodificação greedy, **sem** modelo de linguagem) | Transcrição literal em letras. Alinhamento palavra a palavra + diff de letras + rótulo do erro (rotacismo, omissão de S final, surda/sonora, metátese…). |
| **Sons** | `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (reconhecedor de **fonemas**, sem léxico) | Reconhece os sons (IPA) e compara com a pronúncia esperada gerada pelo `espeak-ng` (pt-br) no mesmo inventário de fones. Tolera variação normal de vogal/sotaque; acusa consoante trocada/omitida. |

Por que duas? Qualquer modelo que escreve *letras* aprende algum vocabulário e, em contexto muito
previsível ("qual é o **problema** dessa frase"), tende a escrever a palavra certa mesmo ouvindo
"probrema". O reconhecedor de fonemas não tem esse viés — nos testes ele distinguiu
`problema → p r o b l e m a` de `probrema → p r o b i m a`.

### Como as duas camadas se combinam (assimétrico, de propósito)

| camada de sons | camada de letras | resultado |
|---|---|---|
| erro | qualquer | **erro confirmado** (vermelho) |
| ok / leve | erro | **suspeita** (azul tracejado) — desconta pouco e pede para ouvir a gravação |
| leve | ok / leve | desvio leve (âmbar) |

A camada de sons decide sobre pronúncia porque é a que não tem dicionário. A de letras, sozinha,
só levanta suspeita: ela erra a transcrição com frequência (num teste ouviu "xuwa colate" para
*chocolate*) e uma falha dela não pode virar acusação de erro de fala.

### Fronteira de palavra não é erro

O CTC decide sozinho onde pôr o espaço, e erra muito — em "a planta" não existe pausa nenhuma
entre o artigo e o substantivo. Por isso o alinhamento sabe **juntar e separar** palavras
(`os meninos` ouvido como `osmeninos`, `chocolate` como `choco late`) e uma vogal a mais na emenda
entre duas palavras (`a planta` → `a aplanta`) é tratada como alongamento/hesitação, não erro.
Epêntese **dentro** da palavra (`pneu` → `pineu`) continua sendo erro.

### Nota

- Cada palavra esperada vale 1 ponto: correta = 1; desvio leve (só acento/vogal) ≈ 0,85–0,9;
  suspeita = 0,85; erro (consoante trocada/omitida, palavra diferente) ≤ 0,6, proporcional à
  semelhança; omitida = 0.
- Palavras extras tiram 0,25 cada (também podem ser falha do reconhecedor).
- Também são mostrados WER, CER, nota-letras e nota-sons separadamente.

### Limitações honestas

- Modelos automáticos erram (o de letras tem ~10 % de erro em fala espontânea; a voz sintética
  do macOS confunde bastante). **Ouça a gravação** antes de concluir — o player está na tela.
- Tudo o que está marcado como **suspeita** é justamente onde as duas camadas discordaram: confie
  no seu ouvido, não na máquina.
- A troca **b/v** é um erro de alfabetização legítimo ("vaca"/"baca") e por isso não é tolerada —
  mas vozes sintéticas produzem um /b/ fricativo que o modelo ouve como /v/, gerando alarme falso.
  Com voz humana isso não acontece.
- Microfone ruim / ambiente barulhento derruba a qualidade. Fale a ~20 cm do mic.
- Frases com números ou siglas: escreva por extenso ("três", não "3").

## Estrutura

```
server/asr.py      transcrição literal (letras) + decodificação de áudio
server/phon.py     camada fonética (fones IPA, fonetização esperada, alinhamento por classes de som)
server/compare.py  alinhamento de palavras, diff de letras, rótulos de erro, métricas
server/phrases.py  banco de frases com o "foco" de cada uma
server/app.py      API FastAPI (/api/analyze, /api/phrases, /api/status) + combinação das camadas
static/index.html  interface (gravação no navegador → WAV 16 kHz → API)
run.sh             sobe tudo (local, sem senha)
expor.sh           sobe tudo + Cloudflare Tunnel, com senha (HTTP Basic)
```

API: `POST /api/analyze` (multipart: `audio`, `expected`) → JSON com `palavras[]`, `erros[]`,
`metricas`, `letras`, `sons`, `veredito`.
