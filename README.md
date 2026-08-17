# speech-lab

Bancada de experimentação para comparar abordagens de análise de fala em português
brasileiro, com foco em detecção de desvios de pronúncia em fala infantil e adolescente.

Não é produto. É instrumento de medida.

## A pergunta

> Qual pipeline consegue transcrever **o que a pessoa realmente falou**, em vez de
> "consertar" para a forma padrão?

O caso-teste canônico: quando o aluno fala **"pobrema"**, o Whisper transcreve
**"problema"**. Não é bug — o decoder autorregressivo do Whisper funciona como modelo
de linguagem e normaliza a saída para a forma mais provável do idioma. Para um lab que
quer *medir o desvio*, essa normalização destrói justamente o dado de interesse.

O lab mede quais engines preservam o desvio e quais não preservam.

## Arquitetura

```
  áudio (MediaRecorder ou upload)
        │
        ▼
  ffmpeg → WAV mono 16 kHz PCM s16le          ← normalização única, todas as engines
        │                                        recebem exatamente o mesmo sinal
        ├──────────┬──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼
  whisper_    whisper_    ctc_       phoneme_   azure_pa    gemini_
  baseline    strict      greedy       gop                   audio
        │          │          │          │          │          │
        └──────────┴──────────┴──────────┴──────────┴──────────┘
                              │
                              ▼
              camada de classificação de variação           ← roda DEPOIS de todas,
              (levenshtein + variantes_ptbr.yaml)             nunca dentro da engine
                              │
                              ▼
        divergencia_de_        variante_          desvio_
        decodificacao          vernacular         atipico
```

Cada engine implementa a mesma interface (`backend/app/engines/base.py`) e é totalmente
opcional: se o modelo não foi baixado ou a chave de API não existe, a engine reporta
`unavailable` com o motivo e o comando que resolve — sem derrubar as outras.

**Engine mede, classificador julga.** Nenhuma engine preenche `deviations`; quem
classifica é a camada de variação, com regras versionadas e auditáveis.

## Matriz de engines

| engine | papel | ortográfico | fonêmico | tempo/fonema | reprodutível | áudio sai da máquina |
|---|---|:-:|:-:|:-:|:-:|:-:|
| `whisper_baseline` | **controle negativo** — existe para demonstrar o problema | ✓ | — | — | ✓ | não |
| `whisper_strict` | variante sem condicionamento, `temperature=0` | ✓ | — | — | ✓ | não |
| `ctc_greedy` | Caminho 1 — decodificação greedy, sem LM | ✓ | — | — | ✓ | não |
| `phoneme_gop` | Caminho 2 — **o principal**: G2P → fonemas livres → alinhamento → GOP | — | ✓ | ✓ | ✓ | não |
| `azure_pa` | Caminho 3 — baseline comercial | ✓ | ✓ | ✓ | ✓ | **sim** |
| `gemini_audio` | comparação de arquitetura — **opina, não mede** | ✓ | — | — | **não** | **sim** |

### Por que o Whisper normaliza e o CTC não

O Whisper decodifica autorregressivamente: cada token é condicionado nos anteriores, o
que embute um modelo de linguagem. "pobrema" tem probabilidade baixíssima em pt-BR;
"problema" tem alta. O decoder escolhe a alta.

O `ctc_greedy` decodifica quadro a quadro, tomando o argmax de cada frame de forma
independente, **sem beam search e sem KenLM**. Não há nada no caminho que puxe a saída
para a forma padrão — daí a expectativa de que escreva `pobrema`.

É por isso que a decodificação greedy é obrigatória aqui. Ligar beam search com LM
reintroduziria exatamente o viés que o lab quer medir.

## As três categorias

Nada na interface chama variante de erro. Todo desvio detectado sai rotulado em uma de
três categorias, **disjuntas por construção**:

| categoria | significado | camada que decide |
|---|---|---|
| `divergencia_de_decodificacao` | leu **outra palavra**, pulou ou inseriu uma — miscue de leitura | palavra |
| `variante_vernacular` | traço documentado do pt-BR; **não é erro, é variação** | fonema + `variantes_ptbr.yaml` |
| `desvio_atipico` | palavra certa, realização fonética sem variante documentada — candidato a revisão humana | fonema |

A precedência (palavra → variante → atípico) é o que impede que as categorias se
sobreponham. Sem texto de referência, a primeira não existe.

### Metátese não é transposição adjacente

Verificado com o espeak-ng 1.52 instalado:

```
problema → prˌoblˈemæ  →  p r o b l e m æ
pobrema  → pˌobrˈemæ   →  p o b r e m æ
```

O `/r/` sai do índice 1 e reaparece no índice 3. A transposição adjacente do
Damerau-Levenshtein **nunca dispara**, e o DP clássico devolve "elisão de /r/ +
substituição l→r" — classificando errado justamente o caso-teste canônico do projeto.

Por isso `phonology/levenshtein.py` roda duas passadas: o DP padrão e depois uma
passada de reparo que funde um símbolo elidido em `i` com sua reaparição em `j`, quando
`|i-j| ≤ janela` (default 3, configurável), num único evento `METATESE`.

## Privacidade

Áudio de criança é dado biométrico sob a LGPD. Mesmo sendo um lab local:

- `SPEECHLAB_DISCARD_AUDIO_AFTER_FEATURES=true` apaga o WAV depois de extrair features,
  guardando só sha256, vetores e métricas. Quebra o player sincronizado e impede
  re-execução — a UI avisa.
- `azure_pa` e `gemini_audio` **enviam áudio para fora** e só rodam com opt-in explícito
  por engine no `.env`. A UI marca visualmente quais engines fazem isso.
- Zero telemetria. `data/audio/` e o banco estão no `.gitignore`.

## Pré-requisitos

```bash
brew install ffmpeg espeak-ng
```

No macOS arm64 o `phonemizer` não localiza a lib do espeak sozinho; o `.env.example` já
traz `PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib`.

A primeira execução baixa ~7 GB de modelos do HuggingFace.

## Uso

```bash
cp .env.example .env
make setup
make check      # ffmpeg? espeak? modelos? chaves?
make dev        # backend :8000 + frontend :5173
```

## Device

Detecção automática `cuda → mps → cpu`. Em Apple Silicon:

- o forward do wav2vec2 roda em **MPS**;
- `torchaudio.functional.forced_align` e o cálculo de GOP são **forçados em CPU** — o
  primeiro não tem kernel MPS, e o segundo produz números que vão ser reportados, então
  precisa ser determinístico;
- `faster-whisper` é **sempre CPU int8**: CTranslate2 não tem backend Metal.

Em 16 GB de RAM, `SPEECHLAB_MAX_LOCAL_CONCURRENCY=2` e o LRU de modelos não são higiene
— são requisito de funcionamento.
