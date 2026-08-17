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
independente, **sem beam search e sem KenLM**. A expectativa era que, sem nada no
caminho puxando para a forma padrão, ele escrevesse `pobrema`.

É por isso que a decodificação greedy é obrigatória aqui. Ligar beam search com LM
reintroduziria exatamente o viés que o lab quer medir.

**A expectativa não se confirmou.** Ver "Resultados do M0" abaixo.

## Resultados do M0 (2026-08-17)

10 amostras cobrindo 7 processos fonológicos, voz sintética `Luciana`, medidas contra
anotação humana. Validade do seed conferida antes: `scripts/validate_seed.py` sintetiza
a frase padrão e a variante com a mesma voz e mede a distância entre os sinais — nos 10
pares o RMS ficou entre 0,11 e 0,28, muito acima do limiar de 0,01. O TTS articulou
mesmo as variantes, então transcrever a forma padrão é normalização, não fidelidade.

| | preservou o desvio | normalizou | ambíguo |
|---|:-:|:-:|:-:|
| `whisper_baseline` | 1 | **9** | 0 |
| `whisper_strict` | 1 | **9** | 0 |
| `ctc_greedy` | 1 | **8** | 1 |

Três achados:

**1. O caso canônico reproduz.** Com "O pobrema da conta é difícil", o Whisper escreve
`O problema da conta é difícil.` e o CTC escreve `o pobrinma da conta é difícil` — a
metátese (`pobr-` em vez de `probl-`) sobrevive no CTC e desaparece no Whisper.

**2. Tirar o condicionamento não muda nada.** `whisper_strict`
(`condition_on_previous_text=False`, `temperature=0`) produziu saída idêntica ao
`whisper_baseline` nas 10 amostras. A normalização não vem do condicionamento no texto
anterior; vem do decoder.

**3. O CTC greedy também normaliza — em 8 dos 10 casos.** Este é o resultado que
contraria a hipótese de partida. Em `blusa → brusa` os papéis chegaram a se inverter: o
Whisper preservou `brusa` e o CTC escreveu `blusa`.

A hipótese para o item 3, ainda não testada: o modelo XLSR foi fine-tunado sobre
transcrições **ortográficas** de fala real, e isso assa um prior lexical na própria
camada de saída acústica. Remover o KenLM remove só uma das duas fontes de
normalização; a outra veio junto com o fine-tuning.

Se isso se confirmar, é o argumento mais forte a favor do Caminho 2: um modelo de
**fonemas** não tem léxico ortográfico para o qual puxar.

Ressalva que limita todas as conclusões acima: o áudio é TTS adulto, fora do domínio de
um modelo treinado em fala espontânea real (CoRAA), e mais ainda fora do domínio da fala
infantil. Isto mede o comportamento das engines nestes dados, não responde à pergunta de
pesquisa. Quem responde é o aparato de ground truth com fala real (M6).

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
