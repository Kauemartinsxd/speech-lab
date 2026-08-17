/**
 * Diff de palavras entre a transcrição de uma engine e o texto de referência.
 *
 * Cuidado de leitura: aqui, casar com a referência NÃO é sucesso. Se o aluno
 * falou "pobrema" e a engine escreveu "problema", ela bate com a referência
 * justamente porque destruiu o dado. A UI diz isso em texto.
 */

export type WordOp = "igual" | "diferente" | "ausente" | "extra";

export interface WordDiff {
  text: string;
  op: WordOp;
}

function normalizar(palavra: string): string {
  return palavra
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^\p{L}\p{N}]/gu, "");
}

export function tokenizar(texto: string): string[] {
  return texto.trim().split(/\s+/).filter(Boolean);
}

/** Alinhamento por LCS: marca as palavras que divergem da referência. */
export function diffPalavras(referencia: string, hipotese: string): WordDiff[] {
  const ref = tokenizar(referencia);
  const hip = tokenizar(hipotese);
  const a = ref.map(normalizar);
  const b = hip.map(normalizar);

  const lcs: number[][] = Array.from({ length: a.length + 1 }, () =>
    new Array(b.length + 1).fill(0),
  );
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }

  const saida: WordDiff[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      saida.push({ text: hip[j], op: "igual" });
      i++;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      // palavra da referência que a engine não produziu; se logo em seguida
      // vier uma palavra extra, o par vira uma substituição
      if (j < b.length && lcs[i + 1][j] === lcs[i][j + 1]) {
        saida.push({ text: hip[j], op: "diferente" });
        i++;
        j++;
      } else {
        saida.push({ text: ref[i], op: "ausente" });
        i++;
      }
    } else {
      saida.push({ text: hip[j], op: "extra" });
      j++;
    }
  }
  while (j < b.length) saida.push({ text: hip[j++], op: "extra" });
  while (i < a.length) saida.push({ text: ref[i++], op: "ausente" });

  return saida;
}

export function contarDivergencias(diff: WordDiff[]): number {
  return diff.filter((d) => d.op !== "igual").length;
}
