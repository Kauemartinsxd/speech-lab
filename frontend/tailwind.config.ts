import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // bancada de instrumentos, não landing page
        bench: {
          bg: "#0f1115",
          panel: "#171a21",
          border: "#262b36",
          text: "#e6e8ec",
          muted: "#8b93a7",
        },
        // as três categorias têm cores distintas e nenhuma delas é "erro"
        categoria: {
          decodificacao: "#e8a33d",
          vernacular: "#4ea8de",
          atipico: "#c76b98",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
