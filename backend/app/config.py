"""Configuração do lab. Tudo vem do .env — nada de constante mágica em código."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPEECHLAB_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- geral ---------------------------------------------------------------
    db_path: Path = Path("data/speechlab.db")
    audio_dir: Path = Path("data/audio")
    log_level: str = "INFO"

    # --- device --------------------------------------------------------------
    device: str = "auto"
    force_cpu_for_alignment: bool = True
    max_local_concurrency: int = 2
    engine_timeout_s: int = 300
    model_cache_size: int = 2

    # --- privacidade ---------------------------------------------------------
    discard_audio_after_features: bool = False

    # --- engines habilitadas -------------------------------------------------
    enable_whisper_baseline: bool = True
    enable_whisper_strict: bool = True
    enable_ctc_greedy: bool = True
    enable_phoneme_gop: bool = False
    enable_azure_pa: bool = False
    enable_gemini_audio: bool = False

    # --- whisper -------------------------------------------------------------
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "int8"

    # --- ctc -----------------------------------------------------------------
    ctc_model: str = "Edresson/wav2vec2-large-xlsr-coraa-portuguese"

    # --- phoneme_gop ---------------------------------------------------------
    phoneme_model: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    gop_threshold: float = -2.0
    metathesis_window: int = 3

    # --- custo (tarifas mudam; ficam fora do código) -------------------------
    azure_usd_per_hour: float = 1.00
    gemini_usd_per_1k_input: float = 0.0
    gemini_usd_per_1k_output: float = 0.0
    gemini_model: str = "gemini-2.5-flash"

    @property
    def abs_db_path(self) -> Path:
        return self._absolute(self.db_path)

    @property
    def abs_audio_dir(self) -> Path:
        return self._absolute(self.audio_dir)

    def _absolute(self, p: Path) -> Path:
        return p if p.is_absolute() else (REPO_ROOT / p)

    def enabled_engines(self) -> dict[str, bool]:
        return {
            "whisper_baseline": self.enable_whisper_baseline,
            "whisper_strict": self.enable_whisper_strict,
            "ctc_greedy": self.enable_ctc_greedy,
            "phoneme_gop": self.enable_phoneme_gop,
            "azure_pa": self.enable_azure_pa,
            "gemini_audio": self.enable_gemini_audio,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
