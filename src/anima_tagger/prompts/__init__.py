from __future__ import annotations

from importlib.resources import files

from anima_tagger.domain.models import LoraMode, Stage


def load_prompt(mode: LoraMode, stage: Stage, *, trigger_word: str, first_caption: str = "") -> str:
    name = f"{mode.value}_{stage.value}.txt"
    template = files(__package__).joinpath(name).read_text(encoding="utf-8")
    return template.replace("{trigger_word}", trigger_word).replace("{first_caption}", first_caption)

