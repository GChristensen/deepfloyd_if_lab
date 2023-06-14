import torch
import os

from pathlib import Path

from .const import SEQ_LOAD_OFF, SEQ_LOAD_MERGE, SEQ_LOAD_SEPARATE
from .settings import settings


def get_total_gb_vram():
    if torch.cuda.is_available():
        return round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)
    else:
        return 8


def apply_default_mem_settings():
    total_gb_vram = get_total_gb_vram()

    seq_load = settings.get("sequential_load", None)

    if not seq_load:
        if total_gb_vram >= 24:
            settings.set("sequential_load", SEQ_LOAD_OFF)
        elif total_gb_vram >= 12:
            settings.set("sequential_load", SEQ_LOAD_MERGE)
        else:
            settings.set("sequential_load", SEQ_LOAD_SEPARATE)
            settings.set("stageI_model", "IF-I-L-v1.0")
            settings.set("stageII_model", "IF-II-L-v1.0")

        settings.save()

    return settings.get("sequential_load")


def init_installation():
    apply_default_mem_settings()

