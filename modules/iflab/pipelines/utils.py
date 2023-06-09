import torch

def get_total_gb_vram():
    if torch.cuda.is_available():
        return round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)
    else:
        return 0


def get_default_settings():
    result = {}

    total_gb_vram = get_total_gb_vram()

    if total_gb_vram >= 24:
        result = {"alternate_load": False}
    elif total_gb_vram >= 12:
        result = {
            "alternate_load": True
        }
    else:
        result = {
            "stageI_model": "IF-I-L-v1.0",
            "stageII_model": "IF-II-M-v1.0",
            "alternate_load": True
        }

    return result
