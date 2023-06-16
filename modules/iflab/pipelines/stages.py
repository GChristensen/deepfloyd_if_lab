import gc
import importlib
import os
import sys
import time
import traceback

import torch

from huggingface_hub import notebook_login, login
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII

from .. import SEQ_LOAD_OFF, SEQ_LOAD_SEPARATE
from ..const import EXPERIMENTAL, DEBUG
from ..init import apply_default_mem_settings
from ..settings import settings

if sys.platform == "darwin":
    from . import mac

IFLAB_HOME = os.getenv("IFLAB_HOME")
LOGGED_IN_HF = os.path.exists(os.path.join(IFLAB_HOME, "home", "huggingface", "token"))

DEFAULT_MODEL_T5 = "t5-v1_1-xxl"
DEFAULT_MODEL_I = "IF-I-XL-v1.0"
DEFAULT_MODEL_II = "IF-II-L-v1.0"
DEFAULT_MODEL_III = "stable-diffusion-x4-upscaler"

if EXPERIMENTAL:
    try:
        experimental_module = importlib.import_module('.'.join(__name__.split('.')[:-1]) + ".experimental")
    except ImportError as e:
        experimental_module = None
    ModIFStageI = experimental_module.ModIFStageI
    ModIFStageII = experimental_module.ModIFStageII
    ModIFStageIII = experimental_module.ModIFStageIII
else:
    ModIFStageI = IFStageI
    ModIFStageII = IFStageII
    ModIFStageIII = StableStageIII


class ModelError(Exception):
    pass

def get_t5_device_map(device):
    return {
        'shared': device,
        'encoder.embed_tokens': device,
        'encoder.block.0': device,
        'encoder.block.1': device,
        'encoder.block.2': device,
        'encoder.block.3': device,
        'encoder.block.4': device,
        'encoder.block.5': device,
        'encoder.block.6': device,
        'encoder.block.7': device,
        'encoder.block.8': device,
        'encoder.block.9': device,
        'encoder.block.10': 'cpu',
        'encoder.block.11': 'cpu',
        'encoder.block.12': 'cpu',
        'encoder.block.13': 'cpu',
        'encoder.block.14': 'cpu',
        'encoder.block.15': 'cpu',
        'encoder.block.16': 'cpu',
        'encoder.block.17': 'cpu',
        'encoder.block.18': 'cpu',
        'encoder.block.19': 'cpu',
        'encoder.block.20': 'cpu',
        'encoder.block.21': 'cpu',
        'encoder.block.22': 'cpu',
        'encoder.block.23': 'cpu',
        'encoder.final_layer_norm': 'cpu',
        'encoder.dropout': 'cpu',
    }


def offload_to_disk(device_map):
    device_map = device_map.copy()

    for k, v in device_map.items():
        if v == "cpu":
            device_map[k] = "disk"

    return device_map

class DeepFloydIFStages:
    def __init__(self):
        self.device = os.getenv("IFLAB_DEVICE", "cuda:0")

        if sys.platform == "darwin":
            self.device = "mps"

        self.t5 = None
        self.t5_on_gpu = False
        self.if_I = None
        self.if_II = None
        self.if_III = None

        self.sequential_load = settings.get("sequential_load", None)

        if self.sequential_load is None:
            self.sequential_load = apply_default_mem_settings()

        self.t5_model_name = DEFAULT_MODEL_T5
        self.t5_dtype = "bfloat16" if os.getenv("IFLAB_T5_DTYPE", "float32").lower() == "bfloat16" else "float32"
        self.stageI_model_name = settings.get("stageI_model", DEFAULT_MODEL_I)
        self.stageII_model_name = settings.get("stageII_model", DEFAULT_MODEL_II)
        self.stageIII_model_name = settings.get("stageIII_model", DEFAULT_MODEL_III)
        self.has_missing_models = False
        self.loaded = False

        vram_fraction = os.getenv("IFLAB_VRAM_FRACTION", None)
        if vram_fraction:
            torch.cuda.set_per_process_memory_fraction(float(vram_fraction), self.device)

    def load(self):
        if not self.loaded:
            if not LOGGED_IN_HF:
                import pyperclip
                hf_token = os.getenv("HF_TOKEN") or pyperclip.paste()

                if hf_token and hf_token.startswith("hf_"):
                    login(hf_token)
                else:
                    notebook_login()

            if os.getenv("IFLAB_INSTALL", False):
                self.ensure_t5()
                self.ensure_stageI()
                self.unload_stageI()
                self.ensure_stageII()
                self.unload_stageII()

                try:  # may raise error just after it was downloaded from HF
                    self.load_stageIII()
                except:
                    try:
                        time.sleep(.5)
                        self.load_stageIII()
                    except:
                        traceback.print_exc()
                self.unload_stageIII()
            else:
                self.has_missing_models = (not self.downloaded_t5() or not self.downloaded_stageI()
                                           or not self.downloaded_stageII() or not self.downloaded_stageIII())

            self.loaded = True

    def load_t5(self):
        try:
            t5_dtype = torch.bfloat16 if self.t5_dtype == "bfloat16" else torch.float32
            t5_model_kwargs = None
            t5_device = "cpu"

            if os.getenv("IFLAB_T5_ON_GPU", False):
                self.t5_on_gpu = True
                t5_device = self.device
                t5_device_map = get_t5_device_map(t5_device)

                if os.getenv("IFLAB_LOWRAM", False):
                    t5_device_map = offload_to_disk(t5_device_map)

                t5_model_kwargs = {"low_cpu_mem_usage": True,
                                   "device_map": t5_device_map,
                                   "offload_folder": "../home/t5-offload"}

            self.t5 = T5Embedder(device=t5_device, torch_dtype=t5_dtype, t5_model_kwargs=t5_model_kwargs)
        except:
            if DEBUG:
                traceback.print_exc()

    def has_t5(self):
        return not not self.t5

    def downloaded_t5(self):
        return self.if_stage_exists(self.t5_model_name)

    def ensure_t5(self):
        result = bool(self.t5)
        if not self.t5:
            self.load_t5()
        return result

    def unload_t5(self, empty_cache=True):
        if self.t5:
            self.unload_stage("t5", empty_cache=empty_cache)

    def free_t5(self, empty_cache=True):
        if self.t5_on_gpu:
            self.unload_t5(empty_cache=empty_cache)

    def load_stageI(self):
        try:
            self.if_I = ModIFStageI(self.stageI_model_name, device=self.device)
            return self.if_I
        except:
            traceback.print_exc()

    def has_stageI(self):
        current_stageI = settings.get("stageI_model", DEFAULT_MODEL_I)

        if current_stageI != self.stageI_model_name:
            self.stageI_model_name = current_stageI
            self.unload_stageI()

        return not not self.if_I

    def downloaded_stageI(self):
        return self.if_stage_exists(self.stageI_model_name)

    def if_stage_exists(self, model_name):
        return os.path.exists(f"../home/.cache/IF_/{model_name}")

    def ensure_stageI(self):
        result = bool(self.if_I)
        if not self.if_I:
            self.load_stageI()
        return result

    def unload_stageI(self, empty_cache=True):
        if self.if_I:
            self.unload_stage("if_I", empty_cache=empty_cache)

    def unload_stage(self, stage, empty_cache=True):
        try:
            getattr(self, stage).model.to("cpu")
        except:
            if DEBUG:
                traceback.print_exc()

        setattr(self, stage, None)
        if empty_cache:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def free_stageI(self, empty_cache=True):
        if self.sequential_load != SEQ_LOAD_OFF:
            self.unload_stageI(empty_cache=empty_cache)

    def load_stageII(self):
        try:
            self.if_II = ModIFStageII(self.stageII_model_name, device=self.device)
            return self.if_II
        except:
            traceback.print_exc()

    def has_stageII(self):
        current_stageII = settings.get("stageII_model", DEFAULT_MODEL_II)

        if current_stageII != self.stageII_model_name:
            self.stageII_model_name = current_stageII
            self.unload_stageII()

        return not not self.if_II

    def downloaded_stageII(self):
        return self.if_stage_exists(self.stageII_model_name)

    def ensure_stageII(self):
        result = bool(self.if_II)
        if not self.if_II:
            self.load_stageII()
        return result

    def free_stageII(self, empty_cache=True):
        if self.sequential_load != SEQ_LOAD_OFF:
            self.unload_stageII(empty_cache=empty_cache)

    def unload_stageII(self, empty_cache=True):
        if self.if_II:
            self.unload_stage("if_II", empty_cache=empty_cache)

    def load_stageIII(self):
        separate = self.sequential_load == SEQ_LOAD_SEPARATE
        kwargs = None

        if separate:
            kwargs = {"precision": "16"}

        self.if_III = ModIFStageIII(self.stageIII_model_name, device=self.device, model_kwargs=kwargs)

        if separate:
            self.if_III.model.enable_sequential_cpu_offload()

        return self.if_III

    def has_stageIII(self):
        current_stageIII = settings.get("stageIII_model", DEFAULT_MODEL_III)

        if current_stageIII != self.stageIII_model_name:
            self.stageIII_model_name = current_stageIII
            self.unload_stageIII()

        return not not self.if_III

    def downloaded_stageIII(self):
        if self.stageIII_model_name.startswith("IF"):
            return self.if_stage_exists(self.stageIII_model_name)
        else:
            return os.path.exists("../home/huggingface/hub/models--stabilityai--stable-diffusion-x4-upscaler/snapshots")

    def ensure_stageIII(self):
        result = bool(self.if_III)
        if not self.if_III:
            self.load_stageIII()
        return result

    def unload_stageIII(self, empty_cache=True):
        if self.if_III:
            self.unload_stage("if_III", empty_cache=empty_cache)

    def free_stageIII(self, empty_cache=True):
        if self.sequential_load != SEQ_LOAD_OFF:
            self.unload_stageIII(empty_cache=empty_cache)
