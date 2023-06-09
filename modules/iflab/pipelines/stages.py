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

from .utils import get_total_gb_vram
from ..const import EXPERIMENTAL
from ..settings import settings

IFLAB_HOME = os.getenv("IFLAB_HOME")
LOGGED_IN_HF = os.path.exists(os.path.join(IFLAB_HOME, "home", "huggingface", "token"))

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


class DeepFloydIFStages:
    def __init__(self):
        self.cuda_device = "cuda:0"

        if sys.platform == "darwin":
            self.cuda_device = "mps"

        self.alternate_load = settings.get("alternate_load", True)

        self.stageI_model_name = settings.get("stageI_model", "IF-I-XL-v1.0")
        self.stageII_model_name = settings.get("stageII_model", "IF-II-L-v1.0")
        self.stageIII_model_name = settings.get("stageIII_model", "stable-diffusion-x4-upscaler")
        self.loaded = False

    def load(self):
        if not self.loaded:
            if not LOGGED_IN_HF:
                import pyperclip
                hf_token = os.getenv("HF_TOKEN") or pyperclip.paste()

                if hf_token and hf_token.startswith("hf_"):
                    login(hf_token)
                else:
                    notebook_login()

            t5_dtype_text = settings.get("t5_dtype", None)
            t5_dtype = torch.float32 if t5_dtype_text == "float32" else torch.bfloat16
            self.t5 = T5Embedder(device="cpu", torch_dtype=t5_dtype)

            self.if_I = self.load_stageI()
            self.if_II = None
            self.if_III = None

            if not self.alternate_load:
                self.if_II = self.load_stageII()

                try:  # may raise error just after it was downloaded from HF
                    self.if_III = self.load_stageIII()
                except:
                    try:
                        time.sleep(.5)
                        self.if_III = self.load_stageIII()
                    except:
                        traceback.print_exc()

            self.loaded = True

    def load_stageI(self):
        try:
            self.if_I = ModIFStageI(self.stageI_model_name, device=self.cuda_device)
            return self.if_I
        except:
            traceback.print_exc()

    def has_stageI(self):
        return not not self.if_I

    def ensure_stageI(self):
        if not self.if_I:
            self.load_stageI()

    def unload_stageI(self):
        if self.if_I:
            self.if_I.model.to("cpu")
            self.if_I = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def free_stageI(self):
        if self.alternate_load:
            self.unload_stageI()

    def load_stageII(self):
        try:
            self.if_II = ModIFStageII(self.stageII_model_name, device=self.cuda_device)
            return self.if_II
        except:
            traceback.print_exc()

    def has_stageII(self):
        return not not self.if_II

    def ensure_stageII(self):
        if not self.if_II:
            self.load_stageII()

    def free_stageII(self, empty_cache=True):
        if self.alternate_load:
            self.unload_stageII(empty_cache)

    def unload_stageII(self, empty_cache=True):
        if self.if_II:
            self.if_II.model.to("cpu")
            self.if_II = None

            if empty_cache:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def load_stageIII(self):
        self.if_III = ModIFStageIII(self.stageIII_model_name, device=self.cuda_device)
        return self.if_III

    def has_stageIII(self):
        return not not self.if_III

    def ensure_stageIII(self):
        if not self.if_III:
            self.load_stageIII()

    def unload_stageIII(self):
        if self.if_III:
            self.if_III.model.to("cpu")
            self.if_III = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def free_stageIII(self):
        if self.alternate_load:
            self.unload_stageIII()
