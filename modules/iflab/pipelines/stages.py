import importlib
import os
import time

from huggingface_hub import notebook_login, login
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII

from ..const import EXPERIMENTAL
from ..settings import settings

FLOYD_HOME = os.getenv("FLOYD_HOME")
LOGGED_IN_HF = os.path.exists(os.path.join(FLOYD_HOME, "home", "huggingface", "token"))

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


class DeepFloydIFStages:
    def __init__(self):
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

            device = 'cuda:0'
            self.t5 = T5Embedder(device="cpu")
            self.if_I = ModIFStageI(self.stageI_model_name, device=device)
            self.if_II = ModIFStageII(self.stageII_model_name, device=device)

            try:
                self.if_III = ModIFStageIII(self.stageIII_model_name, device=device)
            except Exception as e:
                time.sleep(.5)
                self.if_III = ModIFStageIII(self.stageIII_model_name, device=device)

            self.loaded = True
            print("DeepFloyd IF stages are loaded.")
