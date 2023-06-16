import importlib
import time
import random
import traceback

from abc import ABC, abstractmethod
from collections import UserDict
from dataclasses import dataclass
from datetime import datetime
from random import randint

import numpy as np
from torch import autocast

from .stages import ModelError
from .. import SEQ_LOAD_OFF, SEQ_LOAD_SEPARATE, DEBUG, SEQ_LOAD_MERGE


@dataclass
class IFResult:
    images: ...
    tensors: ...
    generations: ...
    args: ...
    seed: ...
    time: ...
    iterations: ...
    duration: ...


class Pipeline(ABC):
    def __init__(self, stages):
        self.override_args = {}
        self.settings = {}
        self.stages = stages

        self.count = 1

        self.aspect_ratio = "1:1"
        self.prompt = None
        self.negative_prompt = None
        self.style_prompt = None

        self.t5_embs = None
        self.negative_t5_embs = None
        self.style_t5_embs = None

        self.guidanceI = 7.0
        self.stepsI = "smart100"
        self.guidanceII = 4.0
        self.stepsII = "smart50"
        self.guidanceIII = 4.0
        self.stepsIII = 75
        self.noiseIII = 20

        self.custom_paramsI = None
        self.custom_paramsII = None
        self.custom_paramsIII = None

        self.iterationsI = 0
        self.iterationsII = 0
        self.iterationsIII = 0

        self.result_stageI = None
        self.result_upscale = None
        self.generationsI = {}

        self.support_image = None
        self.mask_image = None
        self.disable_watermark = False
        self.pass_prompt_to_stage_III = None

        self.on_before_embeddings = lambda: None
        self.on_before_generation = lambda: None
        self.on_before_upscale = lambda: None
        self.on_before_checkpoints_loaded = lambda m: None
        self.on_checkpoints_loaded = lambda: None

        try:
            self.experimental = importlib.import_module('.'.join(__name__.split('.')[:-1]) + ".experimental")
        except ImportError as e:
            self.experimental = None

    def prepare_embeddings(self):
        if not self.stages.has_t5():
            self.on_before_checkpoints_loaded(False)

        self.stages.free_stageIII()
        self.stages.free_stageII()
        self.stages.free_stageI()

        was_loaded = self.stages.ensure_t5()

        if not self.has_t5_loaded:
            raise ModelError("Error loading T5 encoder.")

        if was_loaded:
            self.on_checkpoints_loaded()

        self.on_before_embeddings()

    def compute_t5_embs(self, update_prompt=True, update_negative=True, update_style=True):
        self.prepare_embeddings()

        if update_prompt:
            if self.prompt is not None:
                promptv = [self.prompt] * self.count
                self.t5_embs = self.stages.t5.get_text_embeddings(promptv)
            else:
                self.t5_embs = None

        if update_negative:
            if self.negative_prompt is not None:
                promptv = [self.negative_prompt] * self.count
                self.negative_t5_embs = self.stages.t5.get_text_embeddings(promptv)
            else:
                self.negative_t5_embs = None

        if update_style:
            if self.style_prompt is not None:
                promptv = [self.style_prompt] * self.count
                self.style_t5_embs = self.stages.t5.get_text_embeddings(promptv)
            else:
                self.style_t5_embs = None

    def generate_seed(self):
        return randint(0, np.iinfo(np.int32).max)

    def prepare_prompts(self):
        prompt = [self.prompt] if isinstance(self.prompt, str) else self.prompt
        negative = [self.negative_prompt] if isinstance(self.negative_prompt, str) else self.negative_prompt
        style = [self.style_prompt] if isinstance(self.style_prompt, str) else self.style_prompt

        return prompt, negative, style

    def merge_args(self, args, override):
        for k, v in override.items():
            if (isinstance(v, dict) or isinstance(v, UserDict)) and k in args:
                self.merge_args(args[k], v)
            elif v is not None:
                args[k] = v

    def add_custom_parameters(self, stage_args, params):
        if params and stage_args:
            for k, v in params.items():
                stage_args[k] = v

    @property
    def is_optimized(self):
        return self.stages.sequential_load != SEQ_LOAD_OFF

    @property
    def has_t5_loaded(self):
        return self.stages.has_t5()

    @property
    def has_stageI_loaded(self):
        return self.stages.has_stageI()

    @property
    def has_stageII_loaded(self):
        return self.stages.has_stageII()

    @property
    def has_stageIII_loaded(self):
        return self.stages.has_stageIII()

    def prepare_generation(self):
        if not self.has_stageI_loaded:
            self.on_before_checkpoints_loaded(not self.stages.downloaded_stageI())

        self.stages.free_t5()
        self.stages.free_stageII()
        self.stages.free_stageIII()
        was_loaded = self.stages.ensure_stageI()

        if not self.has_stageI_loaded:
            raise ModelError("Error loading stage I model.")

        if was_loaded:
            self.on_checkpoints_loaded()

        self.on_before_generation()

    def generate(self, seed=None, progress=True, is_reference=False):
        self.prepare_generation()

        if seed is None:
            seed = self.generate_seed()

        prompt, negative, style = self.prepare_prompts()

        if_I_kwargs = UserDict({
            "guidance_scale": self.guidanceI,
            "sample_timestep_respacing": self.stepsI
        })
        self.add_custom_parameters(if_I_kwargs, self.custom_paramsI)

        if prompt:
            if_I_kwargs["t5_embs"] = self.t5_embs
        if negative:
            if_I_kwargs["negative_t5_embs"] = self.negative_t5_embs
        if style:
            if_I_kwargs["style_t5_embs"] = self.style_t5_embs

        kwargs = dict(
            t5=self.stages.t5, if_I=self.stages.if_I,
            prompt=prompt,
            negative_prompt=negative,
            style_prompt=style,
            disable_watermark=self.disable_watermark,
            seed=seed,
            if_I_kwargs=if_I_kwargs,
            return_tensors=True,
            progress=progress
        )

        self.modify_args(kwargs)
        self.merge_args(kwargs, self.override_args)
        seed = kwargs["seed"]

        time_start = time.perf_counter()

        invoke = self.invoke_ref if is_reference else self.invoke
        self.result_stageI = invoke(**kwargs)

        duration = time.perf_counter() - time_start
        its = self.iterationsI / duration

        images, tensors = self.result_stageI
        output = images.get("output", [[]])
        self.result_stageI = IFResult(images, tensors, output, if_I_kwargs, seed, time.time(), its, duration)
        self.generationsI[seed] = self.result_stageI

        return self.result_stageI

    def prepare_upscale(self, stage):
        if not self.stages.has_stageII() or not self.stages.has_stageIII():
            missing = stage == "II" and not self.stages.downloaded_stageII() or \
                      stage == "III" and not self.stages.downloaded_stageIII()

            self.on_before_checkpoints_loaded(missing)

        self.stages.free_t5()
        self.stages.free_stageI()

        was_loaded = False
        if stage == "II":
            if self.stages.sequential_load == SEQ_LOAD_SEPARATE and self.has_stageIII_loaded:
                self.stages.free_stageIII()

            was_loaded = self.stages.ensure_stageII()

            if not self.has_stageII_loaded:
                raise ModelError("Error loading stage II model.")

        if stage == "III" or self.stages.sequential_load == SEQ_LOAD_MERGE:
            if self.stages.sequential_load == SEQ_LOAD_SEPARATE and self.has_stageII_loaded:
                self.stages.free_stageII()

            try:
                was_loaded = self.stages.ensure_stageIII() or was_loaded
            except Exception as e:
                if DEBUG:
                    traceback.print_exc()

            if not self.has_stageIII_loaded:
                raise ModelError("Error loading stage III model.")

        if was_loaded:
            self.on_checkpoints_loaded()

        self.on_before_upscale()

    def get_upscaleII_kwargs(self, resultI):
        if_II_kwargs = UserDict({
            "guidance_scale": self.guidanceII,
            "sample_timestep_respacing": self.stepsII
        })
        self.add_custom_parameters(if_II_kwargs, self.custom_paramsII)
        # the usage of patched pipelines is assumed here, although technically it will work without the patch
        # some extra parameters are passed as metadata to preserve the original interface
        resultI.args.imagesI = resultI.images["I"]
        resultI.args.tensorsI = resultI.tensors[0]

        return if_II_kwargs

    def get_upscaleIII_kwargs(self, resultII):
        if_III_kwargs = UserDict({
            "guidance_scale": self.guidanceIII,
            "noise_level": self.noiseIII,
            "sample_timestep_respacing": str(self.stepsIII),
        })
        self.add_custom_parameters(if_III_kwargs, self.custom_paramsIII)

        if_III_kwargs.imagesII = resultII.images["II"]
        if_III_kwargs.tensorsII = resultII.tensors[1]
        if_III_kwargs.pass_prompt_to_sIII = self.pass_prompt_to_stage_III

        return if_III_kwargs

    def upscale(self, resultI=None, resultII=None, progress=False, is_reference=False):
        stage = "III" if resultII else "II"

        self.prepare_upscale(stage)

        seed = resultI.seed
        rtime = resultI.time
        if_I_kwargs = resultI.args
        if_II_kwargs = self.get_upscaleII_kwargs(resultI) if stage == "II" else None
        if_III_kwargs = self.get_upscaleIII_kwargs(resultII) if stage == "III" else None
        if_III = self.stages.if_III if stage == "III" else None
        prompt, negative, style = self.prepare_prompts()

        time_start = time.perf_counter()

        kwargs = dict(
            t5=self.stages.t5, if_I=self.stages.if_I, if_II=self.stages.if_II, if_III=if_III,
            prompt=prompt,
            negative_prompt=negative,
            style_prompt=style,
            disable_watermark=self.disable_watermark,
            seed=seed,
            if_I_kwargs=if_I_kwargs,
            if_II_kwargs=if_II_kwargs,
            if_III_kwargs=if_III_kwargs,
            return_tensors=True,
            progress=progress
        )

        self.modify_args(kwargs)
        self.merge_args(kwargs, self.override_args)
        seed = kwargs["seed"]

        invoke = self.invoke_ref if is_reference else self.invoke
        self.result_upscale = invoke(**kwargs)

        duration = time.perf_counter() - time_start

        images, tensors = self.result_upscale
        output = images.get("output", [[]])
        self.result_upscale = IFResult(images, tensors, output, if_II_kwargs, seed, rtime, 0, duration)

        return self.result_upscale

    @abstractmethod
    def invoke(self, **kwargs):
        pass

    @abstractmethod
    def invoke_ref(self, **kwargs):
        pass

    @abstractmethod
    def modify_args(self, args):
        pass