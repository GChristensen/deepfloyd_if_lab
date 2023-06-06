import time
from abc import ABC
from collections import UserDict

from deepfloyd_if.pipelines import super_resolution
from .pipeline import Pipeline, IFResult
from .. import EXPERIMENTAL


class SuperResolutionPipeline(Pipeline):
    def __init__(self, stages):
        super().__init__(stages)

        self.custom_paramsII = {"aug_level": 0.2}

    def modify_args(self, args):
        pass

    def invoke(self, **kwargs):
        if EXPERIMENTAL and self.experimental:
            return self.experimental.super_resolution_mod(**kwargs)
        else:
            return super_resolution(**kwargs)

    def invoke_ref(self, **kwargs):
        if EXPERIMENTAL and self.experimental:
            print('reference super_resolution')
            return self.experimental.super_resolution_ref(**kwargs)
        else:
            return super_resolution(**kwargs)

    def generate(self, seed=None, progress=True, reference=False):
        if seed is None:
            seed = self.generate_seed()

        if_II_kwargs = UserDict({
            'sample_timestep_respacing': self.stepsII,
            'guidance_scale': self.guidanceII,
        })
        self.add_custom_parameters(if_II_kwargs, self.custom_paramsII)
        if_II_kwargs.t5_embs = self.t5_embs
        if_II_kwargs.negative_t5_embs = self.negative_t5_embs

        if_III_kwargs = {
            "guidance_scale": self.guidanceIII,
            "noise_level": self.noiseIII,
            "sample_timestep_respacing": str(self.stepsIII),
        }

        kwargs = dict(
            t5=self.stages.t5,
            if_III=self.stages.if_II,
            support_pil_img=self.support_image,
            img_scale=4.,
            img_size=64,
            disable_watermark=self.disable_watermark,
            progress=progress,
            seed=seed,
            if_III_kwargs=if_II_kwargs
        )

        prompt, negative, _ = self.prepare_prompts()

        if prompt:
            kwargs["prompt"] = prompt
        if negative:
            kwargs["negative_prompt"] = negative

        time_start = time.perf_counter()
        self.merge_args(kwargs, self.override_args)
        seed = kwargs["seed"]

        invoke = self.invoke_ref if reference else self.invoke
        middle_res = invoke(
            **kwargs
        )
        self.result_upscale = invoke(
            t5=self.stages.t5,
            if_III=self.stages.if_III,
            prompt=[''],
            support_pil_img=middle_res['III'][0],
            img_scale=4.,
            img_size=256,
            disable_watermark=self.disable_watermark,
            progress=progress,
            seed=seed,
            return_tensors=True,
            if_III_kwargs=if_III_kwargs
        )

        duration = time.perf_counter() - time_start

        images, tensors = self.result_upscale
        output = images["output"]
        self.result_upscale = IFResult(images, tensors, output, None, seed, time.time(), 0, duration)
        return self.result_upscale
