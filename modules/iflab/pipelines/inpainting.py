import numpy as np
from PIL import Image

from deepfloyd_if.pipelines import inpainting
from deepfloyd_if.pipelines.utils import _prepare_pil_image

from .pipeline import Pipeline
from .. import EXPERIMENTAL


class InpaintingPipeline(Pipeline):
    def __init__(self, stages):
        super().__init__(stages)

        self.custom_paramsI = {"sample_timestep_respacing": "10,10,10,10,10,0,0,0,0,0",
                               "support_noise_less_qsample_steps": 0}
        self.custom_paramsII = {"aug_level": 0.0}

    def invoke(self, **kwargs):
        if EXPERIMENTAL and self.experimental:
            return self.experimental.inpainting_mod(**kwargs)
        else:
            return inpainting(**kwargs)

    def invoke_ref(self, **kwargs):
        if EXPERIMENTAL and self.experimental:
            print('reference inpainting')
            return self.experimental.inpainting_ref(**kwargs)
        else:
            return inpainting(**kwargs)

    def modify_args(self, args):
        del args["style_prompt"]
        if hasattr(args["if_I_kwargs"], 'style_t5_embs'):
            del args["if_I_kwargs"]['style_t5_embs']

        if self.support_image:
            args["support_pil_img"] = self.support_image

            inpainting_mask = None
            if self.mask_image:
                inpainting_mask = np.array(self.mask_image)
            else:
                blank_pil_image = Image.new('RGB', self.support_image.size, (255, 255, 255))
                inpainting_mask = np.array(blank_pil_image)

            inpainting_mask = np.moveaxis(inpainting_mask, -1, 0)

            args['inpainting_mask'] = inpainting_mask

            args["if_I_kwargs"].low_res = _prepare_pil_image(self.support_image, 64)
            args["if_I_kwargs"].mid_res = _prepare_pil_image(self.support_image, 256)
            args["if_I_kwargs"].high_res = _prepare_pil_image(self.support_image, 1024)

