from deepfloyd_if.pipelines import style_transfer
from deepfloyd_if.pipelines.utils import _prepare_pil_image

from .pipeline import Pipeline, IFResult
from .stages import ModelError
from .. import EXPERIMENTAL, SEQ_LOAD_OFF


class StylePipeline(Pipeline):
    def __init__(self, stages):
        super().__init__(stages)

        self.custom_paramsI = {"sample_timestep_respacing": "10,10,10,10,10,10,10,10,0,0",
                               "support_noise_less_qsample_steps": 5}
        self.custom_paramsII = {"support_noise_less_qsample_steps": 5}

    def invoke(self, **kwargs):
        if EXPERIMENTAL and self.experimental:
            return self.experimental.style_transfer_mod(**kwargs)
        else:
            return style_transfer(**kwargs)

    def invoke_ref(self, **kwargs):
        if EXPERIMENTAL and self.experimental:
            print('reference style_transfer')
            return self.experimental.style_transfer_ref(**kwargs)
        else:
            return style_transfer(**kwargs)

    def modify_args(self, args):
        args["support_pil_img"] = self.support_image
        # TODO: move support images to the class level in all pipelines for optimization
        args["if_I_kwargs"].low_res = _prepare_pil_image(self.support_image, 64)
        args["if_I_kwargs"].mid_res = _prepare_pil_image(self.support_image, 256)

    def upscale(self, **kwargs):
        if self.stages.sequential_load != SEQ_LOAD_OFF:
            raise ModelError("Upscale is implemented only for I+II+III")
        else:
            return super().upscale(**kwargs)
