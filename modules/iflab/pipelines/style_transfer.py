from IPython.core.display_functions import display
from deepfloyd_if.pipelines import style_transfer
from deepfloyd_if.pipelines.utils import _prepare_pil_image

from .pipeline import Pipeline, IFResult
from .. import EXPERIMENTAL


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
        args["if_I_kwargs"].low_res = _prepare_pil_image(self.support_image, 64)
        args["if_I_kwargs"].mid_res = _prepare_pil_image(self.support_image, 256)

