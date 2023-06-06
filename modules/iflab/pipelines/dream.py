from deepfloyd_if.pipelines.dream import dream
from .pipeline import Pipeline
from .. import EXPERIMENTAL


class DreamPipeline(Pipeline):
    def __init__(self, stages):
        super().__init__(stages)

    def invoke(self, **kwargs):
        if EXPERIMENTAL and self.experimental:
            return self.experimental.dream_mod(**kwargs)
        else:
            return dream(**kwargs)

    def invoke_ref(self, **kwargs):
        if EXPERIMENTAL and self.experimental:
            print('reference dream')
            return self.experimental.dream_ref(**kwargs)
        else:
            return dream(**kwargs)

    def modify_args(self, args):
        args["aspect_ratio"] = self.aspect_ratio