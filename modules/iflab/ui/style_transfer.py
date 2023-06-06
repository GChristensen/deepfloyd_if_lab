from .pipeline import PipelineUI


class Img2ImgUI(PipelineUI):
    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.IMAGE_FOLDER = "style_transfer"

    def _tune_ui(self):
        self.info_button.tooltip = "Upload source image and provide a style prompt to produce image of a different style"

    def on_display_info(self, button):
        pass

    def get_title(self):
        return "Style Transfer"

    def generate_series(self, **kwargs):
        with self.output:
            if self.pipeline.style_prompt:
                super().generate_series(**kwargs)
            else:
                print("Please provide a style prompt")

