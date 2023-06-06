import ipywidgets as widgets

from .pipeline import PipelineUI, catch_handler_errors


class SuperResolutionUI(PipelineUI):
    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.IMAGE_FOLDER = "super_resolution"

    def _tune_ui(self):
        self.SERIES_BUTTON_LABEL = "Batch Stage III"
        self.result_box.layout.display = "none"
        self.upscale_button.layout.display = "none"
        self.clear_results_button.layout.display = "none"
        self.generate_button.description = "Stage III"
        self.generate_series_button.description = self.SERIES_BUTTON_LABEL
        self.info_button.tooltip = "Upload source image and provide a prompt to generate an upscaled version"

    def on_display_info(self, button):
        pass

    def get_title(self):
        return "Super Resolution"

    @catch_handler_errors
    def generate_series(self, button=None, seed=None, steps=None, single=False):
        try:
            self.upscaling = True
            self.upscaling_stage = "III"
            self.pipeline.generate_series(steps=steps, seed=seed, callback=self.process_stageI_result, progress=self.update_progress)
            self.reset_progress()
            self.generation_thread = None
        finally:
            self.generation_thread = None
            if not single:
                button.description = self.SERIES_BUTTON_LABEL

    def process_stageI_result(self, result):
        with self.output:
            if self.upscaling_progress_event:
                self.upscaling_progress_event.set()

            self.process_upscale_result(result.seed, result, "III")

            duration = round(result.duration, 1)
            label = "Stages II-III"
            self.status_box.children = [widgets.Label(f"{label}: {duration}s")]

    def set_support_image(self, image, parameters):
        self.support_img_view.value = self._image_to_bytes(image)
        self.support_img_view.layout.display = "inline-block"
        self.prompt_text.value = parameters.get("prompt", "")
        self.negative_prompt_text.value = parameters.get("negative_prompt", "")
        self.style_prompt_text.value = parameters.get("style_prompt", "")

