from .pipeline import PipelineUI, catch_handler_errors

class Txt2ImgUI(PipelineUI):
    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.IMAGE_FOLDER = "dream"

    def _tune_ui(self):
        self.images_box.layout.display = "none"
        self.info_button.tooltip = "Generate images from a text prompt"

    def on_display_info(self, button):
        pass
        #webbrowser.open("https://github.com/deep-floyd/IF#i-dream")

    def get_title(self):
        return "Dream"

