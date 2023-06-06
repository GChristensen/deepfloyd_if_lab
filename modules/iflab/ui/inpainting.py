from .pipeline import PipelineUI


class InpaintingUI(PipelineUI):
    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.IMAGE_FOLDER = "inpainting"

    def _tune_ui(self):
        self.upload_mask_img_button.layout.display = "block"
        self.paste_mask_img_button.layout.display = "block"
        self.info_button.tooltip = ("Inpaint on a region of the source image defined by the mask image\n" +
                                    "Source and mask images should be of the same size")

    def on_display_info(self, button):
        pass

    def get_title(self):
        return "Inpainting"