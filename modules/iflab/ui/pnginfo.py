import weakref

import ipywidgets as widgets
from ipywidgets import HBox, Layout

from .pipeline import PipelineUI, catch_handler_errors


class PNGInfoUI(PipelineUI):
    def __init__(self, uis, tabs):
        super().__init__(None)
        self.uis = [weakref.ref(ui) for ui in uis]
        self.tabs = tabs

    def _tune_ui(self):
        self.upload_support_img_button.description = "Load Image"
        self.upload_support_img_button.observe(self.load_pnginfo_image)
        self.send_to_label = widgets.Label("Send to:")
        self.dream_button = widgets.Button(description='Dream')
        self.dream_button.on_click(self.send_pnginfo)
        self.style_button = widgets.Button(description='Style Transfer')
        self.style_button.on_click(self.send_pnginfo)
        self.sr_button = widgets.Button(description='Super Resolution')
        self.sr_button.on_click(self.send_pnginfo)
        self.inpainting_button = widgets.Button(description='Inpainting')
        self.inpainting_button.on_click(self.send_pnginfo)
        self.spacer = HBox(layout=Layout(flex="1 0 auto"))
        self.pnginfo_button = widgets.FileUpload(
            description='PNG Info',
            accept='.png',
            multiple=False,
            tooltip='Load PNG Info from image'
        )
        self.pnginfo_button.observe(self.load_pnginfo)
        self.button_box.children = [self.send_to_label, self.dream_button, self.style_button, self.sr_button,
                                    self.inpainting_button]
        self.paste_support_img_button.layout.display = "none"
        self.info_button.tooltip = "Browse PNGInfo"

    def on_display_info(self, button):
        pass

    def load_pnginfo_image(self, e):
        self.load_support_image(e)
        self.load_pnginfo(e)

    def send_pnginfo(self, button):
        for i, c in enumerate(self.button_box.children):
            if c is button:
                ui = self.uis[i - 1]()
                parameters = self.get_ui_parameters()
                ui.set_ui_parameters(parameters)
                self.tabs.selected_index = i - 1
                break

    def get_title(self):
        return "PNG Info"
