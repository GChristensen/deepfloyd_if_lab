import importlib

import ipywidgets as widgets
from IPython.core.display_functions import clear_output
from IPython.display import display
from ipywidgets import HTML, VBox, Layout

from ..const import VERSION
from ..settings import settings
from ..pipelines.inpainting import InpaintingPipeline
from ..pipelines.super_resolution import SuperResolutionPipeline
from ..pipelines.style_transfer import StylePipeline
from ..pipelines.dream import DreamPipeline
from .settings import SettingsUI
from .super_resolution import SuperResolutionUI
from .dream import Txt2ImgUI
from .style_transfer import Img2ImgUI
from .inpainting import InpaintingUI
from .pnginfo import PNGInfoUI


class DeepFloydIFUI:
    def __init__(self, stages=None):
        self.stages = stages
        self.root_box = None
        self.title_label = None
        self.uis = None
        self.inpainting = None
        self.super_resolution = None
        self.style_transfer = None
        self.dream = None
        self.tabs = None

        if stages:
            self.init(stages)

    def init(self, stages):
        self.tabs = widgets.Tab()

        self.dream = self.create_dream_ui(stages)
        self.style_transfer = self.create_style_ui(stages)
        self.super_resolution = self.create_sr_ui(stages)
        self.inpainting = self.create_inpainting_ui(stages)

        self.uis = [
            self.dream,
            self.style_transfer,
            self.super_resolution,
            self.inpainting
        ]
        pipeline_uis = self.uis[:]

        pnginfo_ui = PNGInfoUI(self.uis, self.tabs)
        self.uis.append(pnginfo_ui)

        settings_ui = SettingsUI(self.uis)
        self.uis.append(settings_ui)

        for ui in pipeline_uis:
            ui.save_ui_state = self.save_ui_state
            ui.load_ui_state = self.load_ui_state
            ui.send_to_sr = self.send_to_super_resolution
            ui.restore_ui_state()

        self.tabs.children = [ui.get() for ui in self.uis]

        for i in range(len(self.tabs.children)):
            self.tabs.set_title(i, self.uis[i].get_title())

        self.title_label = widgets.Label(f"DeepFloyd IF Lab v{VERSION}", layout=Layout(display="flex",
                                                                                       justify_content="flex-end"))
        self.root_box = VBox([self.title_label, self.tabs])

    def create_dream_ui(self, stages):
        dream_pipeline = DreamPipeline(stages)
        return Txt2ImgUI(dream_pipeline)

    def create_style_ui(self, stages):
        style_pipeline = StylePipeline(stages)
        return Img2ImgUI(style_pipeline)

    def create_sr_ui(self, stages):
        sr_pipeline = SuperResolutionPipeline(stages)
        return SuperResolutionUI(sr_pipeline)

    def create_inpainting_ui(self, stages):
        inpainting_pipeline = InpaintingPipeline(stages)
        return InpaintingUI(inpainting_pipeline)

    def save_ui_state(self, key, state):
        ui_state = settings.get("ui_state", {})
        ui_state[key] = state
        settings["ui_state"] = ui_state
        settings.save()

    def load_ui_state(self, key):
        ui_state = settings.get("ui_state", {})
        return ui_state.get(key, None)

    def send_to_super_resolution(self, image, parameters):
        self.super_resolution.set_support_image(image, parameters)
        self.tabs.selected_index = 2

    def get(self):
        return self.root_box

    def show(self):
        self._apply_style()
        display(self.root_box)

    @staticmethod
    def load():
        stages_module = importlib.import_module('.'.join(__name__.split('.')[:-2]) + ".pipelines.stages")
        if_stages = stages_module.DeepFloydIFStages()
        if_stages.load()

        clear_output()

        ui = DeepFloydIFUI()
        ui.init(if_stages)
        return ui

    def _apply_style(self):
        display(
            HTML(
                """
                    <style>
                    .lm-TabBar-tab {
                        flex: 0 1 var(--jp-widgets-horizontal-tab-width);
                        min-width: 35px;
                        min-height: calc(var(--jp-widgets-horizontal-tab-height) + var(--jp-border-width));
                        line-height: var(--jp-widgets-horizontal-tab-height);
                        margin-left: calc(-1 * var(--jp-border-width));
                        padding: 0px 10px;
                        background: var(--jp-layout-color2);
                        color: var(--jp-ui-font-color2);
                        border: var(--jp-border-width) solid var(--jp-border-color1);
                        border-bottom: none;
                        position: relative;
                    }
                    
                    .lm-TabBar-tab.p-mod-current {
                        color: var(--jp-ui-font-color0);
                        /* We want the background to match the tab content background */
                        background: var(--jp-layout-color1);
                        min-height: calc(var(--jp-widgets-horizontal-tab-height) + 2 * var(--jp-border-width));
                        transform: translateY(var(--jp-border-width));
                        overflow: visible;
                    }
                    
                    .lm-TabBar-tab.p-mod-current:before {
                        position: absolute;
                        top: calc(-1 * var(--jp-border-width));
                        left: calc(-1 * var(--jp-border-width));
                        content: '';
                        height: var(--jp-widgets-horizontal-tab-top-border);
                        width: calc(100% + 2 * var(--jp-border-width));
                        background: var(--jp-brand-color1);
                    }
                    
                    .lm-TabBar-tab:first-child {
                        margin-left: 0;
                    }
                    
                    .lm-TabBar-tab:hover:not(.p-mod-current) {
                        background: var(--jp-layout-color1);
                        color: var(--jp-ui-font-color1);
                    }
                    
                    .lm-TabBar-tabIcon,
                    .lm-TabBar-tabLabel,
                    .lm-TabBar-tabCloseIcon {
                        line-height: var(--jp-widgets-horizontal-tab-height);
                    }
                    
                    .iflab-title-label {
                        width: 99%;
                        background-color: var(--jp-layout-color2);
                        margin-top: 10px;
                    }
                    </style>
                """
            ))
