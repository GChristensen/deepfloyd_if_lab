import json

import ipywidgets as widgets
from ipywidgets import VBox, Layout

from ..settings import settings


class SettingsUI():
    def __init__(self, uis):
        self.uis = uis

        models_label = widgets.Label(" Models (requires kernel restart)",
                                     style=dict(background="var(--jp-layout-color2)"))

        self.stageI_model_dropdown = widgets.Dropdown(
            options=['IF-I-XL-v1.0', 'IF-I-L-v1.0', 'IF-I-M-v1.0'],
            value='IF-I-XL-v1.0',
            description='Stage I: ',
            style= {'description_width': 'max-content'},
            disabled=False,
        )
        self.stageI_model_dropdown.observe(self.on_settings_changed, 'value', type='change')
        self.stageI_model_dropdown.__setting = "stageI_model"
        self.stageI_model_dropdown.value = settings.get(self.stageI_model_dropdown.__setting,
                                                        self.stageI_model_dropdown.options[0])
        self.stageII_model_dropdown = widgets.Dropdown(
            options=['IF-II-L-v1.0', 'IF-II-M-v1.0'],
            value='IF-II-L-v1.0',
            description='Stage II: ',
            style= {'description_width': 'max-content'},
            disabled=False,
        )
        self.stageII_model_dropdown.observe(self.on_settings_changed, 'value', type='change')
        self.stageII_model_dropdown.__setting = "stageII_model"
        self.stageII_model_dropdown.value = settings.get(self.stageII_model_dropdown.__setting,
                                                         self.stageII_model_dropdown.options[0])
        self.stageIII_model_dropdown = widgets.Dropdown(
            options=['stable-diffusion-x4-upscaler', 'IF-III-L-v1.0'],
            value='stable-diffusion-x4-upscaler',
            description='Stage III: ',
            style= {'description_width': 'max-content'},
            disabled=False,
        )
        self.stageIII_model_dropdown.observe(self.on_settings_changed, 'value', type='change')
        self.stageIII_model_dropdown.__setting = "stageIII_model"
        self.stageIII_model_dropdown.value = settings.get(self.stageIII_model_dropdown.__setting,
                                                          self.stageIII_model_dropdown.options[0])

        self.sequential_loading_dropdown = widgets.Dropdown(
            options=[('I+II+III (24GB)', 'off',), ('I/II+III (12GB)', 'merge'), ('I/II/III (8GB)', 'separate')],
            value='separate',
            description='Layout: ',
            style= {'description_width': 'max-content'},
            disabled=False,
        )
        self.sequential_loading_dropdown.observe(self.on_settings_changed, 'value', type='change')
        self.sequential_loading_dropdown.__setting = "sequential_load"
        self.sequential_loading_dropdown.value = settings.get(self.sequential_loading_dropdown.__setting,
                                                              self.sequential_loading_dropdown.options[2][1])

        generation_label = widgets.Label(" Generation",
                                         style=dict(background="var(--jp-layout-color2)"))

        self.disable_watermark_check = widgets.Checkbox(
            description="Disable watermark"
        )
        self.disable_watermark_check.observe(self.on_settings_changed, 'value', type='change')
        self.disable_watermark_check.__setting = "disable_watermark"
        self.disable_watermark_check.value = settings.get(self.disable_watermark_check.__setting, False)

        ui_label = widgets.Label(" User Interface",
                                 style=dict(background="var(--jp-layout-color2)"))
        self.remember_ui_state_check = widgets.Checkbox(
            description="Remember UI state"
        )
        self.remember_ui_state_check.observe(self.on_settings_changed, 'value', type='change')
        self.remember_ui_state_check.__setting = "remember_ui_state"
        self.remember_ui_state_check.value = settings.get(self.remember_ui_state_check.__setting, False)

        self.root_box = VBox([models_label,
                              self.stageI_model_dropdown, self.stageII_model_dropdown,
                              self.stageIII_model_dropdown,
                              self.sequential_loading_dropdown,
                              generation_label,
                              self.disable_watermark_check,
                              ui_label,
                              self.remember_ui_state_check],
                             layout=Layout(min_height="500px"))

    def get(self):
        return self.root_box

    def get_title(self):
        return "Settings"

    def load_settings(self):
        settings.load()
        self.notify_uis()

    def on_settings_changed(self, e):
        key = e.owner.__setting
        value = e["new"]

        self.setting_action(key, value)

        settings[key] = value
        settings.save()
        self.notify_uis()

    def setting_action(self, key, value):
        pass

    def settings_changed(self, e):
        pass

    def notify_uis(self):
        for ui in self.uis:
            ui.settings_changed(settings)
