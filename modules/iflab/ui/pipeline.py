import ast
import os
import random
import re
import json
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from abc import ABC, abstractmethod

from threading import Thread
from io import BytesIO
from pathlib import Path

import ipywidgets as widgets

from ipywidgets import HBox, VBox, Layout
from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageGrab

from ..const import DEBUG, RESPACING_MODES, UI_THREADS, DEBUG_PROMPT
from ..pipelines.stages import ModelError
from .. import SEQ_LOAD_SEPARATE


LOADING_CHECKPOINTS = "Loading checkpoints..."
DOWNLOADING_CHECKPOINTS = "Downloading checkpoints, please wait..."

def catch_handler_errors(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        result = None

        try:
            result = method(*args, **kwargs)
        except Exception as e:
            with args[0].output:
                traceback.print_exc()

        return result
    return wrapped


class PipelineUI(ABC):
    def __init__(self, pipeline):
        self.DEFAULT_PROMPT = DEBUG_PROMPT if DEBUG else ""
        self.STAGE_I_SCALE = 2.5
        self.STOP_BUTTON_LABEL = "Stop"
        self.SERIES_BUTTON_LABEL = "Batch Stage I"
        self.UPSCALE_BUTTON_LABEL = "🔬Upscale"

        self.settings = {}
        self.pipeline = pipeline

        if self.pipeline:
            self.pipeline.on_before_embeddings = self.on_before_embeddings
            self.pipeline.on_before_generation = self.on_before_generation
            self.pipeline.on_before_upscale = self.on_before_upscale
            self.pipeline.on_before_checkpoints_loaded = self.on_before_checkpoints_loaded
            self.pipeline.on_checkpoints_loaded = self.on_checkpoints_loaded

        self.generation_thread = None
        self.stop_generation = False
        self.resultsI = {}

        self.upscaleII = []
        self.upscaleIII = []
        self.upscale_resultsII = {}
        self.upscale_result_boxes = {}
        self.upscaling = False
        self.upscaling_stage = None
        self.upscaling_stage_max = None
        self.upscaling_progress_thread = None
        self.upscaling_progress_event = None
        self.stop_upscale = False

        self.stageI_time = 0
        self.stageII_time = 0
        self.stageIII_time = 0
        self.stageIII_iter_time = 0

        self.save_ui_state = lambda k, s: None
        self.load_ui_state = lambda k: None
        self.send_to_sr = lambda i, p: None

        self.output = widgets.Output()
        self.prompt_label = widgets.Label("Prompt")
        self.prompt_text = widgets.Textarea(value=self.DEFAULT_PROMPT, layout=Layout(width="99%", height="6em"))
        self.negative_prompt_label = widgets.Label("Negative prompt")
        self.negative_prompt_text = widgets.Textarea(value="", layout=Layout(width="99%", height="6em"))
        self.style_prompt_label = widgets.Label("Style prompt")
        self.style_prompt_text = widgets.Textarea(value="", layout=Layout(width="99%", height="6em"))

        self.stageI_custom_params_text = widgets.Text(
            description='if_I_kwargs:',
            value=self.format_custom_parameters(pipeline.custom_paramsI if pipeline else ""),
            layout=Layout(width="99%")
        )
        self.stageII_custom_params_text = widgets.Text(
            description='if_II_kwargs:',
            value=self.format_custom_parameters(pipeline.custom_paramsII if pipeline else ""),
            layout=Layout(width="99%")
        )
        self.stageIII_custom_params_text = widgets.Text(
            description='if_III_kwargs:',
            value=self.format_custom_parameters(pipeline.custom_paramsIII if pipeline else ""),
            layout=Layout(width="99%")
        )
        self.aspect_ratio_text = widgets.Text(
            value='1:1',
            description='Aspect Ratio'
        )
        self.sIII_pass_prompt_check = widgets.Checkbox(
            description="Pass prompt to the stage III"
        )
        self.sIII_pass_prompt_check.value = True
        self.custom_parameter_box = VBox([
            widgets.Label("Advanced options"),
            self.stageI_custom_params_text,
            self.stageII_custom_params_text,
            self.stageIII_custom_params_text,
            self.aspect_ratio_text,
            self.sIII_pass_prompt_check
        ], layout=Layout(display="none"))

        self.prompt_box = VBox([self.prompt_label, self.prompt_text,
                                self.negative_prompt_label, self.negative_prompt_text,
                                self.style_prompt_label, self.style_prompt_text,
                                self.custom_parameter_box],
                               layout=Layout(flex="1 0 auto"))

        self.params_label = widgets.Label(" ")
        self.batch_images_slider = widgets.IntSlider(
            value=2 if DEBUG else 100,
            min=1,
            max=250,
            step=1,
            description='Batch Images',
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.respacingI_slider = widgets.SelectionSlider(
            options=RESPACING_MODES,
            value='smart27' if DEBUG else 'smart100',
            description='Respacing I',
            orientation='horizontal',
            readout=True
        )
        self.guidanceI_slider = widgets.IntSlider(
            value=7,
            min=1,
            max=20,
            step=1,
            description='Guidance I',
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.respacingII_slider = widgets.SelectionSlider(
            options=RESPACING_MODES,
            value='smart50',
            description='Respacing II',
            orientation='horizontal',
            readout=True
        )
        self.guidanceII_slider = widgets.IntSlider(
            value=4,
            min=1,
            max=20,
            step=1,
            description='Guidance II',
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.respacingIII_slider = widgets.IntSlider(
            value=75,
            min=0,
            max=200,
            step=5,
            description='Respacing III',
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.guidanceIII_slider = widgets.IntSlider(
            value=4,
            min=1,
            max=20,
            step=1,
            description='Guidance III',
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.noiseIII_slider = widgets.IntSlider(
            value=20,
            min=1,
            max=100,
            step=1,
            description='Noise III',
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.seed_number = widgets.IntText(
            value=-1,
            description='Seed',
            disabled=False
        )

        self.fast_presets = widgets.Button(
            description='Fast',
            tooltip='Fast generation (prompt tuning)',
            layout=Layout(width="33.3%")
        )
        self.fast_presets.on_click(self.on_set_fast_presets)
        self.default_presets = widgets.Button(
            description='Default',
            tooltip='Default settings',
            layout=Layout(width="33.3%")
        )
        self.default_presets.on_click(self.on_set_default_presets)
        self.hq_presets = widgets.Button(
            description='Elaborate',
            tooltip='Slow generation (more details)',
            layout=Layout(width="33.3%")
        )
        self.hq_presets.on_click(self.on_set_hq_presets)
        self.preset_box = HBox([self.fast_presets, self.default_presets, self.hq_presets])

        self.params_box = VBox([self.params_label, self.batch_images_slider, self.respacingI_slider, self.guidanceI_slider,
                                self.respacingII_slider, self.guidanceII_slider,
                                self.respacingIII_slider, self.guidanceIII_slider, self.noiseIII_slider,
                                self.seed_number, self.preset_box])

        self.input_box = HBox([self.prompt_box, self.params_box], layout=Layout(width="100%"))

        self.info_button = widgets.Button(description='ℹ️', tooltip='', layout=Layout(width='40px'),
                                          style={"button_color": "#212121"})
        self.info_button.on_click(self.on_display_info)

        self.generate_button = widgets.Button(
            description='Stage I',
            tooltip='Generate stage I image'
        )
        self.generate_button.on_click(self.on_generate_click)

        self.generate_series_button = widgets.Button(
            description=self.SERIES_BUTTON_LABEL,
            tooltip='Generate a series of stage I images'
        )
        self.generate_series_button.on_click(self.on_generate_series_click)
        self.clear_results_button = widgets.Button(
            description='🗑️',
            tooltip='Clear stage I results',
            layout=Layout(width="40px")
        )
        self.clear_results_button.on_click(self.clear_results)
        self.clear_results_button2 = widgets.Button(
            description='🗑️',
            tooltip='Clear stage I results',
            layout=Layout(width="40px")
        )
        self.clear_results_button2.on_click(self.clear_results)
        self.upscale_button = widgets.Button(
            description=self.UPSCALE_BUTTON_LABEL,
            tooltip='Upscale the selected stage I images'
        )
        self.upscale_button.on_click(self.on_upscale_click)
        self.upscale_button2 = widgets.Button(
            description=self.UPSCALE_BUTTON_LABEL,
            tooltip='Upscale the selected stage I images'
        )
        self.upscale_button2.on_click(self.on_upscale_click)
        self.embeddings_button = widgets.Button(
            description='T5',
            tooltip='Only generate T5 embeddings',
            layout=Layout(width="40px", display="block")
        )
        self.embeddings_button.on_click(lambda b: self.compute_embeddings())
        self.pnginfo_button = widgets.FileUpload(
            description='PNG Info',
            accept='.png',
            multiple=False,
            tooltip='Load PNG Info from image'
        )
        self.pnginfo_button.observe(self.load_pnginfo, 'value', type='change')
        self.clear_upscales_button = widgets.Button(
            description='🗑️',
            tooltip='Clear upscales',
            layout=Layout(width="40px")
        )
        self.clear_upscales_button.on_click(self.clear_upscales)
        self.clear_upscales_button2 = widgets.Button(
            description='🗑️',
            tooltip='Clear upscales',
            layout=Layout(width="40px")
        )
        self.clear_upscales_button2.on_click(self.clear_upscales)
        self.custom_parameters_button = widgets.Button(
            description='🎰',
            tooltip='Advanced options',
            layout=Layout(width="40px")
        )
        self.custom_parameters_button.on_click(self.toggle_custom_parameters)
        self.random_seed_button = widgets.Button(
            description='🎲',
            tooltip='Use random seed',
            layout=Layout(width="40px")
        )
        self.random_seed_button.on_click(lambda b: setattr(self.seed_number, "value", -1))
        spacer = HBox([], layout=Layout(flex="1 0 auto"))
        padder = HBox([], layout=Layout(flex="0 1 10px"))

        self.button_box = HBox([self.info_button, self.generate_button, self.generate_series_button,
                                self.clear_results_button, self.upscale_button, self.clear_upscales_button,
                                self.custom_parameters_button, self.random_seed_button, spacer,
                                self.pnginfo_button, padder],
                               layout=Layout(flex="1 0 auto"))

        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=0,
            bar_style='',
            orientation='horizontal',
            layout=Layout(width="100%")
        )

        self.stageI_results_label = widgets.Label(" Stage I Results", layout=Layout(display="none"),
                                                  style={"font_weight": "bold"})
        self.stageI_results_label.add_class("iflab-title-label")
        self.upscale_results_label = widgets.Label(" Upscale Results", layout=Layout(display="none"),
                                                   style={"font_weight": "bold"})
        self.upscale_results_label.add_class("iflab-title-label")

        self.status_box = HBox([], layout=Layout(width="272px", flex="0 0 auto"))
        self.status_icon = widgets.Label(value="  ", layout=Layout(width="2em"))
        self.status_container = HBox([self.status_icon, self.status_box])

        self.control_box = HBox([self.button_box, self.status_container],
                                layout=Layout(width="100%", margin="20px 0"))

        self.upload_support_img_button = widgets.FileUpload(
            description='Source Image',
            multiple=False,
            tooltip='Load source image'
        )
        self.upload_support_img_button.observe(self.load_support_image, 'value', type='change')
        self.paste_support_img_button = widgets.Button(
            description='Paste',
            tooltip='Paste support image from clipboard'
        )
        self.paste_support_img_button.on_click(self.paste_support_image)

        self.upload_mask_img_button = widgets.FileUpload(
            description='Mask Image',
            multiple=False,
            tooltip='Load mask image',
            layout=Layout(display="none")
        )
        self.upload_mask_img_button.observe(self.load_mask_image, 'value', type='change')
        self.paste_mask_img_button = widgets.Button(
            description='Paste',
            tooltip='Paste mask image from clipboard',
            layout=Layout(display="none")
        )
        self.paste_mask_img_button.on_click(self.paste_mask_image)

        self.support_img_view = widgets.Image(layout=Layout(width="max-content", height="max-content", max_width="95%", display="none"))
        self.mask_img_view = widgets.Image(layout=Layout(width="max-content", height="max-content", max_width="95%", display="none"))

        self.support_image_box = HBox([self.support_img_view], layout=Layout(width="50%"))
        self.mask_image_box = HBox([self.mask_img_view], layout=Layout(width="50%"))

        self.images_box = VBox([
            HBox([
                HBox([self.upload_support_img_button,
                      self.paste_support_img_button],
                     layout=Layout(width="50%")),
                HBox([self.upload_mask_img_button,
                      self.paste_mask_img_button],
                     layout=Layout(width="50%"))
            ]),
            HBox([self.support_image_box,
                  self.mask_image_box])
        ], layout=Layout(width="100%"))

        self.result_box = HBox([], layout=Layout(width="100%", margin="20px 0", flex_flow="row wrap", display="none"))
        self.result_button_box = VBox([
            widgets.HTML("<hr class='iflab-upscale-separator'>"),
            HBox([self.upscale_button2, self.clear_results_button2])
        ], layout=Layout(width="100%", margin="5px 0", display="none"))
        self.upscale_box = VBox([], layout=Layout(width="100%", margin="20px 0", display="none"))
        self.upscale_button_box = VBox([
            widgets.HTML("<hr class='iflab-upscale-separator'>"),
            HBox([self.clear_upscales_button2])
        ], layout=Layout(width="100%", margin="5px 0", display="none"))

        self.root_box = VBox([self.input_box, self.control_box, self.images_box,
                              self.stageI_results_label, self.result_box,self.result_button_box,
                              self.upscale_results_label, self.upscale_box, self.upscale_button_box,
                              self.output],
                             layout=Layout(width="100%"))

        self._tune_ui()

    def _get_nsfw_status(self, result, stage):
        nsfw = result.tensors is not None and result.tensors[stage] is not None \
               and hasattr(result.tensors[stage], "hentai")
        return "*" if nsfw else ""

    def _image_to_bytes(self, image):
        b = BytesIO()
        image.save(b, format='png')
        b.seek(0)
        return b.read()

    def _get_file_name(self, time, seed, stage):
        time_label = self._get_time_label(time)
        output_dir = os.getenv("IFLAB_OUTPUT_DIR", "outputs")

        return f"{output_dir}/{self.IMAGE_FOLDER}/{time_label}-{seed}-stage-{stage}.png"

    def _get_time_label(self, t):
        d = datetime.fromtimestamp(t)
        return d.strftime("%Y%m%d_%H%M%S")

    def status_message(self, text):
        self.status_box.children = [widgets.Label(text)]

    def set_status_idle(self):
        self.status_icon.value = "  "

    def set_status_computing(self):
        self.status_icon.value = "💻"

    def set_status_waiting(self):
        self.status_icon.value = "⏳"

    def set_status_result(self):
        self.status_icon.value = "📶"

    def set_status_error(self):
        self.status_icon.value = "⚠️"

    def show_progress_bar(self):
        self.progress_bar.value = 0
        self.status_box.children = [self.progress_bar]

    def is_prompt_valid(self):
        return self.prompt_text.value or self.negative_prompt_text.value or self.style_prompt_text.value

    def set_ui_parameters(self, parameters):
        self.prompt_text.value = parameters.get("prompt", "")
        self.negative_prompt_text.value = parameters.get("negative_prompt", "")
        self.style_prompt_text.value = parameters.get("style_prompt", "")
        self.seed_number.value = parameters.get("seed", -1)
        self.aspect_ratio_text.value = parameters.get("aspect_ratio", "")
        self.guidanceI_slider.value = parameters.get("guidanceI", 1)
        self.respacingI_slider.value = parameters.get("respacingI", "")
        self.guidanceII_slider.value = parameters.get("guidanceII", 1)
        self.respacingII_slider.value = parameters.get("respacingII", "")
        self.guidanceIII_slider.value = parameters.get("guidanceIII", 1)
        self.respacingIII_slider.value = int(parameters.get("respacingIII", 1))
        self.noiseIII_slider.value = parameters.get("noiseIII", 1)
        self.stageI_custom_params_text.value = parameters.get("if_I_kwargs", "") or ""
        self.stageII_custom_params_text.value = parameters.get("if_II_kwargs", "") or ""
        self.stageIII_custom_params_text.value = parameters.get("if_III_kwargs", "") or ""
        self.sIII_pass_prompt_check.value = parameters.get("pass_prompt_to_stage_III", True)

        self.mask_image_box.children = [widgets.HTML(f"""
            Stage: {parameters.get("stage", "")}<br>
            Stage I model: {parameters.get("stageI_model", "")}<br>
            Stage II model: {parameters.get("stageII_model", "")}<br>
            Stage III model: {parameters.get("stageIII_model", "")}<br>
            T5 precision: {parameters.get("t5_precision", "")}<br>
        """)]

    def set_seed_value(self, seed):
        self.seed_number.value = seed

    def on_set_fast_presets(self, button):
        self.respacingI_slider.value = "smart27"
        self.respacingII_slider.value = "smart27"
        self.respacingIII_slider.value = 20
        self.guidanceIII_slider.value = 4

    def on_set_default_presets(self, button):
        self.aspect_ratio_text.value = "1:1"
        self.guidanceI_slider.value = 7.0
        self.respacingI_slider.value = "smart100"
        self.guidanceII_slider.value = 4.0
        self.respacingII_slider.value = "smart50"
        self.guidanceIII_slider.value = 9.0
        self.respacingIII_slider.value = 75
        self.noiseIII_slider.value = 20

    def on_set_hq_presets(self, button):
        self.respacingI_slider.value = "smart185"
        self.respacingII_slider.value = "smart185"
        self.respacingIII_slider.value = 180
        self.guidanceIII_slider.value = 4

    def get_ui_parameters(self):
        parameters = {
            "prompt": self.prompt_text.value,
            "negative_prompt": self.negative_prompt_text.value,
            "style_prompt": self.style_prompt_text.value,
            "seed": self.seed_number.value,
            "aspect_ratio": self.aspect_ratio_text.value,
            "guidanceI": self.guidanceI_slider.value,
            "respacingI": self.respacingI_slider.value,
            "guidanceII": self.guidanceII_slider.value,
            "respacingII": self.respacingII_slider.value,
            "guidanceIII": self.guidanceIII_slider.value,
            "respacingIII": self.respacingIII_slider.value,
            "noiseIII": self.noiseIII_slider.value,
            "if_I_kwargs": self.stageI_custom_params_text.value,
            "if_II_kwargs": self.stageII_custom_params_text.value,
            "if_III_kwargs": self.stageIII_custom_params_text.value,
            "pass_prompt_to_stage_III": self.sIII_pass_prompt_check.value
        }

        return parameters

    def persist_ui_state(self):
        if self.settings.get("remember_ui_state", False):
            key = self.pipeline.__class__.__name__
            parameters = self.get_ui_parameters()
            self.save_ui_state(key, parameters)

    def restore_ui_state(self):
        if self.settings.get("remember_ui_state", False):
            parameters = self.load_ui_state(self.pipeline.__class__.__name__)
            if parameters:
                self.set_ui_parameters(parameters)

    def get_result_pnginfo(self, seed, stage):
        parameters = {
            "prompt": self.pipeline.prompt or "",
            "negative_prompt": self.pipeline.negative_prompt or "",
            "style_prompt": self.pipeline.style_prompt or "",
            "seed": seed,
            "stage": stage,
            "batch_size": 1,
            "batch_image": 1,
            "t5_precision": self.pipeline.stages.t5_dtype,
            "stageI_model": self.pipeline.stages.stageI_model_name,
            "stageII_model": self.pipeline.stages.stageII_model_name,
            "stageIII_model": self.pipeline.stages.stageIII_model_name,
            "aspect_ratio": self.pipeline.aspect_ratio,
            "guidanceI": self.pipeline.guidanceI,
            "respacingI": self.pipeline.stepsI,
            "guidanceII": self.pipeline.guidanceII,
            "respacingII": self.pipeline.stepsII,
            "guidanceIII": self.pipeline.guidanceIII,
            "respacingIII": str(self.pipeline.stepsIII),
            "noiseIII": self.pipeline.noiseIII,
            "if_I_kwargs": self.pipeline.custom_paramsI,
            "if_II_kwargs": self.pipeline.custom_paramsII,
            "if_III_kwargs": self.pipeline.custom_paramsIII,
            "pass_prompt_to_stage_III": self.sIII_pass_prompt_check.value
        }

        parameters_sd = (self.pipeline.prompt or "") + "\nNegative prompt: " + (self.pipeline.negative_prompt or "")

        pnginfo = PngInfo()
        pnginfo.add_text("deepfloyd_if", json.dumps(parameters))
        pnginfo.add_text("parameters", parameters_sd)
        return pnginfo

    @catch_handler_errors
    def load_pnginfo(self, e):
        if e["name"] == "value":
            file = e["new"][0]
            image = Image.open(BytesIO(file["content"].tobytes()))
            parameters_json = image.text.get("deepfloyd_if", None)

            if parameters_json:
                parameters = json.loads(parameters_json)
                self.set_ui_parameters(parameters)

    @catch_handler_errors
    def load_support_image(self, e):
        if e["name"] == "value":
            file = e["new"][0]
            self.support_img_view.value = file["content"].tobytes()
            self.support_img_view.layout.display = "inline-block"

    @catch_handler_errors
    def paste_support_image(self, button):
        image = ImageGrab.grabclipboard()
        self.support_img_view.value = self._image_to_bytes(image)
        self.support_img_view.layout.display = "inline-block"

    @catch_handler_errors
    def paste_mask_image(self, button):
        image = ImageGrab.grabclipboard()
        self.mask_img_view.value = self._image_to_bytes(image)
        self.mask_img_view.layout.display = "inline-block"

    @catch_handler_errors
    def load_mask_image(self, e):
        if e["name"] == "value":
            file = e["new"][0]
            self.mask_img_view.value = file["content"].tobytes()
            self.mask_img_view.layout.display = "inline-block"

    def toggle_custom_parameters(self, button):
        if self.custom_parameter_box.layout.display == "none":
            self.custom_parameter_box.layout.display = "block"
        else:
            self.custom_parameter_box.layout.display = "none"

    def format_custom_parameters(self, params):
        result = ""

        if params:
            params = [k + "=" + (f"'{v}'" if isinstance(v, str) else str(v)) for k, v in params.items()]
            result = ", ".join(params)

        return result

    def eval_custom_parameters(self, param_str):
        if param_str:
            def param_func(**kwargs):
                return kwargs

            try:
                return eval(f"param_func({param_str})")
            except Exception as e:
                print(f"Error evaluating custom parameters: {param_str}")

    def setup_pipeline(self, override_args={}):
        self.pipeline.override_args = override_args
        self.pipeline.aspect_ratio = self.aspect_ratio_text.value
        self.pipeline.guidanceI = float(self.guidanceI_slider.value)
        self.pipeline.stepsI = self.respacingI_slider.value
        self.pipeline.guidanceII = float(self.guidanceII_slider.value)
        self.pipeline.stepsII = self.respacingII_slider.value
        self.pipeline.guidanceIII = float(self.guidanceIII_slider.value)
        self.pipeline.stepsIII = self.respacingIII_slider.value
        self.pipeline.noiseIII = self.noiseIII_slider.value
        self.pipeline.custom_paramsI = self.eval_custom_parameters(self.stageI_custom_params_text.value)
        self.pipeline.custom_paramsII = self.eval_custom_parameters(self.stageII_custom_params_text.value)
        self.pipeline.custom_paramsIII = self.eval_custom_parameters(self.stageIII_custom_params_text.value)

        self.pipeline.iterationsI = int(re.findall("\d+", self.pipeline.stepsI)[0])
        self.pipeline.iterationsII = int(re.findall("\d+", self.pipeline.stepsII)[0])
        self.pipeline.iterationsIII = self.pipeline.stepsIII
        self.pipeline.disable_watermark = self.settings.get("disable_watermark", False)
        self.pipeline.pass_prompt_to_stage_III = self.sIII_pass_prompt_check.value

        if self.support_img_view.layout.display != "none":
            image = Image.open(BytesIO(self.support_img_view.value))
            self.pipeline.support_image = image

        if self.mask_img_view.layout.display != "none":
            image = Image.open(BytesIO(self.mask_img_view.value))
            self.pipeline.mask_image = image

    def compute_embeddings(self):
        if not self.is_prompt_valid():
            self.status_message("Please provide a prompt")

        prompt = self.prompt_text.value or None
        negative_prompt = self.negative_prompt_text.value or None
        style_prompt = self.style_prompt_text.value or None

        update_prompt = prompt != self.pipeline.prompt
        update_negative = negative_prompt != self.pipeline.negative_prompt
        update_style = style_prompt != self.pipeline.style_prompt

        if update_prompt or update_negative or update_style:
            try:
                self.pipeline.prompt = prompt
                self.pipeline.negative_prompt = negative_prompt
                self.pipeline.style_prompt = style_prompt
                self.pipeline.compute_t5_embs(update_prompt, update_negative, update_style)
                self.output.clear_output()
                self.show_progress_bar()
            except Exception as e:
                self.status_message(str(e))
                self.set_status_error()

    @catch_handler_errors
    def on_generate_click(self, button):
        if self.generation_thread is None:
            with self.output:
                self.persist_ui_state()
                self.reset_results()
                self.setup_pipeline()
                self.compute_embeddings()

                seed = self.seed_number.value
                seed = seed if seed > 0 else None

                if UI_THREADS:
                    self.generation_thread = Thread(target=lambda: self.generate_series(seed=seed, steps=1, single=True))
                    self.generation_thread.start()
                else:
                    self.generate_series(seed=seed, steps=1, single=True)

    @catch_handler_errors
    def on_generate_series_click(self, button):
        with self.output:
            if button.description == self.STOP_BUTTON_LABEL:
                button.description = self.SERIES_BUTTON_LABEL
                self.stop_generation = True
            elif self.generation_thread is None:
                button.description = self.STOP_BUTTON_LABEL
                self.persist_ui_state()
                self.reset_results()
                self.setup_pipeline()
                self.compute_embeddings()

                steps = self.batch_images_slider.value

                if UI_THREADS:
                    self.generation_thread = Thread(target=lambda: self.generate_series(button=button, steps=steps))
                    self.generation_thread.start()
                else:
                    self.generate_series(button=button, steps=steps)

    @catch_handler_errors
    def on_upscale_click(self, button):
        with self.output:
            if button.description == self.STOP_BUTTON_LABEL:
                self.upscale_button.description = self.UPSCALE_BUTTON_LABEL
                self.upscale_button2.description = self.UPSCALE_BUTTON_LABEL
                self.stop_upscale = True
            elif self.generation_thread is None:
                self.upscale_button.description = self.STOP_BUTTON_LABEL
                self.upscale_button2.description = self.STOP_BUTTON_LABEL
                self.persist_ui_state()
                self.show_progress_bar()
                self.reset_upscale_results()
                self.setup_pipeline()
                self.compute_embeddings()

                if UI_THREADS:
                    self.generation_thread = Thread(target=self.generate_upscales)
                    self.generation_thread.start()
                else:
                    self.generate_upscales()

    def on_before_embeddings(self):
        self.set_status_computing()
        self.status_message("Computing T5 Embeddings...")

    def on_before_generation(self):
        self.set_status_computing()
        self.show_progress_bar()

    @catch_handler_errors
    def generate_series(self, button=None, seed=None, steps=None, single=False):
        error = False

        try:
            self.upscaling = False
            generate_seed = seed is None

            self.stop_generation = False
            for i in range(0, steps):
                if self.stop_generation:
                    break

                if generate_seed:
                    seed = self.pipeline.generate_seed()

                result = self.pipeline.generate(seed, progress=self.update_progress)
                self.process_stageI_result(result)

            if not self.upscaling:  # Super Resolution
                self.status_message(f"Stage I: ~{self.stageI_time}s")

        except ModelError as e:
            error = True
            self.status_message(str(e))
        except MemoryError as e:
            error = True
            self.status_message("Memory error. Please restart the kernel.")
        finally:
            self.reset_progress()
            if error:
                self.set_status_error()
            else:
                self.set_status_result()
            self.generation_thread = None
            if not single:
                button.description = self.SERIES_BUTTON_LABEL

    def update_progress(self, n, p):
        try:
            self.stop_stageIII_progress()

            if len(self.status_box.children) and self.status_box.children[0] is not self.progress_bar:
                self.show_progress_bar()

            self.progress_bar.max = n - 1
            self.progress_bar.value = n - 1 - p
        except:
            pass

    def reset_progress(self):
        self.progress_bar.value = 0

    def reset_results(self):
        random.seed(datetime.now().timestamp())
        self.resultsI = {}
        self.upscale_resultsII = {}
        self.upscaleII = []
        self.upscaleIII = []
        self.clear_results(None)
        self.clear_upscales(None)

        self.reset_progress()
        self.output.clear_output()

    def reset_upscale_results(self):
        self.stageII_time = 0
        self.stageIII_time = 0
        self.upscale_resultsII = {}
        self.upscale_box.children = []
        self.upscale_result_boxes = {}

        self.output.clear_output()

    @catch_handler_errors
    def process_stageI_result(self, result):
        seed = result.seed
        image_result = result.images['I'][0]
        size = tuple([int(x * self.STAGE_I_SCALE) for x in image_result.size])

        self.resultsI[seed] = result
        self.stageI_time = round(result.duration)
        self.save_result(image_result, result.time, result.seed, "I")

        if DEBUG:
            image = image_result.resize(size, Image.Resampling.LANCZOS)
            image_view = widgets.Image(value=self._image_to_bytes(image), format='png')
            nsfw = self._get_nsfw_status(result, 0)
        else:
            file_name = self._get_file_name(result.time, seed, "I")
            image_view = widgets.HTML(f"<img src='/files/{file_name}' style='width: {size[0]}px; height: {size[1]}px'/>",
                                      layout=Layout(width=str(size[0]) + "px", height=str(size[1]) + "px"))
            nsfw = ""

        seed_text = widgets.Label(f"Seed: {seed}{nsfw}")
        spacer = HBox([], layout=Layout(flex="1 0 auto"))
        recycle_button = widgets.Button(
            description="♻️",
            tooltip="Reuse image seed",
            layout=Layout(width="30px"),
            style={"button_color": "#212121"}
        )
        recycle_button.on_click(lambda b: self.set_seed_value(seed))
        placeholder = widgets.Label(" ")

        top_box = widgets.HBox([seed_text, spacer, recycle_button, placeholder])

        upscale_text = widgets.Label(f"🔬", layout=Layout(width="20px", flex="1 0 20px"))
        upscaleII_check = widgets.Checkbox(
            value=False,
            description='II',
            indent=False,
            layout=Layout(max_width="38px")
        )
        upscaleII_check.observe(lambda e: self.add_upscaleII(e, seed), 'value', type='change')
        upscaleIII_check = widgets.Checkbox(
            value=False,
            description='III',
            indent=False
        )
        spacer = HBox([], layout=Layout(flex="1 0 auto"))
        upscaleIII_check.observe(lambda e: self.add_upscaleIII(e, seed), 'value', type='change')
        upscale_sr_button = widgets.Button(
            description="SR",
            tooltip="Send to Super Resolution",
            style={"button_color": "#212121"}
        )

        def on_send_to_sr_click(button):
            image_bytes = image_result
            parameters = self.get_ui_parameters()
            self.send_to_sr(image_bytes, parameters)

        upscale_sr_button.on_click(on_send_to_sr_click)

        upscale_box = HBox([upscale_text, upscaleII_check, upscaleIII_check, spacer, upscale_sr_button],
                           layout=Layout(max_width=f"calc({size[0]}px + 10px)"))

        result_box = VBox([top_box, image_view, upscale_box], layout=Layout(max_width=f"calc({size[0]}px + 10px)"))

        self.result_box.layout.display = "flex"
        self.result_button_box.layout.display = "flex"
        self.stageI_results_label.layout.display = "block"
        self.result_box.children += (result_box,)

    def add_upscaleII(self, e, seed):
        if e["name"] == "value" and e["new"]:
            self.upscaleII.append(seed)
        elif e["name"] == "value" and not e["new"]:
            self.upscaleII.remove(seed)

    def add_upscaleIII(self, e, seed):
        if e["name"] == "value" and e["new"]:
            self.upscaleIII.append(seed)
        elif e["name"] == "value" and not e["new"]:
            self.upscaleIII.remove(seed)

    def on_before_upscale(self):
        self.set_status_computing()
        self.show_progress_bar()

        if self.upscaling and self.upscaling_stage == "III":
            self.stop_stageIII_progress()
            self.upscaling_progress_event = threading.Event()
            self.upscaling_progress_thread = Thread(target=self.stageIII_mock_progress)
            self.upscaling_progress_thread.start()

    def on_before_checkpoints_loaded(self, missing):
        if missing:
            self.status_message(DOWNLOADING_CHECKPOINTS)
        else:
            self.status_message(LOADING_CHECKPOINTS)

        self.set_status_waiting()

    def on_checkpoints_loaded(self):
        pass

    @catch_handler_errors
    def generate_upscales(self):
        error = False

        try:
            self.upscaling = True
            generations = {}
            upscales = {}

            for seed in self.resultsI.keys():
                if seed in self.upscaleIII:
                    generations[seed] = "III"
                elif seed in self.upscaleII:
                    generations[seed] = "II"
                else:
                    generations[seed] = None

            for (seed, stage) in generations.items():
                if stage:
                    upscales[seed] = stage

            total_images = len(self.upscaleIII) * 2 + len(set(self.upscaleII) - set(self.upscaleIII))
            i = 0

            self.stop_upscale = False
            for (seed, stage) in upscales.items():
                if self.stop_upscale:
                    break

                if self.pipeline.stages.sequential_load == SEQ_LOAD_SEPARATE:
                    self.generate_upscale(seed, "II", stage, (i := i + 1), total_images)
                else:
                    if stage == "II":
                        self.generate_upscale(seed, "II", stage, (i := i + 1), total_images)
                    elif stage == "III":
                        self.generate_upscale(seed, "II", stage, (i := i + 1), total_images)
                        self.generate_upscale(seed, "III", stage, (i := i + 1), total_images)

            if not self.stop_upscale and self.pipeline.stages.sequential_load == SEQ_LOAD_SEPARATE:
                for (i, seed) in enumerate(self.upscaleIII):
                    if self.stop_upscale:
                        break
                    self.generate_upscale(seed, "III", "III", (i := i + 1), total_images)

            sII_time = f"Stage II: ~{self.stageII_time}s"
            sIII_time = f"Stage III: ~{self.stageIII_time}s"
            result_time = sII_time if self.stageIII_time == 0 else sII_time + ", " + sIII_time
            self.status_message(result_time)

        except ModelError as e:
            error = True
            self.status_message(str(e))
        except MemoryError as e:
            error = True
            self.status_message("Memory error, please restart.")
        finally:
            self.reset_progress()
            if error:
                self.set_status_error()
            else:
                self.set_status_result()
            self.generation_thread = None
            self.upscale_button.description = self.UPSCALE_BUTTON_LABEL
            self.upscale_button2.description = self.UPSCALE_BUTTON_LABEL

    def generate_upscale(self, seed, stage, stage_max, image_index=None, total_images=None):
        self.upscaling_stage = stage
        self.upscaling_stage_max = stage_max

        if stage == "II":
            self.stop_stageIII_progress()

            resultI = self.resultsI[seed]
            result = self.pipeline.upscale(resultI=resultI, progress=self.update_progress)

            self.stageII_time = round(result.duration)
            self.upscale_resultsII[seed] = result
            self.process_upscale_result(seed, result, "II", stage_max, image_index, total_images)

        elif stage == "III":
            resultI = self.resultsI[seed]
            resultII = self.upscale_resultsII[seed]
            result = self.pipeline.upscale(resultI=resultI, resultII=resultII, progress=self.update_progress)

            self.stageIII_time = round(result.duration)
            self.stageIII_iter_time = result.duration / self.pipeline.iterationsIII
            self.process_upscale_result(seed, result, "III", stage_max, image_index, total_images)

    def process_upscale_result(self, seed, result, stage, stage_max=None, image_index=None, total_images=None):
        image = result.images[stage][0]
        self.save_result(image, result.time, seed, stage)

        stage_index = 1 if stage == "II" else 2
        nsfw = self._get_nsfw_status(result, stage_index) if DEBUG else ""
        seed_text = widgets.Label(f"Seed: {result.seed} ({stage}){nsfw}")
        spacer = HBox([], layout=Layout(flex="1 0 auto"))
        image_index = f"{image_index}/{total_images}" if image_index is not None and total_images is not None else ""
        image_index_label = widgets.Label(image_index)
        image_header = HBox([seed_text, spacer, image_index_label])

        if DEBUG:
            image_view = widgets.Image(
                value=self._image_to_bytes(image),
                format='png',
                layout=Layout(width="max-content")
            )
        else:
            file_name = self._get_file_name(result.time, seed, stage)
            image_view = widgets.HTML(f"<img src='/files/{file_name}' style='width: max-content'/>")

        spacer = HBox([], layout=Layout(flex="1 0 auto"))
        generateIII_button = widgets.Button(
            description="III",
            tooltip="Generate stage III",
            layout=Layout(width="30px"),
            style={"button_color": "#212121"}
        )

        def generate_stageIII(button):
            if self.generation_thread is None:
                try:
                    generateIII_button.description = "⏳"
                    self.generation_thread = True
                    self.generate_upscale(seed, "III", "III")
                    generateIII_button.layout.display = "none"
                    self.status_message(f"Stage III: ~{self.stageIII_time}s")
                finally:
                    self.generation_thread = None

        generateIII_button.on_click(generate_stageIII)
        result_footer = HBox([spacer, generateIII_button])

        result_box = VBox([image_header, image_view], layout=Layout(max_width="max-content"))
        if stage_max == "II":
            result_box.children += (result_footer,)
        hr = widgets.HTML("<hr class='iflab-upscale-separator'>")

        upscale_result_box = self.upscale_result_boxes.get(seed, None)

        if upscale_result_box:
            upscale_result_box.children += (result_box,)
        else:
            upscale_result_box = VBox([])
            upscale_result_box.children += (result_box,) if image_index == 1 else (hr, result_box,)
            self.upscale_result_boxes[seed] = upscale_result_box
            self.upscale_box.children += (upscale_result_box,)

        self.upscale_box.layout.display = "flex"
        self.upscale_button_box.layout.display = "flex"
        self.upscale_results_label.layout.display = "block"

    # deeper patching is required to hook into the stage III progress, so here is a mock now
    def stageIII_mock_progress(self):
        self.progress_bar.max = self.pipeline.iterationsIII - 1
        wait_time = 1 if self.stageIII_iter_time == 0 else self.stageIII_iter_time

        for i in range(0, self.pipeline.iterationsIII):
            self.progress_bar.value = i
            if self.upscaling_progress_event and self.upscaling_progress_event.wait(wait_time):
                self.upscaling_progress_event = None
                break

    def stop_stageIII_progress(self):
        if self.upscaling_progress_event:
            try:
                self.upscaling_progress_event.set()
            except:
                traceback.print_exc()

    def clear_results(self, button):
        self.resultsI = {}
        self.result_box.children = []
        self.result_box.layout.display = "none"
        self.result_button_box.layout.display = "none"
        self.stageI_results_label.layout.display = "none"

    def clear_upscales(self, button):
        self.upscale_box.children = []
        self.upscale_box.layout.display = "none"
        self.upscale_button_box.layout.display = "none"
        self.upscale_results_label.layout.display = "none"

    def save_result(self, image, time, seed, stage):
        file_name = self._get_file_name(time, seed, stage)
        file_dir = os.path.dirname(file_name)

        p = Path(file_dir)
        p.mkdir(parents=True, exist_ok=True)

        with open(file_name, "wb") as f:
            image.save(f, format='png', pnginfo=self.get_result_pnginfo(seed, stage))

    def get(self):
        return self.root_box

    def settings_changed(self, new_settings):
        self.settings = new_settings

    class GeneratorFacade:
        def __init__(self, ui, options):
            self.ui = ui
            self.options = options

        def __call__(self, *args, **kwargs):
            return self.ui.invoke_pipeline(kwargs, self.options)

    def get_pipeline(self, **kwargs):
        return PipelineUI.GeneratorFacade(self, kwargs)

    def invoke_pipeline(self, kwargs, options):
        update_ui = options.get("update_ui", False)
        reference_pipeline = options.get("reference", False)
        return_tensors = kwargs.get("return_tensors", False)

        seed = self.seed_number.value
        seed = seed if seed > 0 else None

        self.setup_pipeline(kwargs)
        self.compute_embeddings()
        result = self.pipeline.generate(seed=seed, progress=True, is_reference=reference_pipeline)
        self.resultsI[seed] = result

        if update_ui:
            self.process_stageI_result(result)

        upscale = "if_II_kwargs" in kwargs or "if_III_kwargs" in kwargs
        stage = "III" if "if_III_kwargs" in kwargs else "II"

        if upscale:
            if stage == "II":
                result = self.pipeline.upscale(resultI=result, progress=True, is_reference=reference_pipeline)
                self.upscale_resultsII[seed] = result
                if update_ui:
                    self.process_upscale_result(seed, result, "II", "II")
            elif stage == "III":
                resultII = self.pipeline.upscale(resultI=result, progress=True, is_reference=reference_pipeline)
                self.upscale_resultsII[seed] = result
                if update_ui:
                    self.process_upscale_result(seed, resultII, "II", "III")
                result = self.pipeline.upscale(resultI=result, resultII=resultII, progress=True,
                                               is_reference=reference_pipeline)
                if update_ui:
                    self.process_upscale_result(seed, result, "III", "III")

        if "output" in result.images:
            del result.images["output"]

        return (result.images, result.tensors) if return_tensors else result.images

    @abstractmethod
    def _tune_ui(self):
        pass

    @abstractmethod
    def get_title(self):
        pass
