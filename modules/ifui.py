import sys

from IPython.display import clear_output
import os


def unload_modules(of):
    if_modules = []
    for n, m in sys.modules.items():
        if hasattr(m, "__file__") and m.__file__ and of in m.__file__:
            if_modules.append(n)

    for n in if_modules:
        sys.modules.pop(n)


def show_ui():
    global stages
    print("Loading DeepFloyd IF Lab UI, please wait...")

    unload_modules("iflab")

    from iflab.pipelines.stages import DeepFloydIFStages
    from iflab.ui.main import DeepFloydIFUI

    try:
        stages
    except NameError:
        stages = DeepFloydIFStages()
        stages.load()

    clear_output()

    ui = DeepFloydIFUI(stages)
    ui.show()

    return ui
