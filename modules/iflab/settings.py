import os

from easysettings import JSONSettings

SETTINGS_FILE = "../home/settings.json"

settings = JSONSettings(filename=SETTINGS_FILE)

if os.path.exists(SETTINGS_FILE):
    settings.load()

t5_dtype = os.getenv("IFLAB_T5_DTYPE")

if t5_dtype:
    settings["t5_dtype"] = t5_dtype
