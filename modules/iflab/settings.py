import os

from easysettings import JSONSettings

SETTINGS_FILE = "../home/settings.json"

settings = JSONSettings(filename=SETTINGS_FILE)

if os.path.exists(SETTINGS_FILE):
    settings.load()
