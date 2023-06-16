import os

VERSION = "0.2.5"

DEBUG = os.getenv("IFLAB_DEBUG", False)
DEBUG = not not (DEBUG and DEBUG != "0")
UI_THREADS = not DEBUG
EXPERIMENTAL = os.getenv("IFLAB_EXPERIMENTAL", False)
EXPERIMENTAL = not not (EXPERIMENTAL and EXPERIMENTAL != "0")

SEQ_LOAD_OFF = "off"
SEQ_LOAD_MERGE = "merge"
SEQ_LOAD_SEPARATE = "separate"

RESPACING_MODES = ['fast27', 'smart27', 'smart50', 'smart100', 'smart185', 'super27', 'super40', 'super100']

DEBUG_PROMPT = "ultra close-up color photo portrait of rainbow owl with deer horns in the woods"

if DEBUG:
    PROMPT_FILE = "../home/debug_prompt.txt"
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            DEBUG_PROMPT = f.read()

