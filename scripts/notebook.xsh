import os
import sys
import site
import json
import argparse
from urllib import request
from pathlib import Path

from iflab.const import VERSION as installed_version
from iflab.ui.open import open_ui

has_windows = sys.platform == "win32"
has_linux = sys.platform == "linux"
has_mac = sys.platform == "darwin"

try:
    from pyadl import ADLManager
    has_amd = len(ADLManager.getInstance().getDevices()) > 0
except:
    has_amd = False

GIT_API_URL = "https://api.github.com/repos/GChristensen/deepfloyd_if_lab"

parser = argparse.ArgumentParser(description='DeepFloyd IF Lab')
parser.add_argument('--t5-dtype', dest='t5_dtype', default=None)
parser.add_argument('--force-update', dest='force_update', action='store_true', default=False)
parser.add_argument('--auto-update', dest='auto_update', action='store_true', default=True)
parser.add_argument('--skip-update', dest='skip_update', action='store_true', default=False)
parser.add_argument('--no-remote-access', dest='remote_access', action='store_false', default=False)
parser.add_argument('--remote-access', dest='remote_access', action='store_true', default=False)
parser.add_argument('--open-browser', dest='open_browser', action='store_true', default=False)
parser.add_argument('--output-dir', dest='output_dir', default='outputs')
parser.add_argument('--debug', dest='debug', action='store_true', default=False)
parser.add_argument('--experimental', dest='experimental', action='store_true', default=False)
parser.add_argument('--port', dest='tcp_port', default="8888")

args = parser.parse_args()

if has_windows:
    $PYTHON = "python"
else:
    $PYTHON = "python3"

$PATH = [f"{$IFLAB_HOME}/venv/Scripts"] + $PATH
$PATH = [f"{$IFLAB_HOME}/bin/git/cmd"] + $PATH
$HF_HOME = $IFLAB_HOME + "/home/huggingface"

if "USERPROFILE" in ${...}:
    $IFLAB_TRUE_USERPROFILE = $USERPROFILE

if "HOME" in ${...}:
    $IFLAB_TRUE_HOME = $HOME

if args.open_browser:
    open_ui(args.tcp_port)

$HOME = $IFLAB_HOME + "/home"
$USERPROFILE=$HOME
$PIP_CACHE_DIR = $HOME + "/pip/cache"
$PYTHONPATH=$IFLAB_HOME + "/modules"

if "IFLAB_FORCE_UPDATE" in ${...}:
    args.force_update = True

if args.auto_update and not args.skip_update:
    remote_version = None

    if not args.force_update:
        response = request.urlopen(f"{GIT_API_URL}/releases/latest", timeout=5)
        json_text = response.read().decode('utf-8')
        release = json.loads(json_text)
        remote_version = release["tag_name"]

    if args.force_update or remote_version != installed_version:
        requirements_file = Path("requirements.txt")

        if has_mac:
            requirements = requirements_file.read_text()
            requirements = requirements.replace("+cu118", "")
            requirements_file = Path("requirements_mac.txt")
            requirements_file.write_text(requirements)

        if has_amd:
            requirements = requirements_file.read_text()
            requirements = requirements.replace("cu118", "rocm5.4.2")
            requirements_file = Path("requirements_amd.txt")
            requirements_file.write_text(requirements)

        if not has_windows:
            $TMPDIR = $HOME + "/tmp"
            $[mkdir $TMPDIR]

        $[git pull origin main]
        $[pip install -r @(str(requirements_file))]
        $[pip install deepfloyd_if==1.0.2rc0 --no-deps --force-reinstall]

        libpath = next(filter(lambda s: "packages" in s, site.getsitepackages()), "")
        $[pushd @(libpath)]
        $[$PYTHON @(libpath)/patch_ng.py $IFLAB_HOME/patches/hf_hub.patch]
        $[$PYTHON @(libpath)/patch_ng.py $IFLAB_HOME/patches/deepfloyd_if.patch]
        $[popd]

        from iflab.pipelines.utils import get_default_settings
        default_settings = json.dumps(get_default_settings())

        settings_file = Path("home/settings.json")
        if not os.path.exists(str(settings_file)):
            settings_file.write_text(default_settings)

if args.experimental:
    $IFLAB_EXPERIMENTAL = "1"

if args.debug:
    $IFLAB_DEBUG = "1"

if args.remote_access:
    $IFLAB_REMOTE_ACCESS = "1"

if args.t5_dtype:
    $IFLAB_T5_DTYPE = args.t5_dtype

$IFLAB_OUTPUT_DIR = args.output_dir

$[jupyter labextension disable "@jupyterlab/apputils-extension:announcements"]
$[jupyter lab --config=$IFLAB_HOME/home/.jupyter/lab/jupyter_lab_config.py --port=@(args.tcp_port)]
