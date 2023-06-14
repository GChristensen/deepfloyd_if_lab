import os
import sys
import site
import json
import shlex
import shutil
import argparse
import traceback

from pathlib import Path

from iflab.ui.open import open_ui

has_windows = sys.platform == "win32"
has_linux = sys.platform == "linux"
has_mac = sys.platform == "darwin"

parser = argparse.ArgumentParser(description='DeepFloyd IF Lab')
parser.add_argument('--device', dest='device', default="cuda:0")
parser.add_argument('--t5-dtype', dest='t5_dtype', default=None)
parser.add_argument('--t5-on-gpu', dest='t5_on_gpu', action='store_true', default=False)
parser.add_argument('--lowram', dest='lowram', action='store_true', default=False)
parser.add_argument('--vram-fraction', dest='vram_fraction', default=None)
parser.add_argument('--force-update', dest='force_update', action='store_true', default=False)
parser.add_argument('--auto-update', dest='auto_update', action='store_true', default=True)
parser.add_argument('--skip-update', dest='skip_update', action='store_true', default=False)
parser.add_argument('--remote-access', dest='remote_access', action='store_true', default=False)
parser.add_argument('--no-remote-access', dest='remote_access', action='store_false', default=False)
parser.add_argument('--open-browser', dest='open_browser', action='store_true', default=True)
parser.add_argument('--no-browser', dest='open_browser', action='store_false')
parser.add_argument('--output-dir', dest='output_dir', default='outputs')
parser.add_argument('--debug', dest='debug', action='store_true', default=False)
parser.add_argument('--experimental', dest='experimental', action='store_true', default=False)
parser.add_argument('--port', dest='tcp_port', default="18888")

if "--debug" in sys.argv:
    $IFLAB_DEBUG = "1"

if os.path.exists("user.xsh"):
    try:
        import user
        if "USER_ARGS" in ${...}:
            user_args = shlex.split($USER_ARGS)
            sys.argv += user_args
    except:
        traceback.print_exc()

try:
    from pyadl import ADLManager
    has_amd = len(ADLManager.getInstance().getDevices()) > 0
except:
    has_amd = False

args = parser.parse_args()

if has_windows:
    $PYTHON = "python"
else:
    $PYTHON = "python3"

$HF_HOME = $IFLAB_HOME + "/home/huggingface"

if "USERPROFILE" in ${...}:
    $IFLAB_TRUE_USERPROFILE = $USERPROFILE

if "HOME" in ${...}:
    $IFLAB_TRUE_HOME = $HOME

$HOME = $IFLAB_HOME + "/home"
$USERPROFILE=$HOME
$PIP_CACHE_DIR = $HOME + "/pip/cache"
$PYTHONPATH=$IFLAB_HOME + "/modules"

if args.experimental:
    $IFLAB_EXPERIMENTAL = "1"

if args.debug:
    $IFLAB_DEBUG = "1"

if args.remote_access:
    $IFLAB_REMOTE_ACCESS = "1"

if args.device:
    $IFLAB_DEVICE = args.device

if args.t5_dtype:
    $IFLAB_T5_DTYPE = args.t5_dtype

if args.t5_on_gpu:
    $IFLAB_T5_ON_GPU = args.t5_on_gpu

if args.lowram:
    $IFLAB_LOWRAM = "1"

if args.vram_fraction:
    $IFLAB_VRAM_FRACTION = args.vram_fraction

if has_mac:
    $PYTORCH_ENABLE_MPS_FALLBACK=1

$IFLAB_OUTPUT_DIR = args.output_dir

if "IFLAB_FORCE_UPDATE" in ${...}:
    args.force_update = True

if args.auto_update and not args.skip_update:
    if args.force_update or "IFLAB_NEW_VERSION" in ${...}:
        requirements_file = Path("requirements.txt")

        if has_mac:
            requirements = requirements_file.read_text()
            requirements = requirements.replace("+cu118", "")
            requirements_file = Path("requirements_mac.txt")
            requirements_file.write_text(requirements)

            $[mkdir -p modules/iflab/pipelines/mac]
            $[pushd modules/iflab/pipelines/mac]
            $[curl -OL "https://github.com/GChristensen/deepfloyd_if_lab_mac/releases/latest/download/mac.zip"]
            $[unzip -o mac.zip -d ./]
            $[rm -f mac.zip]
            $[popd]

        if has_amd:
            requirements = requirements_file.read_text()
            requirements = requirements.replace("cu118", "rocm5.4.2")
            requirements_file = Path("requirements_amd.txt")
            requirements_file.write_text(requirements)

        if not has_windows:
            $TMPDIR = $HOME + "/tmp"
            $[mkdir $TMPDIR]

        $[pip install -r @(str(requirements_file))]
        $[pip install deepfloyd_if==1.0.2rc0 --no-deps --force-reinstall]

        libpath = next(filter(lambda s: "packages" in s, site.getsitepackages()), "")
        $[pushd @(libpath)]
        $[$PYTHON @(libpath)/patch_ng.py $IFLAB_HOME/patches/hf_hub.patch]
        $[$PYTHON @(libpath)/patch_ng.py $IFLAB_HOME/patches/deepfloyd_if.patch]
        $[popd]

        if not os.path.exists("notebooks"):
            os.makedirs("notebooks")

        print("Preparing Python libraries, please wait...")
        $[pushd $IFLAB_HOME/notebooks]
        from iflab.init import init_installation
        init_installation()
        $[popd]

        if not os.path.exists("home/.jupyter"):
            shutil.copytree("home/default/.jupyter", "home/.jupyter")

        if not os.path.exists("notebooks/deepfloyd-if-lab.ipynb"):
            shutil.copy("home/default/deepfloyd-if-lab.ipynb", "notebooks")

if args.open_browser:
    open_ui(args.tcp_port)

$[jupyter labextension disable "@jupyterlab/apputils-extension:announcements"]
$[jupyter lab --config=$IFLAB_HOME/home/.jupyter/lab/jupyter_lab_config.py --port=@(args.tcp_port)]
