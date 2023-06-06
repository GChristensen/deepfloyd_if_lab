import sys
import site
import json
import argparse
from urllib import request

from iflab.const import VERSION as installed_version
from iflab.ui.open import open_ui

GIT_API_URL = "https://api.github.com/repos/GChristensen/deepfloyd_if_lab"

parser = argparse.ArgumentParser(description='DeepFloyd IF Lab')
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

if sys.platform == "win32":
    $PYTHON = "python"
else:
    $PYTHON = "python3"

$PATH = [f"{$FLOYD_HOME}/venv/Scripts"] + $PATH
$PATH = [f"{$FLOYD_HOME}/bin/git/cmd"] + $PATH
$HF_HOME = $FLOYD_HOME + "/home/huggingface"

$HOME = $FLOYD_HOME + "/home"
$USERPROFILE=$HOME
$PIP_CACHE_DIR = $HOME + "/pip/cache"
$PYTHONPATH=$FLOYD_HOME + "/modules"

update_file = f"{$FLOYD_HOME}/update"

if args.auto_update and not args.skip_update:
    remote_version = None

    if not args.force_update:
        response = request.urlopen(f"{GIT_API_URL}/releases/latest", timeout=5)
        json_text = response.read().decode('utf-8')
        release = json.loads(json_text)
        remote_version = release["tag_name"]

    if args.force_update or remote_version != installed_version:
        libpath = site.getsitepackages()[0]
        $TMPDIR = $HOME + "/tmp"
        $[mkdir $TMPDIR]
        $[git pull origin main]
        $[pip install -r requirements.txt]
        $[pip install deepfloyd_if==1.0.2rc0 --no-deps]
        $[pushd @(libpath)]
        $[$PYTHON @(libpath)/patch_ng.py $FLOYD_HOME/patches/hf_hub.patch]
        $[$PYTHON @(libpath)/patch_ng.py $FLOYD_HOME/patches/deepfloyd_if.patch]
        $[popd]

if args.experimental:
    $IFLAB_EXPERIMENTAL = "1"

if args.debug:
    $IFLAB_DEBUG = "1"

if args.remote_access:
    $IFLAB_REMOTE_ACCESS = "1"

if args.open_browser:
    open_ui(args.tcp_port)

$IFLAB_OUTPUT_DIR = args.output_dir

$[jupyter labextension disable "@jupyterlab/apputils-extension:announcements"]
$[jupyter lab --config=$FLOYD_HOME/home/.jupyter/lab/jupyter_lab_config.py --port=@(args.tcp_port)]
