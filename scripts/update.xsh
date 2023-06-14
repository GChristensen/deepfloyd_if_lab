import argparse
import json
import os
import shlex
import sys
import traceback

from urllib import request

from iflab.const import VERSION as installed_version

if os.path.exists("user.xsh"):
    try:
        import user
        if "USER_ARGS" in ${...}:
            user_args = shlex.split($USER_ARGS)
            sys.argv += user_args
    except:
        traceback.print_exc()

parser = argparse.ArgumentParser()
parser.add_argument('--force-update', dest='force_update', action='store_true', default=False)
parser.add_argument('--auto-update', dest='auto_update', action='store_true', default=True)
parser.add_argument('--skip-update', dest='skip_update', action='store_true', default=False)
parser.add_argument('args', nargs=argparse.REMAINDER)

args, unknown = parser.parse_known_args()

if "IFLAB_FORCE_UPDATE" in ${...}:
    args.force_update = True

if args.auto_update and not args.skip_update:
    remote_version = None

    if not args.force_update:
        git_api_url = "https://api.github.com/repos/GChristensen/deepfloyd_if_lab"
        print("Checking for updates...")
        response = request.urlopen(f"{git_api_url}/releases/latest", timeout=10)
        json_text = response.read().decode('utf-8')
        release = json.loads(json_text)
        remote_version = release["tag_name"]

    if args.force_update or remote_version != installed_version:
        if not args.force_update:
            print(f"Updating to version {remote_version}")
        $IFLAB_NEW_VERSION = True
        pull_branch = $PULL_BRANCH if "PULL_BRANCH" in ${...} else "main"
        $[git pull origin @(pull_branch)]