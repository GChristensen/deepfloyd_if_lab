import socket
import subprocess
import time
import os
import sys
import traceback
from contextlib import closing
from threading import Thread


def port_online(port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(0.1)
        result = sock.connect_ex(("127.0.0.1", int(port)))
        if result == 0:
            return True
        else:
            return False


def wait_for_port(port):
    ctr = 6000

    while ctr > 0:
        if port_online(port):
            return True
        ctr -= 1
        time.sleep(1)

    return False


def open_ui_thread(port):
    if wait_for_port(port):
        time.sleep(3)

        url = f"http://localhost:{port}"

        if sys.platform == "win32":
            os.startfile(url)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", url])
        else:
            try:
                env_mod = os.environ.copy()
                ture_userprofile = os.getenv("IFLAB_TRUE_USERPROFILE", None)
                ture_home = os.getenv("IFLAB_TRUE_HOME", None)

                if ture_userprofile:
                    env_mod["USERPROFILE"] = ture_userprofile
                if ture_home:
                    env_mod["HOME"] = ture_home

                subprocess.Popen(["xdg-open", url], env=env_mod)

            except OSError:
                print("Please open a browser on: " + url)

def open_ui(port):
    Thread(target=lambda: open_ui_thread(port), daemon=True).start()
