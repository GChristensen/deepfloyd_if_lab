import socket
import subprocess
import time
import os
import sys
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

        if sys.platform  == "win32":
            os.startfile(url)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", url])
        else:
            try:
                subprocess.Popen(["xdg-open", url])
            except OSError:
                print("Please open a browser on: " + url)

def open_ui(port):
    Thread(target=lambda: open_ui_thread(port)).start()
