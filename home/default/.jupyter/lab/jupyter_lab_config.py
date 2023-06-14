# Configuration file for lab.
import os

REMOTE_ACCESS = os.getenv("IFLAB_REMOTE_ACCESS")

c = get_config()

c.IdentityProvider.token = ''
c.NotebookApp.token = ''
c.PasswordIdentityProvider.password_required = False
c.ExtensionApp.open_browser = False
c.ServerApp.root_dir = "notebooks"

if REMOTE_ACCESS and REMOTE_ACCESS != "0":
    c.ServerApp.allow_origin = '*'
    c.ServerApp.ip = '0.0.0.0'
