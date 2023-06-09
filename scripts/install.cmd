set path=%~dp0/bin/git/cmd;%~dp0/venv/Scripts;%~dp0/bin/python;%path%

curl -OL https://github.com/GChristensen/deepfloyd_if_lab/releases/download/bin/bin.zip
tar -xf bin.zip
del /F /Q bin.zip

git init
git remote add origin https://github.com/GChristensen/deepfloyd_if_lab.git
git pull origin main
attrib +h notebooks/deepfloyd-if-lab-dev.ipynb
attrib +h open-notebook-dev.cmd

call open-notebook --force-update