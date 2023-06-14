curl -OL https://github.com/GChristensen/deepfloyd_if_lab/releases/download/bin/bin.zip
tar -xf bin.zip
del /F /Q bin.zip

git init
git remote add origin https://github.com/GChristensen/deepfloyd_if_lab.git
git pull origin main

@set IFLAB_INSTALL=1

call open-notebook --force-update