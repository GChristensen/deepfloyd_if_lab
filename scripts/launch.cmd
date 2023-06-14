cd %~dp0/..
@set IFLAB_HOME=%cd%
@set PATH=%IFLAB_HOME%/bin/git/cmd;%IFLAB_HOME%/venv/Scripts;%IFLAB_HOME%/bin/python;%PATH%
@set PIP_CACHE_DIR=%IFLAB_HOME%/home/pip/cache
@set PYTHONPATH=%IFLAB_HOME%/modules

@if exist venv\ goto HASVENV

python -m venv venv
python -m pip install --upgrade pip
call venv\scripts\activate
pip install "xonsh[full]"

set IFLAB_FORCE_UPDATE=1

:HASVENV

call .\venv\scripts\activate

xonsh scripts/update.xsh %*
xonsh scripts/notebook.xsh %*