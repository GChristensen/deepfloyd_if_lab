cd %~dp0/..
set FLOYD_HOME=%cd%
set PYTHONPATH=%FLOYD_HOME%/modules
set PATH=%FLOYD_HOME%/venv/Scripts;%PATH%

call venv/scripts/activate

xonsh scripts/notebook.xsh %* %COMMANDLINE_ARGS%