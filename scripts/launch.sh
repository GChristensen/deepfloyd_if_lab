cd "$( dirname -- "$0"; )/.."
export FLOYD_HOME=$( pwd; )
export PYTHONPATH=$FLOYD_HOME/modules

source venv/bin/activate

xonsh scripts/notebook.xsh ${COMMANDLINE_ARGS}
