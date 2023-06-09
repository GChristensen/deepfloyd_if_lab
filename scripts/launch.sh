cd "$( dirname -- "$0"; )/.."
export IFLAB_HOME=$( pwd; )
export PYTHONPATH=$IFLAB_HOME/modules

if [ ! -d "venv" ]
then

  if [ "$(uname)" == "Darwin" ]; then
      export PYTHON=python3.10
  else
      export PYTHON=python3
  fi

  $PYTHON -m ensurepip --upgrade
  $PYTHON -m venv venv
  $PYTHON -m pip install --upgrade pip
  source ./venv/bin/activate
  pip install "xonsh[full]"

  export IFLAB_FORCE_UPDATE=1
else
  source venv/bin/activate
fi

xonsh scripts/notebook.xsh ${COMMANDLINE_ARGS}
