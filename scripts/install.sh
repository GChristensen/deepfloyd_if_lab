git init
git remote add origin https://github.com/GChristensen/deepfloyd_if_lab.git
git pull origin main
chmod +x ./*.sh
chmod +x ./**/*.sh

python3 -m ensurepip --upgrade
python3 -m venv venv
python3 -m pip install --upgrade pip
source ./venv/bin/activate
pip install "xonsh[full]"

./open-notebook.sh --force-update
