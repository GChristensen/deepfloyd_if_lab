if [ "$(uname)" == "Darwin" ]; then
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  brew install python@3.10 git
fi

git init
git remote add origin https://github.com/GChristensen/deepfloyd_if_lab.git
git pull origin main
chmod +x ./*.sh
chmod +x ./**/*.sh

./open-notebook.sh --force-update
