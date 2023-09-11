apk add git || exit 1

python -m venv .env || exit 1
source .env/bin/activate || exit 1

pip install --upgrade pip || exit 1
pip install -r requirements.txt || exit 1

python run.py $@
