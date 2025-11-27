@echo off
setlocal
if not exist .venv (
    echo 仮想環境 .venv を作成します...
    python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt
python main.py %*
endlocal
