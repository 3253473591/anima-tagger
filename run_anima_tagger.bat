@echo off
chcp 65001 >nul
call conda activate py311_env
set PYTHONUTF8=1
set PYTHONPATH=%~dp0src
python -m anima_tagger.app
