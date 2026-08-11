@echo off
rem Local dev launcher: force the extension (and terminal) to run the 1.3.4
rem source in this repo, bypassing the broken pip-installed 1.2.0.
set "PYTHONPATH=D:\I\Sousaku\paritok-4b-v1"
python -B -m paritok.cli %*
