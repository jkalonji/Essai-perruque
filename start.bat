@echo off
cd /d "%~dp0webapp"
echo Lancement du serveur local sur http://localhost:8000 ...
start "" http://localhost:8000
python -m http.server 8000
