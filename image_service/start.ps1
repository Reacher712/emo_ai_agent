$ErrorActionPreference = 'Stop'
$imagePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $imagePython -m uvicorn server:app --app-dir $PSScriptRoot --host 127.0.0.1 --port 8001

