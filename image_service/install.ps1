$ErrorActionPreference = 'Stop'
$imagePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $imagePython)) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv --python 3.12 (Join-Path $PSScriptRoot '.venv')
    } else {
        py -3.12 -m venv (Join-Path $PSScriptRoot '.venv')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment creation failed' }
}
& $imagePython -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'pip initialization failed' }
& $imagePython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
& $imagePython -m pip install 'torch==2.14.0+cu130' --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
& $imagePython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Inference dependencies installation failed' }
& $imagePython -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed' }

