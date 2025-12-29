Set-Location -Path (Join-Path $PSScriptRoot "..")
try {
  if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Throw "Python is not installed or not found in PATH."
  }
  Write-Output "Ensuring python environment..."
  If (-not (Test-Path -Path "pyenv" -PathType Container)) {
    python -m venv pyenv
  }
  .\pyenv\Scripts\activate

  $requirements_file = "requirements.txt"
  $requirements_dev_file = "requirements-dev.txt"
  $requirements_lock_file = Join-Path ".devops" "requirements-lock.txt"

  Write-Output "Updating python package manager..."
  python -m pip install --upgrade pip

  Write-Output "Installing python packages..."
  python -m pip install -r $requirements_dev_file -r $requirements_file

  Write-Output "Creating python package lock..."
  python -m pip freeze > $requirements_lock_file
}
catch {
  Write-Error $_.ScriptStackTrace
  exit 1
}
finally { Pop-Location }
exit 0
