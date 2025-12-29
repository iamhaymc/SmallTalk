Set-Location -Path (Join-Path $PSScriptRoot "..")
try {
  if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Throw "Python is not installed or not found in PATH."
  }
  If (-not (Test-Path -Path "pyenv" -PathType Container)) {
    Throw "Python virtual environment not found"
  }
  .\pyenv\Scripts\activate

  python -m unittest discover -v -s "." -p "*_test.py"
}
catch {
  Write-Error $_.ScriptStackTrace
  exit 1
}
finally { Pop-Location }
exit 0
