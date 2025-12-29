$OUT_DIR = (Join-Path $PSScriptRoot "../models")
if (-not (Test-Path -Path $OUT_DIR)) {
  New-Item -ItemType Directory -Path $OUT_DIR | Out-Null
}
Set-Location -Path $OUT_DIR
try {
  if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Throw "Git is not installed or not found in PATH."
  }

  $out_dir = ".\gemma-3-1b-it"
  if (-not (Test-Path -Path $out_dir)) {
    git clone `
      "https://huggingface.co/google/gemma-3-1b-it" `
      $out_dir
  }

  $out_dir = ".\sail-vl2-2b"
  if (-not (Test-Path -Path $out_dir)) {
    git clone `
      "https://huggingface.co/BytedanceDouyinContent/SAIL-VL2-2B" `
      $out_dir
  }

  $out_dir = ".\granite-3.3-2b-instruct"
  if (-not (Test-Path -Path $out_dir)) {
    git clone `
      "https://huggingface.co/ibm-granite/granite-3.3-2b-instruct" `
      $out_dir
  }
}
catch {
  Write-Error $_.ScriptStackTrace
  exit 1
}
finally { Pop-Location }
exit 0
