$OUT_DIR = (Join-Path $PSScriptRoot "../models")
if (-not (Test-Path -Path $OUT_DIR)) {
  New-Item -ItemType Directory -Path $OUT_DIR | Out-Null
}
Set-Location -Path $OUT_DIR
try {
  if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Throw "Git is not installed or not found in PATH."
  }

  $out_dir = ".\gemma-3-26b-tiger"
  if (-not (Test-Path -Path $out_dir)) {
    git clone `
      "https://huggingface.co/TheDrummer/Big-Tiger-Gemma-27B-v3" `
      $out_dir
  }

  $out_dir = ".\sail-vl2-8b-think"
  if (-not (Test-Path -Path $out_dir)) {
    git clone `
      "https://huggingface.co/BytedanceDouyinContent/SAIL-VL2-8B-Thinking" `
      $out_dir
  }

  $out_dir = ".\granite-3.3-8b-instruct"
  if (-not (Test-Path -Path $out_dir)) {
    git clone `
      "https://huggingface.co/ibm-granite/granite-3.3-8b-instruct" `
      $out_dir
  }
}
catch {
  Write-Error $_.ScriptStackTrace
  exit 1
}
finally { Pop-Location }
exit 0
