---
title: SmallTalk
short_description: Language model inference
license: mit
pinned: true
emoji: 💎
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: 5.47.2
python_version: 3.10
app_file: app.py
suggested_hardware: a10g-small
suggested_storage: small
---

## Install
### Windows
```powershell
python -m venv pyenv
.\pyenv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.dev.txt -r requirements.txt
```
### Linux/Mac
```bash
python -m venv pyenv
source ./pyenv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.dev.txt -r requirements.txt
```

## Hardware
### Requirements:
- CPU Cores >= 6
- Memory >= ~16GB
- Video Memory >= 24GB
- Storage >= ~128GB

### GPU Options:
- `4090/3090 RTX`
- `a10g-small` *@$1.00/hr*

### LLM Options:
- **<= 8B** *w/ 16bit quantization*:
    - `ibm-granite/granite-3.3-8b-instruct`
      - Storage Size: 16.3 GB
    - `ibm-granite/granite-3.3-2b-instruct`
      - Storage Size: 5.07 GB
    - `joeddav/xlm-roberta-large-xnli`
      - Storage Size: 6.74 GB
    - `deepset/xlm-roberta-large-squad2`
      - Storage Size: 4.48 GB
- **<= 34B** *w/ 4bit quantization*:
    - `TheDrummer/Big-Tiger-Gemma-27B-v3`
      - Storage Size: 54.9 GB
