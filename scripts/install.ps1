# Cross-platform asennusskripti Theia AI -kehitysympäristölle

# Tarkista admin-oikeudet
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Tarvitaan admin-oikeudet. Käynnistä PowerShell admin-oikeuksilla." -ForegroundColor Red
    exit 1
}

# Funktio GPU:n tarkistamiseen
function Get-GPUInfo {
    $gpu = Get-WmiObject Win32_VideoController
    if ($gpu.Name -match "NVIDIA") {
        return "NVIDIA"
    }
    elseif ($gpu.Name -match "AMD") {
        return "AMD"
    }
    return "CPU"
}

# Funktio Python-version tarkistamiseen
function Test-PythonVersion {
    try {
        $version = python --version 2>&1
        if ($version -match "Python 3\.(9|10|11|12|13)") {
            return $true
        }
    }
    catch {
        return $false
    }
    return $false
}

# Funktio Node.js version tarkistamiseen
function Test-NodeVersion {
    try {
        $version = node --version
        if ($version -match "v(16|18|20)") {
            return $true
        }
    }
    catch {
        return $false
    }
    return $false
}

Write-Host "Theia AI -kehitysympäristön asennus" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Tarkista Python
Write-Host "Tarkistetaan Python..." -ForegroundColor Yellow
if (-not (Test-PythonVersion)) {
    Write-Host "Python 3.9+ puuttuu. Asennetaan..." -ForegroundColor Red
    winget install Python.Python.3.13
}

# Tarkista Node.js
Write-Host "Tarkistetaan Node.js..." -ForegroundColor Yellow
if (-not (Test-NodeVersion)) {
    Write-Host "Node.js v16+ puuttuu. Asennetaan..." -ForegroundColor Red
    winget install OpenJS.NodeJS.LTS
}

# Tarkista GPU
Write-Host "Tarkistetaan GPU..." -ForegroundColor Yellow
$gpuType = Get-GPUInfo
Write-Host "Löydetty GPU: $gpuType" -ForegroundColor Green

# Luo virtuaaliympäristö
Write-Host "Luodaan Python virtuaaliympäristö..." -ForegroundColor Yellow
python -m venv venv
. .\venv\Scripts\activate

# Asenna Python-riippuvuudet
Write-Host "Asennetaan Python-riippuvuudet..." -ForegroundColor Yellow
python -m pip install --upgrade pip
if ($gpuType -eq "NVIDIA") {
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}
else {
    pip install torch torchvision torchaudio
}

# Asenna kehitystyökalut
pip install transformers accelerate bitsandbytes
pip install black ruff mypy isort
pip install python-dotenv python-lsp-server[all] pylsp-mypy python-lsp-ruff
pip install jupyter ipykernel

# Konfiguroi pre-commit
Write-Host "Konfiguroidaan pre-commit..." -ForegroundColor Yellow
pip install pre-commit
pre-commit install

# Luo pre-commit konfiguraatio
$preCommitConfig = @"
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
    -   id: black
        args: [--line-length=100]

-   repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
    -   id: isort
        args: ["--profile", "black"]

-   repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.2.1
    hooks:
    -   id: ruff
        args: [--fix, --exit-non-zero-on-fix]

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
    -   id: mypy
        additional_dependencies: [types-all]
"@
$preCommitConfig | Out-File -FilePath ".pre-commit-config.yaml" -Encoding utf8

# Asenna Node.js riippuvuudet
Write-Host "Asennetaan Node.js riippuvuudet..." -ForegroundColor Yellow
npm install -g yarn
yarn install

# Kopioi ympäristömuuttujat
Write-Host "Kopioidaan ympäristömuuttujat..." -ForegroundColor Yellow
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Muista lisätä API-avaimet .env tiedostoon!" -ForegroundColor Red
}

# Lataa ja optimoi WizardCoder
Write-Host "Ladataan WizardCoder..." -ForegroundColor Yellow
python -c @"
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

# Lataa konfiguraatio
with open('model_config.json', 'r') as f:
    config = json.load(f)['model_settings']

model_name = config['model_name']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Käytetään laitetta: {device}')

# Määritä kvantisointiasetukset
if device == 'cpu':
    quant_config = config['optimization']['quantization']['cpu']
else:
    quant_config = config['optimization']['quantization']['gpu']

# Luo BitsAndBytes-konfiguraatio
bnb_config = BitsAndBytesConfig(
    load_in_4bit=quant_config['bits'] == 4,
    load_in_8bit=quant_config['bits'] == 8,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
)

# Määritä muistiasetukset
max_memory = config['optimization']['memory']['cpu_max_memory'] if device == 'cpu' else config['optimization']['memory']['gpu_max_memory']
batch_size = config['optimization']['batch_size']['cpu'] if device == 'cpu' else config['optimization']['batch_size']['gpu']

# Luo offload-kansio
os.makedirs('offload_cache', exist_ok=True)

print(f'Kvantisaatio: {quant_config["bits"]} bittiä')
print(f'Muistiraja: {max_memory}')
print(f'Eräkoko: {batch_size}')

# Lataa tokenizer ja malli optimoiduilla asetuksilla
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    model_max_length=config['context_length'],
    padding_side='left',
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device,
    max_memory={device: max_memory},
    offload_folder='offload_cache',
    trust_remote_code=True,
    torch_dtype=torch.float16 if device != 'cpu' else torch.float32
)

# Konfiguroi mallin generointiasetukset
model.config.max_length = config['context_length']
model.config.pad_token_id = tokenizer.pad_token_id
model.config.temperature = config['generation']['temperature']
model.config.top_p = config['generation']['top_p']
model.config.top_k = config['generation']['top_k']
model.config.repetition_penalty = config['generation']['repetition_penalty']
model.config.max_new_tokens = config['generation']['max_new_tokens']

# Tallenna optimoitu malli
output_dir = './models/wizardcoder'
os.makedirs(output_dir, exist_ok=True)

tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)

print('Malli ladattu ja optimoitu onnistuneesti!')
print(f'Kontekstin pituus: {config["context_length"]} tokenia')
print(f'Generointiasetukset:')
print(f'- Lämpötila: {config["generation"]["temperature"]}')
print(f'- Top-p: {config["generation"]["top_p"]}')
print(f'- Top-k: {config["generation"]["top_k"]}')
print(f'- Toiston esto: {config["generation"]["repetition_penalty"]}')
"@

Write-Host "Asennus valmis!" -ForegroundColor Green
Write-Host "Käynnistä Theia komennolla: yarn start" -ForegroundColor Cyan
