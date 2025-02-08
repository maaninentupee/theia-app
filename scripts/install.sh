#!/bin/bash

# Cross-platform asennusskripti Theia AI -kehitysympäristölle

# Värikoodit
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Funktio Python-version tarkistamiseen
check_python() {
    if command -v python3 >/dev/null 2>&1; then
        python3 --version | grep -q "Python 3\.[9-13]"
        return $?
    fi
    return 1
}

# Funktio Node.js version tarkistamiseen
check_node() {
    if command -v node >/dev/null 2>&1; then
        node --version | grep -q "v\(16\|18\|20\)"
        return $?
    fi
    return 1
}

# Funktio GPU:n tarkistamiseen
check_gpu() {
    if [[ "$(uname)" == "Darwin" ]]; then
        if [[ "$(uname -m)" == "arm64" ]]; then
            echo "MPS"
        else
            echo "CPU"
        fi
    else
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "NVIDIA"
        else
            echo "CPU"
        fi
    fi
}

echo -e "${CYAN}Theia AI -kehitysympäristön asennus${NC}"
echo -e "${CYAN}=================================${NC}"

# Tarkista käyttöjärjestelmä ja asenna työkalut
if [[ "$(uname)" == "Darwin" ]]; then
    echo -e "${YELLOW}Tarkistetaan Homebrew...${NC}"
    if ! command -v brew >/dev/null 2>&1; then
        echo -e "${RED}Homebrew puuttuu. Asennetaan...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    # Tarkista Python
    echo -e "${YELLOW}Tarkistetaan Python...${NC}"
    if ! check_python; then
        echo -e "${RED}Python 3.9+ puuttuu. Asennetaan...${NC}"
        brew install python@3.13
    fi

    # Tarkista Node.js
    echo -e "${YELLOW}Tarkistetaan Node.js...${NC}"
    if ! check_node; then
        echo -e "${RED}Node.js v16+ puuttuu. Asennetaan...${NC}"
        brew install node@20
    fi
fi

# Tarkista GPU
echo -e "${YELLOW}Tarkistetaan GPU...${NC}"
GPU_TYPE=$(check_gpu)
echo -e "${GREEN}Löydetty GPU: $GPU_TYPE${NC}"

# Luo virtuaaliympäristö
echo -e "${YELLOW}Luodaan Python virtuaaliympäristö...${NC}"
python3 -m venv venv
source venv/bin/activate

# Asenna Python-riippuvuudet
echo -e "${YELLOW}Asennetaan Python-riippuvuudet...${NC}"
python3 -m pip install --upgrade pip

if [[ "$GPU_TYPE" == "MPS" ]]; then
    pip3 install torch torchvision torchaudio
elif [[ "$GPU_TYPE" == "NVIDIA" ]]; then
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

pip3 install transformers accelerate bitsandbytes black ruff mypy isort
pip3 install python-dotenv python-lsp-server[all] pylsp-mypy python-lsp-ruff
pip3 install jupyter ipykernel

# Konfiguroi pre-commit
echo -e "${YELLOW}Konfiguroidaan pre-commit...${NC}"
pip3 install pre-commit
pre-commit install

# Luo pre-commit konfiguraatio
cat > .pre-commit-config.yaml << 'EOL'
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
EOL

# Asenna Node.js riippuvuudet
echo -e "${YELLOW}Asennetaan Node.js riippuvuudet...${NC}"
npm install -g yarn
yarn install

# Kopioi ympäristömuuttujat
echo -e "${YELLOW}Kopioidaan ympäristömuuttujat...${NC}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${RED}Muista lisätä API-avaimet .env tiedostoon!${NC}"
fi

# Lataa ja optimoi WizardCoder
echo -e "${YELLOW}Ladataan WizardCoder...${NC}"
python3 -c '
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import platform
import os

# Lataa konfiguraatio
with open("model_config.json", "r") as f:
    config = json.load(f)["model_settings"]

model_name = config["model_name"]

# Määritä laite
if platform.system() == "Darwin" and platform.machine() == "arm64":
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Käytetään laitetta: {device}")

# Määritä kvantisointiasetukset
if device == "cpu":
    quant_config = config["optimization"]["quantization"]["cpu"]
elif device == "mps":
    quant_config = config["optimization"]["quantization"]["gpu"]
    device = "cpu"  # MPS ei tue kvantisointia, käytetään CPU:ta
else:
    quant_config = config["optimization"]["quantization"]["gpu"]

# Luo BitsAndBytes-konfiguraatio
bnb_config = BitsAndBytesConfig(
    load_in_4bit=quant_config["bits"] == 4,
    load_in_8bit=quant_config["bits"] == 8,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Määritä muistiasetukset
max_memory = config["optimization"]["memory"]["cpu_max_memory"] if device == "cpu" else config["optimization"]["memory"]["gpu_max_memory"]
batch_size = config["optimization"]["batch_size"]["cpu"] if device == "cpu" else config["optimization"]["batch_size"]["gpu"]

# Luo offload-kansio
os.makedirs("offload_cache", exist_ok=True)

print(f"Kvantisaatio: {quant_config['bits']} bittiä")
print(f"Muistiraja: {max_memory}")
print(f"Eräkoko: {batch_size}")

# Lataa tokenizer ja malli optimoiduilla asetuksilla
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    model_max_length=config["context_length"],
    padding_side="left",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device,
    max_memory={device: max_memory},
    offload_folder="offload_cache",
    trust_remote_code=True,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32
)

# Konfiguroi mallin generointiasetukset
model.config.max_length = config["context_length"]
model.config.pad_token_id = tokenizer.pad_token_id
model.config.temperature = config["generation"]["temperature"]
model.config.top_p = config["generation"]["top_p"]
model.config.top_k = config["generation"]["top_k"]
model.config.repetition_penalty = config["generation"]["repetition_penalty"]
model.config.max_new_tokens = config["generation"]["max_new_tokens"]

# Tallenna optimoitu malli
output_dir = "./models/wizardcoder"
os.makedirs(output_dir, exist_ok=True)

tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)

print("Malli ladattu ja optimoitu onnistuneesti!")
print(f"Kontekstin pituus: {config['context_length']} tokenia")
print(f"Generointiasetukset:")
print(f"- Lämpötila: {config['generation']['temperature']}")
print(f"- Top-p: {config['generation']['top_p']}")
print(f"- Top-k: {config['generation']['top_k']}")
print(f"- Toiston esto: {config['generation']['repetition_penalty']}")
'

echo -e "${GREEN}Asennus valmis!${NC}"
echo -e "${CYAN}Käynnistä Theia komennolla: yarn start${NC}"
