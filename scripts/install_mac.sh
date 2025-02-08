#!/bin/bash

# Värit
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Tarkista onko Homebrew asennettu
check_homebrew() {
    if ! command -v brew &> /dev/null; then
        echo -e "${RED}Homebrew ei ole asennettu. Asennetaan...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Lisää Homebrew PATH:iin M1/M2 Maceille
        if [[ $(uname -m) == 'arm64' ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    fi
}

# Tarkista Python-versio
check_python() {
    if ! command -v python3 &> /dev/null || [[ $(python3 -c 'import sys; print(sys.version_info[1])') -lt 11 ]]; then
        echo -e "${YELLOW}Asennetaan Python 3.11+...${NC}"
        brew install python@3.11
        brew link python@3.11
    fi
}

# Tarkista Node.js
check_node() {
    if ! command -v node &> /dev/null || [[ $(node -v | cut -d. -f1 | tr -d 'v') -lt 16 ]]; then
        echo -e "${YELLOW}Asennetaan Node.js...${NC}"
        brew install node@20
        brew link node@20
    fi
}

# Tarkista Yarn
check_yarn() {
    if ! command -v yarn &> /dev/null; then
        echo -e "${YELLOW}Asennetaan Yarn...${NC}"
        npm install -g yarn
    fi
}

# Tarkista Metal-tuki
check_metal() {
    if [[ $(uname -m) == 'arm64' ]]; then
        echo -e "${GREEN}M1/M2 Mac havaittu - Metal-tuki käytettävissä${NC}"
        return 0
    else
        echo -e "${YELLOW}Intel Mac havaittu - käytetään CPU-optimointia${NC}"
        return 1
    fi
}

# Asenna Theia-riippuvuudet
install_theia() {
    echo -e "${YELLOW}Asennetaan Theia-riippuvuudet...${NC}"
    
    # Asenna Node.js riippuvuudet
    yarn install
    
    # Käännä TypeScript
    yarn build
    
    # Lataa VSCode-laajennukset
    mkdir -p plugins
    cd plugins
    
    # Python-tuki
    curl -LO https://open-vsx.org/api/ms-python/python/latest/file/ms-python.python-latest.vsix
    
    # TypeScript-tuki
    curl -LO https://open-vsx.org/api/vscode/typescript-language-features/latest/file/vscode.typescript-language-features-latest.vsix
    
    # JSON-tuki
    curl -LO https://open-vsx.org/api/vscode/json-language-features/latest/file/vscode.json-language-features-latest.vsix
    
    # Git-tuki
    curl -LO https://open-vsx.org/api/vscode/git/latest/file/vscode.git-latest.vsix
    
    cd ..
}

# Pääfunktio
main() {
    echo -e "${CYAN}Theia AI -kehitysympäristön asennus (Mac)${NC}"
    echo -e "${CYAN}=====================================${NC}"

    # Tarkista ja asenna perustyökalut
    check_homebrew
    check_python
    check_node
    check_yarn

    # Luo Python virtuaaliympäristö
    echo -e "${YELLOW}Luodaan virtuaaliympäristö...${NC}"
    python3 -m venv venv
    source venv/bin/activate

    # Päivitä pip
    python3 -m pip install --upgrade pip

    # Tarkista Metal ja asenna PyTorch
    if check_metal; then
        echo -e "${YELLOW}Asennetaan PyTorch Metal-tuella...${NC}"
        pip3 install torch torchvision torchaudio
    else
        echo -e "${YELLOW}Asennetaan PyTorch CPU-optimoinnilla...${NC}"
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    # Asenna kehitystyökalut
    echo -e "${YELLOW}Asennetaan kehitystyökalut...${NC}"
    pip3 install transformers accelerate bitsandbytes
    pip3 install black ruff mypy isort
    pip3 install python-dotenv python-lsp-server[all] pylsp-mypy python-lsp-ruff
    pip3 install jupyter ipykernel

    # Asenna ja konfiguroi pre-commit
    echo -e "${YELLOW}Konfiguroidaan pre-commit...${NC}"
    pip3 install pre-commit
    pre-commit install

    # Asenna Theia ja sen riippuvuudet
    install_theia

    # Kopioi ympäristömuuttujat
    if [ ! -f .env ]; then
        cp .env.example .env
        echo -e "${RED}Muista lisätä API-avaimet .env tiedostoon!${NC}"
    fi

    # Testaa Metal-tuki
    echo -e "${YELLOW}Testataan Metal-tuki...${NC}"
    python3 test_metal.py

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
    print("Käytetään Metal-kiihdytystä")
else:
    device = "cpu"
    print("Käytetään CPU:ta")

# Määritä kvantisointiasetukset
if device == "cpu":
    quant_config = config["optimization"]["quantization"]["cpu"]
else:
    quant_config = config["optimization"]["quantization"]["gpu"]
    if device == "mps":
        device = "cpu"  # MPS ei tue kvantisointia

# Luo BitsAndBytes-konfiguraatio
bnb_config = BitsAndBytesConfig(
    load_in_4bit=quant_config["bits"] == 4,
    load_in_8bit=quant_config["bits"] == 8,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Määritä muistiasetukset
max_memory = config["optimization"]["memory"]["cpu_max_memory"]
batch_size = config["optimization"]["batch_size"]["cpu"]

print(f"Kvantisaatio: {quant_config[\'bits\']} bittiä")
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

# Tallenna optimoitu malli
output_dir = "./models/wizardcoder"
os.makedirs(output_dir, exist_ok=True)

tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)

print("Malli ladattu ja optimoitu onnistuneesti!")
'

    echo -e "${GREEN}Asennus valmis!${NC}"
    echo -e "${CYAN}Käynnistä Theia komennolla: yarn start${NC}"
}

# Suorita pääfunktio
main
