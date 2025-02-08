#!/bin/bash

# Tarkista Python-versio
if ! command -v python3 &> /dev/null; then
    echo "Python 3 ei ole asennettu. Asennetaan Homebrew ja Python..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install python3
else
    echo "Python 3 löytyy järjestelmästä"
fi

# Tarkista pip
if ! command -v pip3 &> /dev/null; then
    echo "pip3 ei ole asennettu. Asennetaan..."
    python3 -m ensurepip --upgrade
else
    echo "pip3 löytyy järjestelmästä"
fi

# Luo virtuaaliympäristö
echo "Luodaan virtuaaliympäristö..."
python3 -m venv venv

# Aktivoi virtuaaliympäristö
echo "Aktivoidaan virtuaaliympäristö..."
source venv/bin/activate

# Päivitä pip
echo "Päivitetään pip..."
pip install --upgrade pip

# Asenna riippuvuudet
echo "Asennetaan riippuvuudet..."
pip install -r requirements.txt

# Luo tarvittavat hakemistot
echo "Luodaan hakemistot..."
mkdir -p logs
mkdir -p cache
mkdir -p data
mkdir -p temp

# Kopioi ympäristömuuttujat
if [ ! -f .env ]; then
    echo "Kopioidaan .env.example -> .env"
    cp .env.example .env
    echo "Muista päivittää API-avaimet .env-tiedostoon!"
fi

# Tarkista oikeudet
echo "Tarkistetaan tiedosto-oikeudet..."
chmod +x *.py
chmod 600 .env

echo "Asennus valmis! Voit käynnistää sovelluksen komennolla:"
echo "source venv/bin/activate && python main.py"
