# Magic Jim's Personal Website Theme

A beautiful and modern WordPress theme for Magic Jim's personal website.

## Features
- Responsive design with mobile menu
- Customizable sections through WordPress Customizer
- Social media integration
- Portfolio section
- Contact form support
- Smooth scroll animations
- SEO friendly

## Theme Structure
- `style.css`: Theme styles and WordPress theme information
- `index.php`: Main template file
- `header.php`: Header template with responsive navigation
- `footer.php`: Footer template with social media links
- `functions.php`: Theme setup and customizer options
- `script.js`: Interactive features and animations

## Installation
1. Upload the theme folder to your WordPress themes directory
2. Activate the theme through WordPress admin panel
3. Customize the theme settings in WordPress Customizer

## Customization
The theme can be customized through WordPress Customizer:
- Site title and description
- About section content
- Portfolio settings
- Social media links
- Footer information

## Development
This theme was developed using modern web technologies:
- WordPress theme development best practices
- Responsive CSS with mobile-first approach
- jQuery for smooth animations
- PHP for WordPress integration

## API-avainten hallinta

## VAROITUS: Väliaikainen kehitysympäristön ratkaisu

Tämä projekti käyttää väliaikaista ratkaisua API-avainten hallintaan. **ÄLÄ käytä tätä ratkaisua tuotantoympäristössä!**

### Nykyinen toteutus (väliaikainen)

API-avaimet on tallennettu suoraan `api_keys.py` tiedostoon kehitystyön helpottamiseksi. Tämä ratkaisu on käytössä vain kunnes .env-tiedostojen tuki palautuu.

```python
API_KEYS = {
    "openai": "sk-...",
    "huggingface": "hf_...",
    "gpt4_model": "gpt-4",
    ...
}
```

### Siirtyminen .env-tiedostoon

Kun .env-tiedostojen tuki palautuu, siirrä API-avaimet `.env` tiedostoon seuraavassa muodossa:

```env
# API Avaimet
OPENAI_API_KEY=sk-...
HUGGINGFACE_API_KEY=hf_...

# Mallien tunnukset
GPT4_MODEL=gpt-4
GPT35_MODEL=gpt-3.5-turbo
STARCODER_MODEL=bigcode/starcoder
```

### Turvallisuussuositukset

1. **ÄLÄ jaa API-avaimia**: Pidä avaimet aina salassa
2. **ÄLÄ commitoi avaimia**: Lisää `api_keys.py` .gitignore tiedostoon
3. **Käytä ympäristömuuttujia**: Siirry .env käyttöön heti kun mahdollista

### API-avainten käyttö koodissa

```python
from api_keys import key_manager

# Hae API-avain
openai_key = key_manager.get_key("openai")

# Listaa kaikki avaimet (turvallisesti)
key_manager.list_keys()
```

### Testaus

Testaa API-integraatio suorittamalla:

```bash
python test_delegation.py
```

Testit varmistavat, että:
- API-avaimet ovat oikeassa muodossa
- Integraatiot toimivat (OpenAI, Hugging Face)
- Virheenkäsittely toimii oikein

## macOS-asennus

Sovelluksen asentaminen Mac mini -palvelimelle:

1. Kloonaa repositorio:
```bash
git clone [repository-url]
cd [project-directory]
```

2. Suorita asennusskripti:
```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

3. Päivitä API-avaimet .env-tiedostoon:
```bash
nano .env
```

4. Käynnistä sovellus:
```bash
source venv/bin/activate
python main.py
```

### Järjestelmävaatimukset (Mac mini)
- macOS Catalina tai uudempi
- Python 3.8 tai uudempi
- 16 Gt RAM-muistia (suositus)
- 500 Gt levytilaa
- Intel i7 2,3 GHz tai tehokkaampi

### Suorituskykysuositukset
- Käytä virtuaaliympäristöä (venv)
- Aseta PYTHONPATH oikein
- Varmista että käyttäjällä on tarvittavat oikeudet hakemistoihin
- Käytä asynkronista prosessointia raskaissa operaatioissa

### Vianmääritys
Jos kohtaat ongelmia:
1. Tarkista lokitiedostot: `logs/app.log` ja `logs/metrics.json`
2. Varmista että kaikki riippuvuudet on asennettu: `pip list`
3. Tarkista ympäristömuuttujat: `env | grep API_KEY`
4. Tarkista tiedosto-oikeudet: `ls -la`
