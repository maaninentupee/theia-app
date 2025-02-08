# API-avainten Asetusohje

## 1. Luo .env tiedosto

Luo projektin juurihakemistoon tiedosto nimeltä `.env` ja lisää seuraavat rivit:

```env
GPT4_API_KEY=your_gpt4_key_here
GPT3_API_KEY=your_gpt3_key_here
CLAUDE_API_KEY=your_claude_key_here
STARCODER_API_KEY=your_starcoder_key_here
AUTOGPT_API_KEY=your_autogpt_key_here
```

## 2. Hanki API-avaimet

### OpenAI (GPT-4 ja GPT-3.5)
1. Mene osoitteeseen https://platform.openai.com/
2. Kirjaudu sisään tai luo tili
3. Mene API-avaimet osioon
4. Luo uusi avain ja kopioi se
5. Aseta avain sekä GPT4_API_KEY että GPT3_API_KEY kohtiin

### Anthropic (Claude)
1. Mene osoitteeseen https://www.anthropic.com/
2. Hanki Claude API-avain
3. Aseta avain CLAUDE_API_KEY kohtaan

### Hugging Face (Starcoder)
1. Mene osoitteeseen https://huggingface.co/
2. Kirjaudu sisään tai luo tili
3. Mene asetuksiin ja luo API-avain
4. Aseta avain STARCODER_API_KEY kohtaan

### AutoGPT
1. Seuraa AutoGPT:n virallisia ohjeita API-avaimen hankkimiseen
2. Aseta avain AUTOGPT_API_KEY kohtaan

## 3. Asenna riippuvuudet

Asenna tarvittavat Python-paketit:

```bash
pip install -r requirements.txt
```

## 4. Testaa asennus

Varmista että asennus toimii ajamalla testit:

```bash
python test_cascade.py
```

## Huomioitavaa

- Älä koskaan jaa API-avaimiasi muille
- Älä koskaan lisää .env tiedostoa versionhallintaan
- Tarkista säännöllisesti API-avainten voimassaolo
- Seuraa API-käyttöä kustannusten hallitsemiseksi

## Vianetsintä

Jos kohtaat ongelmia:

1. Tarkista että .env tiedosto on oikeassa paikassa
2. Varmista että API-avaimet on kopioitu oikein
3. Tarkista että riippuvuudet on asennettu
4. Katso virheilmoitukset test_cascade.py ajosta
