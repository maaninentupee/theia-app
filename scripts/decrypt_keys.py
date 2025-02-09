import json
from cryptography.fernet import Fernet
from base64 import b64decode

def decrypt_keys():
    # Pyydä salausavain käyttäjältä
    key = input("Syötä salausavain: ")
    
    try:
        # Lue salattu paketti
        with open('keys_transfer.json', 'r') as f:
            package = json.load(f)
        
        # Pura salaus
        f = Fernet(b64decode(key))
        decrypted = f.decrypt(b64decode(package['content']))
        
        # Tallenna .env tiedostoon
        with open('../.env', 'w') as f:
            f.write(decrypted.decode('utf-8'))
        
        print("API-avaimet purettu onnistuneesti!")
        print("Avaimet tallennettu .env tiedostoon.")
        
    except Exception as e:
        print("Virhe avainten purkamisessa:", str(e))

if __name__ == "__main__":
    decrypt_keys()
