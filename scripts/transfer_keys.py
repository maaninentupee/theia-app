import os
import json
from cryptography.fernet import Fernet
from base64 import b64encode

def create_secure_transfer():
    # Lue .env tiedosto
    with open('../.env', 'r') as f:
        env_content = f.read()
    
    # Luo salausavain
    key = Fernet.generate_key()
    f = Fernet(key)
    
    # Salaa .env sisältö
    encrypted_content = f.encrypt(env_content.encode())
    
    # Luo paketti
    package = {
        "key": b64encode(key).decode('utf-8'),
        "content": b64encode(encrypted_content).decode('utf-8')
    }
    
    # Tallenna paketti
    with open('keys_transfer.json', 'w') as f:
        json.dump(package, f)
    
    print("Salausavain:", b64encode(key).decode('utf-8'))
    print("\nTallenna tämä avain turvalliseen paikkaan!")
    print("Tarvitset sitä purkaaksesi avaimet Macilla.")

if __name__ == "__main__":
    create_secure_transfer()
