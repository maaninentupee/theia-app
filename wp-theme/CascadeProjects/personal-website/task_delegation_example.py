"""
Esimerkki tehtävien delegoinnista eri AI-malleille.
"""

import json
import os
from dotenv import load_dotenv

# Lataa ympäristömuuttujat
load_dotenv()

# Lataa konfiguraatio
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

def delegate_task(task_type):
    """
    Delegoi tehtävän sopivalle mallille tehtävätyypin perusteella.
    
    Args:
        task_type (str): Tehtävän tyyppi
    
    Returns:
        str: Mallin nimi, jolle tehtävä delegoidaan
    """
    task_assignment = config['integration']['task_assignment']
    
    # Tarkista jokainen malli
    for model, tasks in task_assignment.items():
        if task_type in tasks:
            return model.lower()
    
    # Jos tehtävätyyppiä ei löydy, käytä varamallia
    return config['integration']['task_allocation']['fallback_model']

def execute_task(task_type, details):
    """
    Suorittaa tehtävän käyttäen sopivaa mallia.
    
    Args:
        task_type (str): Tehtävän tyyppi
        details (dict): Tehtävän yksityiskohdat
    
    Returns:
        dict: Tehtävän tulos
    """
    # Delegoi tehtävä sopivalle mallille
    model = delegate_task(task_type)
    
    # Testitila: palauta vain mallin nimi ja syöte
    return {
        "status": "success",
        "model": model,
        "result": f"[TEST MODE] {model} suoritti tehtävän: {details['prompt']}"
    }

if __name__ == "__main__":
    # Esimerkki tehtävän suorituksesta
    task = {
        "type": "code_generation",
        "details": {
            "prompt": "Generoi Python-funktio, joka laskee Fibonaccin lukujonon"
        }
    }
    
    try:
        result = execute_task(task["type"], task["details"])
        print(f"Tulos: {result['result']}")
    except Exception as e:
        print(f"Virhe tehtävän suorituksessa: {str(e)}")
