"""
Testit eri AI-mallien toiminnalle.
Testaa GPT-4, GPT-3.5 ja Starcoder mallien toimintaa eri tehtävätyypeillä.
"""

import unittest
from task_delegation_example import delegate_task, execute_task

class TestAPIIntegration(unittest.TestCase):
    def setUp(self):
        """Alusta testit"""
        print("\n=== Testataan API-avainten latausta ===")

    def test_api_keys(self):
        """Testaa API-avainten toimivuus"""
        print("\n=== Testataan API-avaimia ===")

    def test_openai_integration(self):
        """Testaa OpenAI integraatio"""
        print("\n=== Testataan OpenAI integraatiota ===")

    def test_huggingface_integration(self):
        """Testaa Hugging Face integraatio"""
        print("\n=== Testataan Hugging Face integraatiota ===")

    def test_cascade_fallback(self):
        """Testaa Cascade varajärjestelmä"""
        print("\n=== Testataan Cascade varajärjestelmää ===")

    def test_error_handling(self):
        """Testaa virheenkäsittely"""
        print("\n=== Testataan virheenkäsittelyä ===")

class TestModelDelegation(unittest.TestCase):
    """Testaa mallien delegointia eri tehtävätyypeille"""
    
    def test_gpt4_delegation(self):
        """Testaa GPT-4 delegointia syväluotaaville tehtäville"""
        tasks = [
            "complex_analysis",
            "detailed_text_generation",
            "deep_analysis",
            "advanced_reasoning"
        ]
        for task in tasks:
            self.assertEqual(delegate_task(task), "gpt_4")
    
    def test_gpt35_delegation(self):
        """Testaa GPT-3.5 delegointia nopeille tehtäville"""
        tasks = [
            "quick_responses",
            "simple_tasks",
            "basic_analysis",
            "chat_completion"
        ]
        for task in tasks:
            self.assertEqual(delegate_task(task), "gpt_3_5")
    
    def test_starcoder_delegation(self):
        """Testaa Starcoder delegointia koodaustehtäville"""
        tasks = [
            "code_generation",
            "code_optimization",
            "code_review",
            "bug_fixing"
        ]
        for task in tasks:
            self.assertEqual(delegate_task(task), "starcoder")

class TestModelExecution(unittest.TestCase):
    """Testaa mallien suoritusta eri tehtävätyypeillä"""
    
    def test_gpt4_analysis(self):
        """Testaa GPT-4 syväluotaavaa analyysiä"""
        task = {
            "type": "complex_analysis",
            "details": {
                "prompt": "Analysoi tekoälyn vaikutuksia yhteiskuntaan seuraavan 10 vuoden aikana."
            }
        }
        try:
            result = execute_task(task["type"], task["details"])
            self.assertIsNotNone(result["result"])
            print("\nGPT-4 Analyysi:")
            print(result["result"][:200] + "...")  # Näytä vain alku
        except Exception as e:
            print(f"GPT-4 testi epäonnistui: {str(e)}")
    
    def test_gpt35_quick_response(self):
        """Testaa GPT-3.5 nopeaa vastausta"""
        task = {
            "type": "quick_responses",
            "details": {
                "prompt": "Mikä on Python-kielen versio 3.12:n tärkein uusi ominaisuus?"
            }
        }
        try:
            result = execute_task(task["type"], task["details"])
            self.assertIsNotNone(result["result"])
            print("\nGPT-3.5 Vastaus:")
            print(result["result"])
        except Exception as e:
            print(f"GPT-3.5 testi epäonnistui: {str(e)}")
    
    def test_starcoder_optimization(self):
        """Testaa Starcoder koodin optimointia"""
        code_to_optimize = """
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
"""
        task = {
            "type": "code_optimization",
            "details": {
                "prompt": f"Optimoi seuraava Fibonacci-lukujonon generointifunktio:\n{code_to_optimize}"
            }
        }
        try:
            result = execute_task(task["type"], task["details"])
            self.assertIsNotNone(result["result"])
            print("\nStarcoder Optimoitu Koodi:")
            print(result["result"])
        except Exception as e:
            print(f"Starcoder testi epäonnistui: {str(e)}")

def run_example_tasks():
    """Suorittaa esimerkki tehtävät kaikille malleille"""
    print("\nSuoritetaan esimerkki tehtävät:")
    
    # GPT-4: Syväluotaava analyysi
    print("\n1. GPT-4: Syväluotaava analyysi")
    task = {
        "type": "complex_analysis",
        "details": {
            "prompt": "Analysoi tekoälyn eettisiä vaikutuksia ohjelmistokehityksessä."
        }
    }
    try:
        result = execute_task(task["type"], task["details"])
        print(result["result"])
    except Exception as e:
        print(f"Virhe: {str(e)}")
    
    # GPT-3.5: Nopea vastaus
    print("\n2. GPT-3.5: Nopea vastaus")
    task = {
        "type": "quick_responses",
        "details": {
            "prompt": "Selitä lyhyesti mitä tarkoittaa API?"
        }
    }
    try:
        result = execute_task(task["type"], task["details"])
        print(result["result"])
    except Exception as e:
        print(f"Virhe: {str(e)}")
    
    # Starcoder: Koodin generointi
    print("\n3. Starcoder: Koodin generointi")
    task = {
        "type": "code_generation",
        "details": {
            "prompt": "Generoi Python-funktio, joka laskee annetun listan lukujen keskiarvon."
        }
    }
    try:
        result = execute_task(task["type"], task["details"])
        print(result["result"])
    except Exception as e:
        print(f"Virhe: {str(e)}")

if __name__ == "__main__":
    # Suorita yksikkötestit
    print("Suoritetaan yksikkötestit...")
    unittest.main(argv=[''], exit=False)
    
    # Suorita esimerkki tehtävät
    print("\nSuoritetaan esimerkki tehtävät...")
    run_example_tasks()
