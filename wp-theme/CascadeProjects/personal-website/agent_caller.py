import openai
import anthropic
from huggingface_hub import HfApi
from user_messages import motivate_user
from model_config import ModelManager

class AgentCaller:
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        if not test_mode:
            self.model_manager = ModelManager()
    
    def call_agent(self, agent, api_key, task_details):
        """Kutsuu agenttia tehtävän suorittamiseen"""
        if self.test_mode:
            return self._test_mode_call(agent, task_details)
            
        try:
            if agent.startswith("gpt"):
                return self._call_openai(agent, task_details)
            elif agent == "claude":
                return self._call_claude(api_key, task_details)
            elif agent == "starcoder":
                return self._call_starcoder(task_details)
            elif agent == "autogpt":
                return self._call_autogpt(api_key, task_details)
            else:
                raise ValueError(f"Tuntematon agentti: {agent}")
        except Exception as e:
            print(f"Virhe agentin kutsussa: {str(e)}")
            return {"error": str(e)}
    
    def _call_openai(self, model_type, task_details):
        """Kutsuu OpenAI:n GPT malleja"""
        model_config = self.model_manager.get_model_config(model_type)
        if not model_config:
            raise ValueError(f"Mallin {model_type} konfiguraatiota ei löydy")
        
        print(f"Kutsutaan OpenAI mallia {model_config.model_id}...")
        openai.api_key = model_config.api_key
        
        response = openai.ChatCompletion.create(
            model=model_config.model_id,
            messages=[
                {"role": "system", "content": "Olet avulias tekoälyassistentti."},
                {"role": "user", "content": task_details}
            ],
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens
        )
        
        motivate_user()
        return {
            "status": "success",
            "response": response.choices[0].message.content,
            "model": model_type
        }
    
    def _call_claude(self, api_key, task_details):
        """Kutsuu Anthropicin Claude mallia"""
        print("Kutsutaan Claude Opus mallia...")
        client = anthropic.Client(api_key=api_key)
        
        response = client.messages.create(
            model="claude-2.1",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": task_details
            }]
        )
        
        motivate_user()
        return {
            "status": "success",
            "response": response.content,
            "model": "claude"
        }
    
    def _call_starcoder(self, task_details):
        """Kutsuu Hugging Face Starcoder mallia"""
        model_config = self.model_manager.get_model_config("starcoder")
        if not model_config:
            raise ValueError("Starcoder mallin konfiguraatiota ei löydy")
            
        print(f"Kutsutaan Starcoder mallia {model_config.model_id}...")
        hf = HfApi(token=model_config.api_key)
        
        # Tässä käytettäisiin Hugging Face API:a koodin optimointiin
        # Esimerkki yksinkertaistettu
        response = "Koodi optimoitu Starcoderilla"
        
        motivate_user()
        return {
            "status": "success",
            "response": response,
            "model": "starcoder"
        }
    
    def _call_autogpt(self, api_key, task_details):
        """Kutsuu AutoGPT:tä"""
        print("Kutsutaan AutoGPT:tä...")
        
        # Tässä käytettäisiin AutoGPT:n API:a
        # Esimerkki yksinkertaistettu
        response = "AutoGPT generoi työnkulun"
        
        motivate_user()
        return {
            "status": "success",
            "response": response,
            "model": "autogpt"
        }
    
    def _test_mode_call(self, agent, task_details):
        """Simuloi API-kutsua testitilassa"""
        print(f"[TESTITILA] Kutsutaan agenttia {agent}...")
        print(f"Tehtävän tiedot: {task_details}")
        
        response = f"Testitilan vastaus agentilta {agent}"
        motivate_user()
        
        return {
            "status": "success",
            "response": response,
            "model": agent
        }
