import json
from user_messages import motivate_user
from api_config import APIConfig

class AgentIntegration:
    def __init__(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
            self.integration_config = config['integration']
        
        self.api_config = APIConfig()
        self.agents = {
            'gpt_4_turbo': self._gpt4_turbo_handler,
            'gpt_3_5_turbo': self._gpt35_turbo_handler,
            'claude_opus': self._claude_opus_handler,
            'starcoder': self._starcoder_handler,
            'autogpt': self._autogpt_handler
        }
    
    def delegate_task(self, task_type, task_data):
        """Delegoi tehtävän sopivalle agentille"""
        for agent, tasks in self.integration_config['task_assignment'].items():
            if task_type in tasks:
                print(f"Delegoidaan tehtävä agentille {agent}")
                return self.agents[agent](task_data)
        
        print("Tehtävätyyppiä ei tunnistettu. Cascade hoitaa tehtävän itse.")
        return self._cascade_handler(task_data)
    
    def _gpt4_turbo_handler(self, task_data):
        api_key = self.api_config.get_api_key('gpt4')
        print("GPT-4-turbo käsittelee monimutkaista tehtävää...")
        # Tässä käytettäisiin API-avainta GPT-4 kutsua varten
        motivate_user()
        return "GPT-4-turbo suoritti tehtävän"
    
    def _gpt35_turbo_handler(self, task_data):
        api_key = self.api_config.get_api_key('gpt3')
        print("GPT-3.5-turbo käsittelee kevyttä tehtävää...")
        # Tässä käytettäisiin API-avainta GPT-3.5 kutsua varten
        motivate_user()
        return "GPT-3.5-turbo suoritti tehtävän"
    
    def _claude_opus_handler(self, task_data):
        api_key = self.api_config.get_api_key('claude')
        print("Claude Opus analysoi kontekstia...")
        # Tässä käytettäisiin API-avainta Claude kutsua varten
        motivate_user()
        return "Claude Opus suoritti tehtävän"
    
    def _starcoder_handler(self, task_data):
        api_key = self.api_config.get_api_key('starcoder')
        print("Starcoder optimoi koodia...")
        # Tässä käytettäisiin API-avainta Starcoder kutsua varten
        motivate_user()
        return "Starcoder suoritti tehtävän"
    
    def _autogpt_handler(self, task_data):
        api_key = self.api_config.get_api_key('autogpt')
        print("AutoGPT generoi ideoita...")
        # Tässä käytettäisiin API-avainta AutoGPT kutsua varten
        motivate_user()
        return "AutoGPT suoritti tehtävän"
    
    def _cascade_handler(self, task_data):
        print("Cascade käsittelee tehtävän itse...")
        motivate_user()
        return "Cascade suoritti tehtävän"
