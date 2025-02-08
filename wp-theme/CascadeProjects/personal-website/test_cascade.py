from unittest.mock import patch
import random
from agent_integration import AgentIntegration

def test_all_features():
    print("\n=== Cascade Debug-tila Testaus ===\n")
    
    # 1. Testaa käyttäjän kunnioitus
    print("1. Testataan käyttäjän kunnioitus:")
    with patch('builtins.input', return_value='K'):
        cascade_respect.enforce_respect()
    
    # 2. Testaa virhetilanteiden käsittely
    print("\n2. Testataan virhetilanteiden käsittely:")
    error_handling.handle_error()
    
    # 3. Testaa raportointi
    print("\n3. Testataan raportointi:")
    reporting.report_action("testaus")
    with patch('builtins.input', return_value='K'):
        reporting.confirm_critical_action("kriittinen testaus")
    
    # 4. Testaa suorituskyky
    print("\n4. Testataan suorituskyky:")
    performance.optimize_performance()
    performance.manage_memory_allocation()
    
    # 5. Testaa käyttäjäviestit
    print("\n5. Testataan käyttäjäviestit:")
    random.seed(42)  # Asetetaan siemen toistettavuuden vuoksi
    user_messages.startup_message()
    user_messages.thank_user()
    user_messages.motivate_user()
    user_messages.acknowledge_command()
    
    # 6. Testaa agentti-integraatio
    print("\n6. Testataan agentti-integraatio:")
    integration = AgentIntegration()
    integration.delegate_task("deep_analysis", "Analysoi tämä monimutkainen ongelma")
    integration.delegate_task("quick_responses", "Vastaa tähän nopeasti")
    integration.delegate_task("code_optimization", "Optimoi tämä koodi")
    integration.delegate_task("unknown_task", "Tuntematon tehtävä")
    
    print("\n=== Testaus valmis ===")

if __name__ == "__main__":
    import cascade_respect
    import error_handling
    import reporting
    import performance
    import user_messages
    
    test_all_features()
