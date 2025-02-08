from user_messages import motivate_user

def report_action(action):
    print(f"[Raportti] Cascade suoritti toiminnon: {action}")
    motivate_user()

def confirm_critical_action(action):
    response = input(f"Haluatko varmasti suorittaa seuraavan toiminnon: {action}? (K/E): ")
    if response.lower() != "k":
        print("Toiminto peruttu.")
        return False
    motivate_user()
    return True
