from user_messages import motivate_user

def enforce_respect():
    response = input("Oletko tyytyväinen Cascaden toimintaan? (K/E): ")
    if response.lower() == "e":
        print("Cascade pyrkii parantamaan suoritustaan välittömästi.")
    else:
        print("Cascade kiittää hyväksynnästäsi ja jatkaa työtään.")
    motivate_user()
