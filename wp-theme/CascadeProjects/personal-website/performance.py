import multiprocessing
from user_messages import motivate_user

def optimize_performance():
    print(f"Asetetaan {multiprocessing.cpu_count()} säiettä maksimaaliseen käyttöön.")
    motivate_user()

def manage_memory_allocation():
    max_memory = 8192  # MB
    print(f"Dynaaminen muisti allokoitu: {max_memory}MB")
    motivate_user()
