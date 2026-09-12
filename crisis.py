crisis_words=['suicide', 'die']

def crisis_flag(message:str) -> bool:
    for word in message:
        if word in crisis_words:
            return True
    return False
