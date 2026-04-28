# src/verifier.py

def verify_generation(original, corrupted):
    """
    Checks if the LLM successfully introduced an error.
    Returns True if the sample is valid, False otherwise.
    """
    # 1. Check for empty strings
    if not corrupted or not original:
        return False
        
    # 2. Check if the LLM just returned the exact same sentence
    if corrupted.strip() == original.strip():
        return False
        
    # 3. Basic length check to ensure it's a full sentence (min 3 words)
    if len(corrupted.split()) < 3:
        return False
        
    return True