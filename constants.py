# ============================================================================
# CONSTANTS AND SAMPLE DATA
# ============================================================================
# This module contains constants and sample data used throughout the application

# Pre-filled sample Hindi text for training demonstration
# This gives users a starting point and shows expected input format
# Contains diverse Hindi vocabulary and sentence structures
SAMPLE_HINDI_TEXT = """हिंदी भारत की राष्ट्रभाषा है।
यह देवनागरी लिपि में लिखी जाती है।
हिंदी बोलने वालों की संख्या बहुत अधिक है।
यह भाषा संस्कृत से विकसित हुई है।
हिंदी में कई सुंदर कविताएं और कहानियां हैं।"""

# Example Hindi sentences for the Encode tab
ENCODE_EXAMPLES = [
    ["हिंदी भारत की राष्ट्रभाषा है।"],  # "Hindi is India's national language."
    ["यह देवनागरी लिपि में लिखी जाती है।"],  # "It is written in Devanagari script."
    ["मैं हिंदी सीख रहा हूँ।"]  # "I am learning Hindi."
]

# Default vocabulary size for training
DEFAULT_VOCAB_SIZE = 10000

# Vocabulary size limits
MIN_VOCAB_SIZE = 100
MAX_VOCAB_SIZE = 50000

# App configuration
APP_TITLE = "Hindi BPE Encoder"

