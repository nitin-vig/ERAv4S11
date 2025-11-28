# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================
# This is the main entry point for the Hindi BPE Encoder application.
# It orchestrates all components: tokenizer, handlers, and UI.

# Import the tokenizer class
from hindi_bpe_encoder import HindiBPEEncoder

# Import Gradio handlers (wrapper functions)
from gradio_handlers import (
    create_encode_handler,
    create_decode_handler
)

# Import UI builder
from gradio_ui import create_app_interface

# ============================================================================
# INITIALIZATION
# ============================================================================
# Create a single global instance of the encoder
# This instance is shared across all Gradio interface functions
# The tokenizer state persists throughout the app's lifetime
encoder = HindiBPEEncoder()

# Create handler functions bound to the encoder instance
# These functions connect the tokenizer to the Gradio UI
encode_handler = create_encode_handler(encoder)
decode_handler = create_decode_handler(encoder)

# ============================================================================
# APP CREATION
# ============================================================================
# Create the complete Gradio application interface
# This combines all UI components with their respective handlers
demo = create_app_interface(
    encoder=encoder,
    encode_handler=encode_handler,
    decode_handler=decode_handler
)

# ============================================================================
# APP LAUNCH
# ============================================================================
# This block runs when the script is executed directly (not imported)
if __name__ == "__main__":
    # Launch the Gradio app
    # server_name="0.0.0.0": Listen on all network interfaces
    #                      Allows access from other devices on network
    #                      Required for Hugging Face Spaces deployment
    # server_port=7860: Default port for Gradio apps
    #                   Also the default port for Hugging Face Spaces
    demo.launch(server_name="0.0.0.0", server_port=7860)
