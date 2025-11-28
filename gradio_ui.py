# ============================================================================
# GRADIO UI COMPONENTS
# ============================================================================
# This module contains the Gradio UI layout and component definitions
# Separated from handlers for better modularity

import gradio as gr
from constants import (
    ENCODE_EXAMPLES,
    APP_TITLE
)


def create_encode_tab(encode_handler):
    """
    Create the Encode tab UI components.
    
    Args:
        encode_handler: Function to handle encoding requests
    
    Returns:
        tuple: (encode_input, encode_btn) - Input components for event binding
    """
    gr.Markdown("### Encode Hindi Text to Tokens")
    
    # Create a row layout with two columns
    with gr.Row():
        # Left column: Input section
        with gr.Column():
            # Text input box for Hindi text
            encode_input = gr.Textbox(
                label="Hindi Text",
                placeholder="हिंदी में कुछ लिखें...",  # Hindi placeholder text
                lines=5  # Height of textbox (5 lines)
            )
            # Primary button to trigger encoding
            encode_btn = gr.Button("Encode", variant="primary")
        
        # Right column: Output section
        with gr.Column():
            # Output box for token IDs (numeric representation)
            token_ids_output = gr.Textbox(
                label="Token IDs",
                lines=3
            )
            # Output box for actual token strings (subwords)
            tokens_output = gr.Textbox(
                label="Tokens",
                lines=5
            )
            # Output box for attention mask (padding information)
            attention_mask_output = gr.Textbox(
                label="Attention Mask",
                lines=3
            )
            # Output box for compression ratio information
            compression_output = gr.Textbox(
                label="Compression Ratio",
                lines=4
            )
    
    # Connect button click to function
    # When button is clicked, call encode_handler function
    # Pass encode_input as input, update four output boxes
    encode_btn.click(
        fn=encode_handler,  # Function to call
        inputs=encode_input,  # Input component
        outputs=[token_ids_output, tokens_output, attention_mask_output, compression_output]  # Output components
    )
    
    # Add example inputs for quick testing
    # Users can click examples to populate the input box
    gr.Examples(
        examples=ENCODE_EXAMPLES,
        inputs=encode_input  # Which input component to populate
    )
    
    return encode_input, encode_btn


def create_decode_tab(decode_handler):
    """
    Create the Decode tab UI components.
    
    Args:
        decode_handler: Function to handle decoding requests
    
    Returns:
        tuple: (decode_input, decode_btn) - Input components for event binding
    """
    gr.Markdown("### Decode Token IDs to Hindi Text")
    
    # Two-column layout: input on left, output on right
    with gr.Row():
        with gr.Column():
            # Input box for token IDs (comma-separated format)
            decode_input = gr.Textbox(
                label="Token IDs (comma-separated)",
                placeholder="1, 2, 3, 4, 5",  # Example format
                lines=3
            )
            # Button to trigger decoding
            decode_btn = gr.Button("Decode", variant="primary")
        
        with gr.Column():
            # Output box for decoded Hindi text
            decode_output = gr.Textbox(
                label="Decoded Text",
                lines=5
            )
    
    # Connect button to decode function
    decode_btn.click(
        fn=decode_handler,  # Function to call
        inputs=decode_input,  # Input: token IDs string
        outputs=decode_output  # Output: decoded text
    )
    
    return decode_input, decode_btn


def create_app_interface(encoder, encode_handler, decode_handler):
    """
    Create the complete Gradio application interface.
    
    Args:
        encoder: HindiBPEEncoder instance
        encode_handler: Function to handle encoding
        decode_handler: Function to handle decoding
    
    Returns:
        gr.Blocks: Gradio Blocks interface
    """
    # Create the main Gradio web application interface
    # gr.Blocks provides more flexibility than gr.Interface for complex layouts
    # with gr.Blocks(...) as demo: creates a context manager for the app
    # Note: theme parameter removed for compatibility with all Gradio versions
    with gr.Blocks(title=APP_TITLE) as demo:
        # Title and description section at the top of the app
        # Markdown allows rich text formatting
        gr.Markdown(f"""
        # 🇮🇳 Hindi BPE (Byte Pair Encoding) Tokenizer
        
        This app provides a BPE tokenizer for Hindi text. You can:
        - Encode Hindi text into token IDs
        - Decode token IDs back to Hindi text
        
        **Current Vocabulary Size:** {encoder.get_vocab_size()}
        
        *Note: To train the tokenizer, use the `train_tokenizer.py` script with your corpus.*
        """)
        
        # Create tabbed interface for organizing different functionalities
        with gr.Tabs():
            # ====================================================================
            # TAB 1: ENCODE
            # ====================================================================
            # This tab allows users to encode Hindi text into tokens
            with gr.TabItem("Encode"):
                create_encode_tab(encode_handler)
            
            # ====================================================================
            # TAB 2: DECODE
            # ====================================================================
            # This tab allows users to decode token IDs back to Hindi text
            with gr.TabItem("Decode"):
                create_decode_tab(decode_handler)
    
    return demo

