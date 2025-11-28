# ============================================================================
# GRADIO INTERFACE HANDLERS
# ============================================================================
# This module contains wrapper functions that connect the tokenizer to Gradio UI
# These functions format inputs/outputs for display in the web interface

from constants import DEFAULT_VOCAB_SIZE


def create_encode_handler(encoder):
    """
    Create an encode handler function bound to a specific encoder instance.
    
    Args:
        encoder: HindiBPEEncoder instance to use for encoding
    
    Returns:
        function: Handler function for Gradio interface
    """
    def encode_text(text):
        """
        Gradio wrapper function for encoding Hindi text.
        
        This function:
        1. Calls the encoder's encode method
        2. Formats the output for display in multiple textboxes
        3. Handles errors gracefully
        
        Args:
            text (str): Hindi text input from Gradio textbox
        
        Returns:
            tuple: (token_ids_str, tokens_str, attention_mask_str)
                   Each is a formatted string for display
        """
        # Call the encoder's encode method
        result = encoder.encode(text)
        
        # Check if encoding resulted in an error
        if isinstance(result, dict) and "error" in result:
            # Return error message in first output, empty strings for others
            return result["error"], "", "", ""
        
        # Format outputs for display:
        # Token IDs: Convert list of integers to comma-separated string
        token_ids_str = ", ".join(map(str, result["token_ids"]))
        
        # Tokens: Join token strings with " | " separator for readability
        tokens_str = " | ".join(result["tokens"])
        
        # Attention mask: Convert list of integers to comma-separated string
        attention_mask_str = ", ".join(map(str, result["attention_mask"]))
        
        # Compression ratio: Format with 2 decimal places
        compression_info = (
            f"Compression Ratio: {result.get('compression_ratio', 0):.2f}:1\n"
            f"Original Characters: {result.get('original_char_count', 0)}\n"
            f"Tokens: {result.get('token_count', 0)}\n"
            f"Space Saved: {result.get('compression_percentage', 0):.1f}%"
        )
        
        # Return tuple for Gradio's multiple outputs
        return token_ids_str, tokens_str, attention_mask_str, compression_info
    
    return encode_text


def create_decode_handler(encoder):
    """
    Create a decode handler function bound to a specific encoder instance.
    
    Args:
        encoder: HindiBPEEncoder instance to use for decoding
    
    Returns:
        function: Handler function for Gradio interface
    """
    def decode_tokens(token_ids_str):
        """
        Gradio wrapper function for decoding token IDs to Hindi text.
        
        Args:
            token_ids_str (str): Comma-separated token IDs from Gradio textbox
        
        Returns:
            str: Decoded Hindi text, or error message if decoding fails
        """
        # Call the encoder's decode method
        # The decode method handles parsing the comma-separated string
        result = encoder.decode(token_ids_str)
        return result
    
    return decode_tokens


def create_train_handler(encoder):
    """
    Create a train handler function bound to a specific encoder instance.
    
    Args:
        encoder: HindiBPEEncoder instance to use for training
    
    Returns:
        function: Handler function for Gradio interface
    """
    def train_tokenizer_interface(training_text, vocab_size):
        """
        Gradio wrapper function for training the tokenizer.
        
        This function:
        1. Validates and converts vocab_size to integer
        2. Calls the encoder's train method
        3. Returns status message for display
        
        Args:
            training_text (str): Hindi text corpus from Gradio textbox
            vocab_size: Vocabulary size from Gradio number input (can be float/int/None)
        
        Returns:
            str: Training status message (success or error)
        """
        # Convert vocab_size to integer, handling various input types
        try:
            # Gradio Number input can return float or None, so convert appropriately
            vocab_size = int(vocab_size) if vocab_size else DEFAULT_VOCAB_SIZE
        except:
            # If conversion fails, use default value
            vocab_size = DEFAULT_VOCAB_SIZE
        
        # Call the encoder's train method
        result = encoder.train_tokenizer(training_text, vocab_size)
        return result
    
    return train_tokenizer_interface

