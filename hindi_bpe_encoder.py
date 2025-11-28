# ============================================================================
# HINDI BPE ENCODER MODULE
# ============================================================================
# This module contains the HindiBPEEncoder class for BPE tokenization of Hindi text.
# It can be imported and used independently of the Gradio web interface.
# Uses byte-level BPE with 256 base tokens (one for each byte 0-255).
# This ensures BPE always learns merges regardless of vocabulary size.

# Hugging Face tokenizers library: Fast, production-ready tokenization
# Tokenizer: Main tokenizer class that handles encoding/decoding
from tokenizers import Tokenizer
# BPE: Byte Pair Encoding model - subword tokenization algorithm
#      Works by iteratively merging most frequent byte pairs
#      With byte_fallback=True, starts with 256 base tokens (one per byte)
#      Text is encoded as UTF-8 bytes, then BPE learns merges on bytes
from tokenizers.models import BPE
# BpeTrainer: Trainer class that learns BPE merges from training data
from tokenizers.trainers import BpeTrainer
# Whitespace: Pre-tokenizer that splits text on whitespace
#             Good for Hindi as words are space-separated in Devanagari script
from tokenizers.pre_tokenizers import Whitespace
# BertProcessing: Post-processor that adds special tokens like <s> and </s>
#                 Similar to BERT's tokenization format
from tokenizers.processors import BertProcessing
# Normalizers: For Unicode normalization to ensure consistent character representation
# NFD: Normalization Form Decomposed - decomposes characters into base + combining marks
#      This ensures consistent Unicode representation during training and encoding
from tokenizers.normalizers import NFD, Sequence

# Standard library imports
import os  # For file system operations (checking if tokenizer file exists)

# Import Hindi text preprocessing functions with regex
from hindi_preprocessor import clean_hindi_text


# ============================================================================
# HINDI BPE ENCODER CLASS
# ============================================================================
# This class encapsulates all BPE tokenization functionality for Hindi text.
# It handles initialization, training, encoding, and decoding operations.
class HindiBPEEncoder:
    def __init__(self, tokenizer_path="hindi_bpe_tokenizer.json"):
        """
        Initialize the HindiBPEEncoder.
        Sets up the tokenizer by either loading an existing trained model
        or creating a new untrained tokenizer.
        
        Args:
            tokenizer_path (str): Path to save/load the tokenizer file (default: "hindi_bpe_tokenizer.json")
        """
        # Tokenizer instance - will hold the actual BPE tokenizer object
        self.tokenizer = None
        
        # Path where the trained tokenizer will be saved/loaded from
        # JSON format allows easy serialization and loading
        self.tokenizer_path = tokenizer_path
        
        # Attempt to load existing tokenizer, or create new one if not found
        self.load_or_initialize_tokenizer()
    
    def load_or_initialize_tokenizer(self):
        """
        Load existing tokenizer from file or initialize a new one.
        
        This method implements a persistence mechanism:
        1. Checks if a saved tokenizer file exists
        2. If exists, tries to load it (may fail if file is corrupted)
        3. If loading fails or file doesn't exist, creates a new tokenizer
        4. This allows the app to remember trained tokenizers across sessions
        """
        # Check if a previously saved tokenizer exists
        if os.path.exists(self.tokenizer_path):
            try:
                # Load the tokenizer from JSON file
                # This restores the vocabulary, merges, and all tokenizer settings
                self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
                print(f"Loaded tokenizer from {self.tokenizer_path}")
            except Exception as e:
                # If loading fails (corrupted file, version mismatch, etc.),
                # fall back to creating a new tokenizer
                print(f"Error loading tokenizer: {e}. Initializing new tokenizer.")
                self.initialize_tokenizer()
        else:
            # No saved tokenizer found, create a fresh one
            self.initialize_tokenizer()
    
    def initialize_tokenizer(self):
        """
        Initialize a new, untrained BPE tokenizer for Hindi with byte-level fallback.
        
        This creates a tokenizer with basic configuration but no vocabulary yet.
        The tokenizer must be trained before it can encode/decode text effectively.
        
        Configuration choices:
        - BPE model: Uses Byte Pair Encoding algorithm for subword tokenization
        - unk_token: "<unk>" token for unknown/out-of-vocabulary words
        - byte_fallback: True enables byte-level BPE with 256 base tokens
          Starts with exactly 256 tokens (one for each byte 0-255)
          Text is encoded as UTF-8 bytes, then BPE learns merges on bytes
          This ensures merges are always learned regardless of vocab_size
        - Unicode normalization (NFD): Normalizes text to Normalization Form Decomposed
          Ensures consistent Unicode representation (composed vs decomposed forms)
          Prevents issues where same character in different forms is treated differently
          Example: "छोड़कर" will always be normalized consistently
        - Whitespace pre-tokenizer: Splits on whitespace, which works well for
          Hindi since words in Devanagari script are space-separated
        """
        # Create a new Tokenizer with BPE model using 256 base tokens (byte-level)
        # BPE(unk_token="<unk>", byte_fallback=True) enables byte-level BPE
        # This starts with exactly 256 base tokens (one for each byte 0-255)
        # Ensures BPE always learns merges regardless of vocab_size
        # Text is first encoded as UTF-8 bytes, then BPE learns merges on bytes
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
        
        # Add Unicode normalization (NFD - Normalization Form Decomposed)
        # This ensures consistent Unicode representation during training and encoding
        # Prevents issues where the same character in different Unicode forms
        # (composed vs decomposed) is treated as different tokens
        # Example: "छोड़कर" will always be normalized consistently
        self.tokenizer.normalizer = Sequence([NFD()])
        
        # Set pre-tokenizer: splits text into words before BPE processing
        # Whitespace() splits on whitespace characters (spaces, tabs, newlines)
        # This is appropriate for Hindi as it's written with spaces between words
        # Alternative could be character-level, but word-level is more efficient
        self.tokenizer.pre_tokenizer = Whitespace()
        
        print("Initialized new BPE tokenizer with 256 base tokens (byte-level BPE + Unicode normalization)")
    
    def train_tokenizer(self, training_texts, vocab_size=5000, use_streaming=False, chunk_size=1024*1024):
        """
        Train the BPE tokenizer on provided Hindi text corpus.
        
        Optimized for large datasets with streaming support and efficient preprocessing.
        
        BPE Training Process:
        1. Preprocess: Clean and normalize training text using regex (optimized)
        2. Starts with character-level vocabulary
        3. Iteratively finds most frequent character pairs
        4. Merges them into new subword units
        5. Repeats until vocabulary reaches specified size
        
        Args:
            training_texts: Hindi text corpus - can be:
                           - str: Text content (for small-medium datasets)
                           - str (file path): Path to file (if use_streaming=True)
            vocab_size (int): Desired vocabulary size (default: 5000)
                             With 256 base tokens, any vocab_size > 256 will learn merges
                             Recommended: 3000-10000 for Hindi
            use_streaming (bool): If True, treat training_texts as file path and stream
            chunk_size (int): Chunk size for streaming (default: 1MB)
        
        Returns:
            str: Success message or error description
        """
        # Validate input
        if not training_texts:
            return "Error: Please provide training text"
        
        try:
            from hindi_preprocessor import clean_hindi_text, clean_hindi_text_streaming
            
            # Initialize BPE trainer with configuration
            trainer = BpeTrainer(
                vocab_size=vocab_size,  # Target vocabulary size
                # Special tokens that will always be in vocabulary:
                # <unk>: Unknown token for OOV words
                # <s>: Start of sequence token
                # </s>: End of sequence token
                # <pad>: Padding token for batch processing
                # <mask>: Masking token (useful for MLM tasks)
                special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"],
                min_frequency=2  # Minimum frequency for a token to be included
                                 # Helps filter out rare/noisy tokens
            )
            
            # Prepare training data based on input type
            if use_streaming:
                # For very large files: stream and process in chunks
                # This avoids loading entire file into memory
                def text_iterator():
                    for cleaned_chunk in clean_hindi_text_streaming(training_texts, chunk_size):
                        # Split into lines for training (already cleaned by clean_hindi_text_streaming)
                        for line in cleaned_chunk.split('\n'):
                            line = line.strip()
                            if line:
                                yield line
                
                training_iterator = text_iterator()
            else:
                # For in-memory text: preprocess once, then iterate
                if isinstance(training_texts, str) and len(training_texts) > 100 * 1024 * 1024:  # >100MB
                    # Very large text: process in chunks
                    from io import StringIO
                    def text_iterator():
                        buffer = StringIO(training_texts)
                        chunk_size = 10 * 1024 * 1024  # 10MB chunks
                        while True:
                            chunk = buffer.read(chunk_size)
                            if not chunk:
                                break
                            cleaned = clean_hindi_text(chunk)
                            # Split into lines for training (already cleaned)
                            for line in cleaned.split('\n'):
                                line = line.strip()
                                if line:
                                    yield line
                    training_iterator = text_iterator()
                else:
                    # Small-medium text: preprocess once
                    cleaned_text = clean_hindi_text(training_texts)
                    # Create efficient iterator (already cleaned)
                    training_iterator = (line.strip() for line in cleaned_text.split('\n') if line.strip())
            
            # Train the tokenizer using the optimized iterator
            # train_from_iterator processes text line by line efficiently
            print(f"\n   Training BPE tokenizer (vocab_size={vocab_size})...")
            self.tokenizer.train_from_iterator(
                training_iterator,
                trainer=trainer
            )
            
            # Validate that merges were learned
            # With 256 base tokens, BPE should always learn merges if vocab_size > 256
            vocab = self.tokenizer.get_vocab()
            
            # Count base tokens (single-byte tokens) vs merged tokens
            # Base tokens are those that are single bytes (0-255)
            base_tokens = 0
            merged_tokens = 0
            for token in vocab.keys():
                # Skip special tokens
                if token in ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]:
                    continue
                # Check if token is a single byte (base token)
                try:
                    token_bytes = token.encode('utf-8')
                    if len(token_bytes) == 1:
                        base_tokens += 1
                    else:
                        merged_tokens += 1
                except:
                    merged_tokens += 1
            
            # Get number of merges from the saved tokenizer data
            # The number of merges = vocab_size - base_tokens - special_tokens
            num_merges = len(vocab) - base_tokens - 5  # 5 special tokens
            
            print(f"   ✓ Training complete!")
            print(f"   Vocabulary size: {len(vocab)}")
            print(f"   Base tokens (256): ~{base_tokens}")
            print(f"   Estimated merges: ~{num_merges}")
            print(f"   Merged tokens: ~{merged_tokens}")
            
            if num_merges <= 0:
                print(f"\n   ⚠️  WARNING: No BPE merges learned!")
                print(f"   This should not happen with byte-level BPE (256 base tokens).")
                print(f"   Check that vocab_size ({vocab_size}) > 256 and training data is sufficient.")
            
            # Set post-processor after training (when special tokens exist in vocab)
            # Post-processor adds special tokens to encoded sequences
            # BertProcessing adds </s> (CLS) and <s> (SEP) tokens like BERT
            try:
                # Get token IDs for special tokens (they exist after training)
                # Use fallback values if tokens don't exist (shouldn't happen)
                cls_token_id = self.tokenizer.token_to_id("</s>") if self.tokenizer.token_to_id("</s>") is not None else 1
                sep_token_id = self.tokenizer.token_to_id("<s>") if self.tokenizer.token_to_id("<s>") is not None else 0
                self.tokenizer.post_processor = BertProcessing(
                    ("</s>", cls_token_id),  # End token (like BERT's [CLS])
                    ("<s>", sep_token_id),   # Start token (like BERT's [SEP])
                )
            except:
                # Post-processor is optional - if it fails, continue without it
                # The tokenizer will still work, just without automatic special tokens
                pass
            
            # Save the trained tokenizer to disk for future use
            # Saves vocabulary, merge rules, and all configuration
            print(f"\n   💾 Saving tokenizer to '{self.tokenizer_path}'...")
            try:
                self.tokenizer.save(self.tokenizer_path)
                # Verify file was created
                import os
                if os.path.exists(self.tokenizer_path):
                    file_size = os.path.getsize(self.tokenizer_path)
                    print(f"   ✓ Tokenizer saved successfully ({file_size:,} bytes)")
                else:
                    return f"Error: Tokenizer file was not created at '{self.tokenizer_path}'"
            except Exception as save_error:
                return f"Error saving tokenizer: {str(save_error)}"
            
            return f"Tokenizer trained successfully with vocab size {vocab_size}!"
        except Exception as e:
            # Return error message if training fails
            # Common causes: invalid input, memory issues, or tokenizer errors
            return f"Error training tokenizer: {str(e)}"
    
    def encode(self, text):
        """
        Encode Hindi text into token IDs and subword tokens.
        
        Encoding Process:
        1. Preprocess: Clean and normalize Hindi text using regex
        2. Pre-tokenize: Split text on whitespace into words
        3. Apply BPE: Break words into subword units using learned merges
        4. Convert to IDs: Map each token to its vocabulary ID
        
        Args:
            text (str): Hindi text to encode
        
        Returns:
            dict: Contains token_ids, tokens, attention_mask, and offsets
                  OR error dict if encoding fails
        """
        # Check if tokenizer is initialized
        if not self.tokenizer:
            return "Error: Tokenizer not initialized"
        
        # Validate input text
        if not text or not text.strip():
            return {"error": "Please provide text to encode"}
        
        try:
            # Preprocess Hindi text using regex
            # This normalizes whitespace, handles punctuation, and cleans the text
            cleaned_text = clean_hindi_text(text)
            
            # Encode the text: converts Hindi string to token representation
            encoded = self.tokenizer.encode(cleaned_text)
            
            # Calculate compression ratio
            # Compression ratio = original character count / number of tokens
            original_char_count = len(cleaned_text)
            token_count = len(encoded.ids)
            compression_ratio = original_char_count / token_count if token_count > 0 else 0
            compression_percentage = (1 - token_count / original_char_count) * 100 if original_char_count > 0 else 0
            
            # Return structured encoding information:
            return {
                "token_ids": encoded.ids,  # List of integer token IDs
                                          # Each ID corresponds to a token in vocabulary
                "tokens": encoded.tokens,  # List of actual token strings (subwords)
                                          # Shows how text was broken into pieces
                "attention_mask": encoded.attention_mask,  # Binary mask (1s for real tokens, 0s for padding)
                                                           # Useful for batch processing
                "offsets": encoded.offsets,  # Character positions of each token in original text
                                           # Useful for mapping tokens back to original positions
                "compression_ratio": compression_ratio,  # Ratio of chars to tokens
                "compression_percentage": compression_percentage,  # Percentage reduction
                "original_char_count": original_char_count,  # Original character count
                "token_count": token_count  # Number of tokens
            }
        except Exception as e:
            # Return error if encoding fails (e.g., tokenizer not trained)
            return {"error": f"Encoding error: {str(e)}"}
    
    def decode(self, token_ids):
        """
        Decode token IDs back to Hindi text.
        
        Decoding Process:
        1. Takes list of token IDs
        2. Maps each ID to its corresponding token string
        3. Concatenates tokens to reconstruct original text
        4. Handles special tokens appropriately
        
        Args:
            token_ids: Can be:
                      - str: Comma-separated token IDs (e.g., "1, 2, 3")
                      - list: List of integers (e.g., [1, 2, 3])
        
        Returns:
            str: Decoded Hindi text, or error message if decoding fails
        """
        # Check if tokenizer is initialized
        if not self.tokenizer:
            return "Error: Tokenizer not initialized"
        
        try:
            # Handle different input formats for flexibility
            # Format 1: String with comma-separated IDs (from UI input)
            if isinstance(token_ids, str):
                if not token_ids.strip():
                    return ""  # Empty input returns empty string
                # Parse comma-separated string into list of integers
                # Split by comma, strip whitespace, convert to int, filter empty
                ids = [int(x.strip()) for x in token_ids.split(",") if x.strip()]
            # Format 2: Already a list of integers (from programmatic use)
            elif isinstance(token_ids, list):
                ids = token_ids
            else:
                return "Error: Invalid input format"
            
            # Decode: convert token IDs back to text
            # Automatically handles subword merging and special tokens
            decoded = self.tokenizer.decode(ids)
            return decoded
        except Exception as e:
            # Return error if decoding fails (e.g., invalid token IDs)
            return f"Decoding error: {str(e)}"
    
    def get_vocab_size(self):
        """
        Get the current vocabulary size of the tokenizer.
        
        Returns:
            int: Number of tokens in vocabulary, or 0 if tokenizer not initialized
                 For untrained tokenizer, this will be 0 or very small
                 For trained tokenizer, this matches the vocab_size used during training
        """
        # Check if tokenizer exists
        if not self.tokenizer:
            return 0
        try:
            # Get vocabulary size from tokenizer
            return self.tokenizer.get_vocab_size()
        except:
            # Return 0 if there's an error (e.g., tokenizer not properly initialized)
            return 0

