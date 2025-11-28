# ============================================================================
# HINDI TEXT PREPROCESSOR (OPTIMIZED FOR PYTHON 3.11)
# ============================================================================
# This module contains regex-based preprocessing functions for Hindi text
# Handles Devanagari script patterns, punctuation, and normalization
# Optimized for large datasets with compiled regex patterns

import re
from typing import Iterator, Optional


# Devanagari Unicode ranges
# Devanagari: U+0900 to U+097F
# Devanagari Extended: U+A8E0 to U+A8FF
DEVANAGARI_PATTERN = r'[\u0900-\u097F\uA8E0-\uA8FF]+'

# Common Hindi punctuation marks
HINDI_PUNCTUATION = r'[।॥,;:!?\-—–()\[\]{}"\'\'""]'

# Numbers (both Devanagari and Arabic)
DEVANAGARI_NUMBERS = r'[\u0966-\u096F]'  # ०-९
ARABIC_NUMBERS = r'[0-9]'

# Pre-compile regex patterns for better performance (Python 3.11 optimization)
# Compiling once and reusing is much faster than compiling on each call
_WHITESPACE_PATTERN = re.compile(r'\s+')
_SPACE_BEFORE_PUNCT = re.compile(r'\s+([,;:!?।॥])')
_PUNCT_AFTER_SPACE = re.compile(r'([,;:!?।॥])([^\s])')
_OPEN_QUOTE_SPACE = re.compile(r'([""\'\([{])\s+')
_CLOSE_QUOTE_SPACE = re.compile(r'\s+([""\'\]}])')
_CLOSE_QUOTE_WORD = re.compile(r'([""\'\]}])([^\s])')
_DASH_NORMALIZE = re.compile(r'[—–]')
_QUOTE_NORMALIZE_DOUBLE = re.compile(r'[""]')
_QUOTE_NORMALIZE_SINGLE = re.compile(r'[\'\']')
_INVISIBLE_CHARS = re.compile(r'[\u200B-\u200D\uFEFF]')

def normalize_hindi_text(text):
    """
    Normalize Hindi text using regex patterns for proper formatting.
    
    This function:
    1. Normalizes whitespace
    2. Handles punctuation properly (no space before, one space after)
    3. Normalizes common characters (dashes, quotes)
    4. Removes invisible characters
    
    Args:
        text (str): Raw Hindi text
    
    Returns:
        str: Normalized Hindi text
    """
    if not text:
        return text
    
    # --- Step 1: Normalize multiple whitespaces to single space ---
    text = re.sub(r'\s+', ' ', text).strip()
    
    # --- Step 2: Handle punctuation properly ---
    
    # A. Remove any existing space *before* standard Hindi punctuation marks
    # (,, ;, :, !, ?, ।, ॥)
    text = re.sub(r'\s+([,;:!?।॥])', r'\1', text)

    # B. Ensure a single space *after* these punctuation marks
    # This correctly formats sentences like "है।अगला" to "है। अगला"
    text = re.sub(r'([,;:!?।॥])([^\s])', r'\1 \2', text)
    
    # C. Handle quotation marks and parentheses correctly
    # Hindi follows Western rules for these: no space inside the enclosure.
    
    # Remove space after opening quotes/parentheses
    text = re.sub(r'([“"‘\([{])\s+', r'\1', text)
    # Remove space before closing quotes/parentheses
    text = re.sub(r'\s+([”"’\])}])', r'\1', text)
    # Ensure a space *after* closing quotes/parentheses if a word follows
    text = re.sub(r'([”"’\])}])([^\s])', r'\1 \2', text)
    
    # --- Step 3: Normalize common characters ---

    # Normalize different types of dashes
    text = re.sub(r'[—–]', '-', text)
    
    # Normalize quotes (if you want to force straight quotes everywhere)
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'[‘’]', "'", text)
    
    # --- Step 4: Remove zero-width characters and other invisible characters ---
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)  # Zero-width spaces etc.
    
    # Strip leading/trailing spaces again after final operations
    text = text.strip()
    
    return text

# Example Usage:

# raw_text = """
#     यह एक उदाहरण पाठ है,जो दिखाताहै कि स्पेसिंग कितनी ज़रूरीहै!
#     "यह एक उद्धरण है।" क्या यह काम करता है?
#     यह।सही है॥हाँ
# """

# normalized_text = normalize_hindi_text(raw_text)

# print("Original Text:")
# print(raw_text)
# print("\nNormalized Text:")
# print(normalized_text)

# Expected Output:
# Normalized Text:
# यह एक उदाहरण पाठ है, जो दिखाता है कि स्पेसिंग कितनी ज़रूरी है! "यह एक उद्धरण है।" क्या यह काम करता है? यह। सही है॥ हाँ


def separate_punctuation(text):
    """
    Separate punctuation marks from Hindi words using regex.
    
    This helps the tokenizer handle punctuation better by treating
    it as separate tokens.
    
    Args:
        text (str): Hindi text
    
    Returns:
        str: Text with punctuation separated
    """
    if not text:
        return text
    
    # Separate punctuation from Devanagari words
    # Pattern: word followed by punctuation or punctuation followed by word
    text = re.sub(r'([\u0900-\u097F\uA8E0-\uA8FF]+)([।॥,;:!?\-()\[\]{}"\'])', r'\1 \2', text)
    text = re.sub(r'([।॥,;:!?\-()\[\]{}"\'])([\u0900-\u097F\uA8E0-\uA8FF]+)', r'\1 \2', text)
    
    return text


def clean_hindi_text(text: str) -> str:
    """
    Comprehensive cleaning of Hindi text using regex.
    
    Combines normalization and punctuation separation.
    Optimized for large text processing.
    
    Args:
        text (str): Raw Hindi text
    
    Returns:
        str: Cleaned and normalized Hindi text
    """
    if not text:
        return text
    
    # Apply normalization (uses pre-compiled patterns)
    text = normalize_hindi_text(text)
    
    # Separate punctuation
    text = separate_punctuation(text)
    
    # Final whitespace normalization (use pre-compiled pattern)
    text = _WHITESPACE_PATTERN.sub(' ', text).strip()
    
    return text


def clean_hindi_text_streaming(file_path: str, chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """
    Stream and clean large Hindi text files in chunks to avoid memory issues.
    
    This is optimized for very large files (>100MB) that don't fit in memory.
    
    Args:
        file_path (str): Path to the text file
        chunk_size (int): Size of chunks to read (default: 1MB)
    
    Yields:
        str: Cleaned text chunks
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        buffer = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                if buffer:
                    yield clean_hindi_text(buffer)
                break
            
            buffer += chunk
            
            # Process complete lines to avoid breaking in the middle of text
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if line.strip():
                    cleaned = clean_hindi_text(line)
                    if cleaned:
                        yield cleaned


def extract_hindi_words(text):
    """
    Extract only Hindi words (Devanagari script) from text using regex.
    
    Useful for filtering out non-Hindi content.
    
    Args:
        text (str): Mixed text
    
    Returns:
        list: List of Hindi words found
    """
    if not text:
        return []
    
    # Find all Devanagari words
    hindi_words = re.findall(DEVANAGARI_PATTERN, text)
    return hindi_words


def is_hindi_text(text):
    """
    Check if text contains Hindi (Devanagari script) using regex.
    
    Args:
        text (str): Text to check
    
    Returns:
        bool: True if text contains Devanagari characters
    """
    if not text:
        return False
    
    return bool(re.search(DEVANAGARI_PATTERN, text))


def filter_hindi_only(text: str, min_hindi_ratio: float = 0.7) -> str:
    """
    Filter text to keep only lines/sentences with significant Hindi content.
    
    This ensures the tokenizer only learns from Hindi text, not mixed content.
    Removes non-Hindi characters and keeps only Devanagari script with allowed punctuation.
    
    Args:
        text (str): Input text (may contain mixed Hindi/English/other)
        min_hindi_ratio (float): Minimum ratio of Devanagari chars to keep a line (0.0-1.0)
                                Default 0.7 means at least 70% Hindi characters
    
    Returns:
        str: Filtered text containing only Hindi-dominant lines
    """
    if not text:
        return text
    
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Count Devanagari characters
        devanagari_chars = len(re.findall(DEVANAGARI_PATTERN, line))
        total_chars = len(re.sub(r'\s', '', line))  # Exclude whitespace
        
        if total_chars == 0:
            continue
        
        # Calculate ratio of Hindi characters
        hindi_ratio = devanagari_chars / total_chars if total_chars > 0 else 0
        
        # Keep line if it has sufficient Hindi content
        if hindi_ratio >= min_hindi_ratio:
            # Extract only Devanagari words and allowed punctuation
            # Keep Hindi words, Hindi punctuation (।॥), and basic punctuation
            hindi_line = re.sub(
                r'[^\u0900-\u097F\uA8E0-\uA8FF\s।॥,;:!?\-()\[\]{}"\']+',
                ' ',
                line
            )
            # Normalize whitespace
            hindi_line = _WHITESPACE_PATTERN.sub(' ', hindi_line).strip()
            if hindi_line:
                filtered_lines.append(hindi_line)
    
    return '\n'.join(filtered_lines)

