# Hindi BPE Encoder - Hugging Face Space

A Byte Pair Encoding (BPE) tokenizer for Hindi text, deployed as a Hugging Face Space application. Built with a modular architecture for maintainability and extensibility.

## Features

- **Encode**: Convert Hindi text into token IDs and subword tokens
- **Decode**: Convert token IDs back to Hindi text
- **Train**: Train the tokenizer on your custom Hindi text corpus
- **Interactive UI**: User-friendly Gradio interface with tabbed navigation
- **Modular Design**: Clean separation of concerns for easy maintenance and extension

## Usage

### Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
python app.py
```

3. Open your browser to `http://localhost:7860`

### Deploying to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Select "Gradio" as the SDK
3. Upload all files from this repository
4. The app will automatically deploy

## Project Structure

The project follows a modular architecture with clear separation of concerns:

```
ERAv4S11/
├── app.py                      # Main entry point - orchestrates all components
├── hindi_bpe_encoder.py        # Core tokenizer class (BPE implementation)
├── hindi_preprocessor.py      # Regex-based Hindi text preprocessing
├── gradio_handlers.py          # Event handlers connecting UI to tokenizer
├── gradio_ui.py                # Gradio UI components and layout
├── constants.py                # Configuration constants and sample data
├── train_tokenizer.py         # Script to train tokenizer from corpus
├── collect_hindi_data.py       # Script to download Hindi Wikipedia dataset
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── hindi_bpe_tokenizer.json    # Saved tokenizer (created after training)
└── hindi_training_corpus.txt   # Training corpus (created by collect_hindi_data.py)
```

### Module Descriptions

- **`app.py`**: Main application entry point that initializes the encoder, creates handlers, builds the UI, and launches the Gradio app.

- **`hindi_bpe_encoder.py`**: Core tokenizer module containing the `HindiBPEEncoder` class. Handles:
  - Tokenizer initialization with byte-level BPE and Unicode normalization
  - Training on Hindi text corpus with streaming support
  - Encoding text to token IDs
  - Decoding token IDs back to text

- **`hindi_preprocessor.py`**: Hindi text preprocessing module with regex-based functions:
  - Text normalization (whitespace, punctuation, quotes)
  - Hindi-specific character handling
  - Streaming support for large files
  - Filtering functions for Hindi content

- **`gradio_handlers.py`**: Contains factory functions that create event handlers bound to the encoder instance. These functions format inputs/outputs for the Gradio interface.

- **`gradio_ui.py`**: Defines all Gradio UI components:
  - Tab creation functions (Encode, Decode, Train)
  - Complete app interface builder
  - UI layout and styling

- **`constants.py`**: Centralized configuration:
  - Sample Hindi text for training
  - Example sentences for the Encode tab
  - Vocabulary size limits and defaults
  - App configuration constants

## How It Works

The BPE (Byte Pair Encoding) tokenizer:
- Splits Hindi text into subword units
- Handles out-of-vocabulary words by breaking them into known subwords
- Uses whitespace pre-tokenization suitable for Hindi
- Supports special tokens: `<unk>`, `<s>`, `</s>`, `<pad>`, `<mask>`

### Technical Implementation Details

#### Byte-Level BPE with UTF-8 Encoding
- **256 Base Tokens**: The tokenizer starts with exactly 256 base tokens (one for each byte value 0-255)
- **UTF-8 Byte Encoding**: Text is first encoded as UTF-8 bytes, then BPE learns merges on these bytes
- **Universal Coverage**: With `byte_fallback=True`, any Unicode character can be handled, even if not seen during training
- **Why This Matters**: Ensures BPE always learns merges regardless of vocabulary size, and can handle any Hindi character or rare Unicode symbol

#### Unicode Normalization (NFD)
- **Normalization Form Decomposed (NFD)**: All text is normalized to NFD before tokenization
- **Consistent Representation**: Prevents issues where the same character in different Unicode forms (composed vs decomposed) is treated as different tokens
- **Real-World Impact**: Fixes tokenization issues with words like "छोड़कर" that might otherwise be split incorrectly
- **Example**: "छोड़कर" will always be normalized consistently, preventing `<unk>` tokens in the middle of valid Hindi words

#### Vocabulary Size
- **Recommended Size**: 10,000 tokens for Hindi (default: 5,000, can be increased)
- **Breakdown**: 
  - 256 base tokens (bytes 0-255)
  - 5 special tokens (`<unk>`, `<s>`, `</s>`, `<pad>`, `<mask>`)
  - ~9,739 merged tokens (learned subword units)
- **Coverage**: Larger vocab sizes (10,000+) provide better coverage of Hindi character combinations
- **Trade-offs**: 
  - Larger vocab = better coverage, fewer `<unk>` tokens, but slower encoding
  - Smaller vocab = faster, but may miss rare patterns
- **Min Frequency**: Set to 2 (can be lowered to 1 for better coverage of rare patterns)

#### Text Preprocessing
The `clean_hindi_text()` function performs regex-based preprocessing:
- **Whitespace Normalization**: Multiple spaces/tabs/newlines → single space
- **Punctuation Handling**: Proper spacing around Hindi punctuation (।, ॥) and standard punctuation
- **Quote Normalization**: Standardizes different quote types (curly quotes → straight quotes)
- **Dash Normalization**: Normalizes different dash types (em dash, en dash → hyphen)
- **Invisible Character Removal**: Removes zero-width spaces and other invisible Unicode characters
- **Purpose**: Ensures consistent text representation for better tokenization quality

### Architecture Flow

```
User Input (Gradio UI)
    ↓
gradio_handlers.py (Format & Validate)
    ↓
hindi_bpe_encoder.py (Process)
    ↓
gradio_handlers.py (Format Output)
    ↓
Gradio UI (Display Results)
```

## Example

**Input (Hindi):**
```
हिंदी भारत की राष्ट्रभाषा है।
```

**Output:**
- Token IDs: `[1, 2, 3, 4, 5, 6]`
- Tokens: `['हिंदी', 'भारत', 'की', 'राष्ट्रभाषा', 'है', '।']`

## Dataset

### Training Data Source
- **Dataset**: Hindi Wikipedia from Hugging Face (`wikimedia/wikipedia`, version `20231101.hi`)
- **Collection Script**: `collect_hindi_data.py` - Downloads and processes the dataset
- **Size**: Full dataset contains ~320,000+ articles (~300-500 MB)
- **Processing**: 
  - Extracts article text
  - Cleans using `clean_hindi_text()` for normalization
  - Filters very short articles (< 50 characters)
  - Saves to `hindi_training_corpus.txt`

### Using the Dataset Collector

```bash
# Download full Hindi Wikipedia dataset
python3 collect_hindi_data.py

# Download sample dataset (10,000 articles, faster)
python3 collect_hindi_data.py --sample

# Specify output file
python3 collect_hindi_data.py --output my_corpus.txt
```

The collected corpus can then be used to train the tokenizer:

```bash
python3 train_tokenizer.py --corpus hindi_training_corpus.txt --vocab-size 10000
```

## Key Learnings & Best Practices

### 1. Vocabulary Size Selection
- **For Hindi**: 10,000 tokens provides good coverage
- **Base Tokens**: Always 256 (one per byte) - these are guaranteed
- **Merged Tokens**: The remaining tokens are learned BPE merges
- **Rule of Thumb**: `vocab_size` should be > 256 to ensure merges are learned
- **Too Small**: If vocab_size < unique characters, BPE just collects characters without learning merges
- **Too Large**: Diminishing returns, slower encoding, but better rare pattern coverage

### 2. UTF-8 and Byte-Level BPE
- **Why Byte-Level**: Handles any Unicode character, even rare ones not in training data
- **How It Works**: 
  1. Text → UTF-8 bytes
  2. BPE learns merges on byte sequences
  3. Unknown characters automatically encoded as bytes
- **Advantage**: No `<unk>` tokens for valid Unicode characters (with proper normalization)
- **Trade-off**: Slightly larger vocabulary needed, but universal coverage

### 3. Unicode Normalization is Critical
- **Problem**: Same character can exist in multiple Unicode forms (composed vs decomposed)
- **Example**: "छोड़कर" might be stored as:
  - Composed: `छ` + `ो` + `ड़` + `क` + `र` (single code points)
  - Decomposed: `छ` + `ो` + `ड` + `़` + `क` + `र` (base + combining marks)
- **Solution**: NFD normalization ensures consistent representation
- **Impact**: Without normalization, same word might tokenize differently → `<unk>` tokens
- **Best Practice**: Always normalize before training and encoding

### 4. Preprocessing Matters
- **Consistency**: Same preprocessing must be used during training and encoding
- **Normalization**: Regex-based cleaning ensures consistent whitespace, punctuation, quotes
- **Filtering**: Can optionally filter non-Hindi content, but be careful not to be too aggressive
- **Performance**: Pre-compiled regex patterns (Python 3.11 optimization) for faster processing

### 5. Training Best Practices
- **Large Corpora**: Use streaming mode for files > 100MB to avoid memory issues
- **Min Frequency**: Start with 2, lower to 1 if you need more rare patterns
- **Validation**: Always check that merges were learned (not just character collection)
- **Retraining**: Delete old tokenizer file before retraining with new settings

## Development

### Extending the Application

The modular structure makes it easy to extend:

1. **Add new features**: Create new handler functions in `gradio_handlers.py` and corresponding UI components in `gradio_ui.py`
2. **Modify tokenizer**: Update `hindi_bpe_encoder.py` without touching UI code
3. **Change configuration**: Update constants in `constants.py`
4. **Customize UI**: Modify `gradio_ui.py` to change layout, styling, or add new tabs

### Using the Tokenizer Programmatically

You can import and use the tokenizer in other Python scripts:

```python
from hindi_bpe_encoder import HindiBPEEncoder

# Initialize encoder
encoder = HindiBPEEncoder()

# Train on your corpus
encoder.train_tokenizer(hindi_text, vocab_size=5000)

# Encode text
result = encoder.encode("हिंदी में कुछ लिखें")
print(result["token_ids"])

# Decode tokens
text = encoder.decode([1, 2, 3, 4])
print(text)
```

## Files

- `app.py`: Main application entry point
- `hindi_bpe_encoder.py`: Core BPE tokenizer implementation with byte-level BPE and Unicode normalization
- `hindi_preprocessor.py`: Regex-based Hindi text preprocessing and normalization
- `gradio_handlers.py`: Gradio event handlers
- `gradio_ui.py`: Gradio UI components
- `constants.py`: Configuration and constants
- `train_tokenizer.py`: Script to train tokenizer from corpus file
- `collect_hindi_data.py`: Script to download Hindi Wikipedia dataset
- `requirements.txt`: Python dependencies
- `README.md`: This file
- `hindi_bpe_tokenizer.json`: Saved tokenizer (created after training)
- `hindi_training_corpus.txt`: Training corpus (created by collect_hindi_data.py)

## License

MIT License
