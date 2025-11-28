# ============================================================================
# TOKENIZER TRAINING SCRIPT (OPTIMIZED FOR PYTHON 3.11)
# ============================================================================
# Script to train the Hindi BPE tokenizer on a corpus file
# Optimized for large datasets with streaming and progress tracking

from hindi_bpe_encoder import HindiBPEEncoder
import os
import sys
import time
from pathlib import Path


def train_from_file(corpus_file="my_corpus.txt", vocab_size=5000):
    """
    Train the Hindi BPE tokenizer from a corpus file.
    
    Args:
        corpus_file (str): Path to the corpus file
        vocab_size (int): Vocabulary size for the tokenizer
    """
    print("=" * 70)
    print("Hindi BPE Tokenizer Training")
    print("=" * 70)
    
    # Check if corpus file exists
    if not os.path.exists(corpus_file):
        print(f"❌ Error: Corpus file '{corpus_file}' not found!")
        print(f"   Please make sure the file exists in the current directory.")
        return False
    
    # Get file size
    file_size = os.path.getsize(corpus_file)
    print(f"\n📄 Corpus file: {corpus_file}")
    print(f"   Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    # Initialize encoder
    print("\n🔧 Initializing Hindi BPE Encoder...")
    encoder = HindiBPEEncoder()
    
    # Check file size to determine if we should use streaming
    file_size_mb = file_size / (1024 * 1024)
    use_streaming = file_size_mb > 100  # Use streaming for files > 100MB
    
    if use_streaming:
        print(f"\n📖 Using streaming mode for large corpus ({file_size_mb:.1f} MB)...")
        print("   This avoids loading the entire file into memory.")
        
        # Train tokenizer with streaming
        print(f"\n🎓 Training tokenizer with vocab_size={vocab_size}...")
        print("   Processing in chunks - this may take several minutes...")
        
        start_time = time.time()
        try:
            result = encoder.train_tokenizer(
                corpus_file,  # Pass file path instead of content
                vocab_size=vocab_size,
                use_streaming=True
            )
            elapsed_time = time.time() - start_time
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        # Load corpus into memory (for smaller files)
        print(f"\n📖 Loading corpus from '{corpus_file}'...")
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                corpus = f.read()
            
            if not corpus or not corpus.strip():
                print("❌ Error: Corpus file is empty!")
                return False
            
            print(f"   Loaded {len(corpus):,} characters")
            print(f"   Estimated words: ~{len(corpus.split()):,}")
            
        except Exception as e:
            print(f"❌ Error reading corpus file: {e}")
            return False
        
        # Train tokenizer
        print(f"\n🎓 Training tokenizer with vocab_size={vocab_size}...")
        print("   This may take a few minutes depending on corpus size...")
        
        start_time = time.time()
        try:
            result = encoder.train_tokenizer(corpus, vocab_size=vocab_size)
            elapsed_time = time.time() - start_time
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n✅ {result}")
        
        # Show vocabulary size and performance stats
        vocab_size_actual = encoder.get_vocab_size()
        print(f"\n📊 Tokenizer Statistics:")
        print(f"   Vocabulary size: {vocab_size_actual:,}")
        print(f"   Tokenizer saved to: {encoder.tokenizer_path}")
        print(f"   Training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        if file_size_mb > 0:
            print(f"   Processing speed: {file_size_mb/elapsed_time:.2f} MB/s")
        
        print("\n" + "=" * 70)
        print("🎉 Training complete!")
        print("=" * 70)
        print("\nYou can now use the trained tokenizer in your app:")
        print("   python3 app.py")
        
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Hindi BPE tokenizer from corpus file"
    )
    parser.add_argument(
        '--corpus',
        type=str,
        default='my_corpus.txt',
        help='Path to corpus file (default: my_corpus.txt)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=5000,
        help='Vocabulary size (default: 5000, uses 256 base tokens so merges are always learned)'
    )
    
    args = parser.parse_args()
    
    success = train_from_file(
        corpus_file=args.corpus,
        vocab_size=args.vocab_size
    )
    
    sys.exit(0 if success else 1)

