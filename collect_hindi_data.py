# ============================================================================
# HINDI DATA COLLECTOR
# ============================================================================
# Script to download and prepare Hindi Wikipedia dataset for BPE tokenizer training
# Downloads the full Hindi Wikipedia dataset from Hugging Face

import os
from pathlib import Path
from datasets import load_dataset
from hindi_preprocessor import clean_hindi_text
import sys


def format_size(size_bytes):
    """Format bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def collect_hindi_wikipedia(output_path="hindi_training_corpus.txt", use_full=True):
    """
    Download full Hindi Wikipedia dataset from Hugging Face.
    
    Args:
        output_path (str): Path to save the training corpus
        use_full (bool): If True, download full dataset; if False, use sample
    
    Returns:
        str: Path to saved corpus file
    """
    print("=" * 70)
    print("Hindi Wikipedia Dataset Collector")
    print("=" * 70)
    
    if use_full:
        print("\n📥 Downloading FULL Hindi Wikipedia dataset...")
        print("   This may take several minutes and requires ~300-500 MB disk space.")
        print("   Estimated download size: ~200-300 MB")
    else:
        print("\n📥 Downloading sample Hindi Wikipedia dataset (10,000 articles)...")
        print("   This is faster and requires ~30-50 MB disk space.")
    
    try:
        # Load the full Hindi Wikipedia dataset
        # The dataset is automatically cached by Hugging Face
        if use_full:
            print("\n⏳ Loading dataset (this may take a while on first run)...")
            dataset = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train")
        else:
            print("\n⏳ Loading sample dataset...")
            dataset = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train[:10000]")
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Total articles: {len(dataset):,}")
        
        # Extract and process text
        print("\n📝 Processing and cleaning text...")
        texts = []
        total_chars = 0
        
        for i, item in enumerate(dataset):
            if item.get('text'):
                text = item['text'].strip()
                if text and len(text) > 50:  # Filter very short articles
                    # Clean the text using our preprocessor
                    cleaned = clean_hindi_text(text)
                    if cleaned:
                        texts.append(cleaned)
                        total_chars += len(cleaned)
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"   Processed {i + 1:,} / {len(dataset):,} articles "
                      f"({(i + 1) / len(dataset) * 100:.1f}%)")
        
        print(f"\n✅ Processing complete!")
        print(f"   Valid articles: {len(texts):,}")
        print(f"   Total characters: {total_chars:,} ({format_size(total_chars)})")
        print(f"   Estimated words: ~{total_chars // 5:,}")
        
        # Combine all text
        print("\n💾 Saving corpus to file...")
        corpus = "\n\n".join(texts)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(corpus)
        
        # Get file size
        file_size = os.path.getsize(output_path)
        
        print(f"\n✅ Corpus saved successfully!")
        print(f"   File: {output_path}")
        print(f"   Size: {format_size(file_size)}")
        print(f"   Articles: {len(texts):,}")
        print(f"   Characters: {len(corpus):,}")
        
        return output_path
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (~500 MB)")
        print("3. Install required packages: pip install datasets")
        print("4. Try using sample mode: python collect_hindi_data.py --sample")
        return None


def main():
    """Main function to run the data collection."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Hindi Wikipedia dataset for BPE training"
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Download sample dataset (10k articles) instead of full dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='hindi_training_corpus.txt',
        help='Output file path (default: hindi_training_corpus.txt)'
    )
    
    args = parser.parse_args()
    
    # Check if output file already exists
    if os.path.exists(args.output):
        response = input(f"\n⚠️  File '{args.output}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Collect data
    result = collect_hindi_wikipedia(
        output_path=args.output,
        use_full=not args.sample
    )
    
    if result:
        print("\n" + "=" * 70)
        print("🎉 Data collection complete!")
        print("=" * 70)
        print(f"\nNext steps:")
        print(f"1. Train your tokenizer using the corpus:")
        print(f"   - Open the app: python app.py")
        print(f"   - Go to 'Train' tab")
        print(f"   - Load '{result}' or paste its contents")
        print(f"\n2. Or use programmatically:")
        print(f"   from hindi_bpe_encoder import HindiBPEEncoder")
        print(f"   encoder = HindiBPEEncoder()")
        print(f"   with open('{result}', 'r', encoding='utf-8') as f:")
        print(f"       corpus = f.read()")
        print(f"   encoder.train_tokenizer(corpus, vocab_size=5000)")
        print("\n" + "=" * 70)
    else:
        print("\n❌ Data collection failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

