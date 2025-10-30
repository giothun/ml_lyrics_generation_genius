import json
import os
import re
import argparse
import multiprocessing
from pathlib import Path
from functools import partial
from typing import Dict, List

parser = argparse.ArgumentParser(description='Train n-gram language model on text data')
parser.add_argument('-indir', '--input-dir',
                    help='path to directory for training data',
                    default="data")
parser.add_argument('-m', '--model', type=str, help='path to save trained model',
                    default="all_grams.txt")
parser.add_argument('-n', '--ngram-size', type=int, help='maximum n-gram size (default: 3)',
                    default=3)
parser.add_argument('-w', '--workers', type=int, 
                    help='number of parallel workers (default: auto-detect)',
                    default=None)
parser.add_argument('--no-parallel', action='store_true',
                    help='disable parallel processing')

args = parser.parse_args()

PUNCTUATION_PATTERN = re.compile(r'[^\w\s]+|[\d]+')
WHITESPACE_PATTERN = re.compile(r' +')
BRACKET_PATTERN = re.compile(r"[\(\[].*?[\)\]]")


def clean_text(text):
    """Clean and normalize text for training.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned and normalized text string
    """
    if '\n' in text:
        text = text[text.find('\n'):]
    
    replacements = {
        'Embed': ' ', 'Lyrics': ' ',
        '—': ' ', '»': ' ', '«': ' ', '–': ' ', '…': ' ', '-': ' ',
        '?': '', ',': '', '.': '', '!': '', ':': '',
        '\n ': '\n', '\n\n': '\n', '\u2005': ' '
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    text = BRACKET_PATTERN.sub("", text)
    text = PUNCTUATION_PATTERN.sub(r'', text).strip()
    text = text.lower()
    text = text.replace('\n', ' ')
    text = WHITESPACE_PATTERN.sub(' ', text)
    text = text.strip()
    
    return text


def process_single_file(file_path: Path, ngram_size: int) -> Dict[str, int]:
    """Process a single file and return its n-gram dictionary.
    
    This function is designed to be picklable for multiprocessing.
    
    Args:
        file_path: Path to the text file
        ngram_size: Maximum n-gram size to compute
        
    Returns:
        Dictionary of n-grams with their frequencies
    """
    try:
        with open(file_path, encoding='utf-8') as f:
            text = f.read()
        
        if not text.strip():
            return {}
        
        cleaned = clean_text(text)
        if not cleaned:
            return {}
        
        return calculate_ngrams(cleaned, ngram_size)
        
    except UnicodeDecodeError:
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, encoding=encoding) as f:
                    text = f.read()
                cleaned = clean_text(text)
                if cleaned:
                    return calculate_ngrams(cleaned, ngram_size)
            except:
                continue
        print(f"Warning: Could not decode {file_path}")
        return {}
    except Exception as e:
        print(f"Warning: Could not process {file_path}: {e}")
        return {}


def calculate_ngrams(text: str, ngram_size: int) -> Dict[str, int]:
    """Calculate n-grams from cleaned text (optimized single-pass).
    
    Args:
        text: Cleaned text string
        ngram_size: Maximum n-gram size
        
    Returns:
        Dictionary of n-grams with frequencies
    """
    grams = {}
    text = WHITESPACE_PATTERN.sub(' ', text)
    arr = [word for word in text.split(' ') if word]
    
    if len(arr) < 1:
        return grams
    
    for i in range(len(arr)):
        for n in range(1, min(ngram_size, len(arr) - i) + 1):
            ngram = " ".join(arr[i:i+n])
            grams[ngram] = grams.get(ngram, 0) + 1
    
    return grams


def merge_ngram_dicts(dict_list: List[Dict[str, int]]) -> Dict[str, int]:
    """Merge multiple n-gram dictionaries by summing frequencies.
    
    Args:
        dict_list: List of n-gram dictionaries
        
    Returns:
        Combined dictionary
    """
    combined = {}
    for grams_dict in dict_list:
        for gram, count in grams_dict.items():
            combined[gram] = combined.get(gram, 0) + count
    return combined


def make_data(texts, ngram_size=3):
    """Build n-gram frequency dictionary from texts.
    
    Legacy function for backward compatibility.
    
    Args:
        texts: List of text strings
        ngram_size: Maximum n-gram size to compute
        
    Returns:
        Dictionary mapping n-grams to their frequencies
    """
    all_grams = {}
    cleaned_texts = []
    
    for text in texts:
        cleaned = clean_text(text)
        if cleaned:
            cleaned_texts.append(cleaned)
    
    if not cleaned_texts:
        print("Warning: No valid text data found after cleaning")
        return all_grams
    
    for text in cleaned_texts:
        grams = calculate_ngrams(text, ngram_size)
        all_grams = merge_ngram_dicts([all_grams, grams])

    return all_grams


if __name__ == '__main__':
    model = args.model
    input_dir = args.input_dir
    ngram_size = args.ngram_size
    use_parallel = not args.no_parallel

    texts = []
    
    if input_dir is None:
        print("Enter text (type 'end' on a new line to finish):")
        text = ""
        while True:
            try:
                line = input()
                if line == "end":
                    break
                text += line + "\n"
            except EOFError:
                break
        if text.strip():
            texts.append(text)
        
        print(f"Training {ngram_size}-gram model...")
        all_grams = make_data(texts, ngram_size)
    else:
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"Error: Directory '{input_dir}' does not exist")
            exit(1)
        
        if not input_path.is_dir():
            print(f"Error: '{input_dir}' is not a directory")
            exit(1)
        
        txt_files = list(input_path.glob('*.txt'))
        
        if not txt_files:
            print(f"Warning: No .txt files found in '{input_dir}'")
            exit(1)
        
        print(f"Found {len(txt_files)} text files")
        
        if args.workers is not None:
            num_workers = args.workers
        else:
            cpu_count = multiprocessing.cpu_count()
            num_workers = min(cpu_count - 1, len(txt_files))
            num_workers = max(1, num_workers)
        
        print(f"Training {ngram_size}-gram model...")
        
        if use_parallel and len(txt_files) > 1 and num_workers > 1:
            print(f"Using {num_workers} parallel workers")
            
            with multiprocessing.Pool(processes=num_workers) as pool:
                process_func = partial(process_single_file, ngram_size=ngram_size)
                print("Processing files...")
                results = pool.map(process_func, txt_files)
                print("Merging results...")
                all_grams = merge_ngram_dicts(results)
        else:
            if not use_parallel:
                print("Parallel processing disabled")
            else:
                print("Using sequential processing")
            
            results = []
            for i, file_path in enumerate(txt_files, 1):
                if i % 100 == 0:
                    print(f"  Processed {i}/{len(txt_files)} files...")
                result = process_single_file(file_path, ngram_size)
                results.append(result)
            
            all_grams = merge_ngram_dicts(results)
    
    print(f"Generated {len(all_grams)} unique n-grams")
    
    gram_counts = {}
    for gram in all_grams.keys():
        size = len(gram.split(' '))
        gram_counts[size] = gram_counts.get(size, 0) + 1
    
    for size in sorted(gram_counts.keys()):
        print(f"  {size}-grams: {gram_counts[size]}")
    
    print(f"Saving model to '{model}'...")
    try:
        with open(model, 'w', encoding='utf-8') as f:
            json.dump(all_grams, f, ensure_ascii=False)
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")
        exit(1)
