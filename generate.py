import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

parser = argparse.ArgumentParser(description='Generate text using trained n-gram language model')
parser.add_argument('-m', '--model', type=str, help='path to the trained model file',
                    default="all_grams.txt")
parser.add_argument('-p', '--prefix', nargs='+',
                    help='optional starting words for generation. If not specified, '
                         'starts with random words from the model.',
                    default=None)
parser.add_argument('-l', '--length', type=int, help='number of words to generate', default=20)
parser.add_argument('-o', '--output', type=str, help='optional output file path', default=None)
parser.add_argument('--use-interpolation', action='store_true',
                   help='use interpolation instead of simple backoff (slower but better quality)')
parser.add_argument('--lambda3', type=float, default=0.6,
                   help='weight for trigram probability (only with --use-interpolation, default: 0.6)')
parser.add_argument('--lambda2', type=float, default=0.3,
                   help='weight for bigram probability (only with --use-interpolation, default: 0.3)')
parser.add_argument('--lambda1', type=float, default=0.1,
                   help='weight for unigram probability (only with --use-interpolation, default: 0.1)')
parser.add_argument('--temperature', type=float, default=1.0,
                   help='temperature for controlling randomness (only with --use-interpolation, 0.5-2.0, default: 1.0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                   help='add-k smoothing parameter (only with --use-interpolation, default: 0.1)')
args = parser.parse_args()


def clean_prefix(text):
    """Clean and normalize prefix text to match training format.
    
    Args:
        text: Raw prefix text
        
    Returns:
        Cleaned prefix string
    """
    text = re.sub(r'[^\w\s]+|[\d]+', r'', text).strip()
    text = text.strip()
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r' +', ' ', text)
    return text


def separate_grams(all_grams):
    """Separate n-grams by size and build efficient lookup structures.
    
    Args:
        all_grams: Dictionary of all n-grams with frequencies
        
    Returns:
        Tuple of (one_grams, two_grams, three_grams, context_map)
    """
    one_grams = {}
    two_grams = {}
    three_grams = {}
    context_map_3 = defaultdict(list)
    context_map_2 = defaultdict(list)
    
    for gram, count in all_grams.items():
        words = gram.split(' ')
        size = len(words)
        
        if size == 1:
            one_grams[gram] = count
        elif size == 2:
            two_grams[gram] = count
            context_map_2[words[0]].append((words[1], count))
        elif size == 3:
            three_grams[gram] = count
            context_map_3[(words[0], words[1])].append((words[2], count))

    return one_grams, two_grams, three_grams, context_map_2, context_map_3


def sample_next_word_3gram(context, one_grams, two_grams, context_map_3):
    """Sample next word using 3-gram model with backoff.
    
    Args:
        context: Tuple of (word1, word2) for context
        one_grams: 1-gram frequency dict
        two_grams: 2-gram frequency dict  
        context_map_3: Map from context to [(next_word, count), ...]
        
    Returns:
        Next word or None if no continuation possible
    """
    candidates = context_map_3.get(context, [])
    
    if candidates:
        context_str = " ".join(context)
        context_count = two_grams.get(context_str, 0)
        
        if context_count > 0:
            words = [w for w, _ in candidates]
            probs = np.array([count / context_count for _, count in candidates])
            probs = probs / probs.sum()
            
            return np.random.choice(words, p=probs)
    
    return None


def sample_next_word_2gram(prev_word, one_grams, context_map_2):
    """Sample next word using 2-gram model with backoff.
    
    Args:
        prev_word: Previous word for context
        one_grams: 1-gram frequency dict
        context_map_2: Map from word to [(next_word, count), ...]
        
    Returns:
        Next word or None if no continuation possible
    """
    candidates = context_map_2.get(prev_word, [])
    
    if candidates:
        prev_count = one_grams.get(prev_word, 0)
        
        if prev_count > 0:
            words = [w for w, _ in candidates]
            probs = np.array([count / prev_count for _, count in candidates])
            probs = probs / probs.sum()
            
            return np.random.choice(words, p=probs)
    
    return None


def sample_random_word(one_grams):
    """Sample a random word from 1-gram distribution.
    
    Args:
        one_grams: 1-gram frequency dict
        
    Returns:
        Random word
    """
    words = list(one_grams.keys())
    counts = np.array([one_grams[w] for w in words])
    probs = counts / counts.sum()
    
    return np.random.choice(words, p=probs)


def apply_temperature(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling to probability distribution.
    
    Args:
        probs: Probability distribution
        temperature: Temperature parameter (< 1.0 = more conservative, > 1.0 = more random)
        
    Returns:
        Temperature-scaled probability distribution
    """
    if temperature == 1.0:
        return probs
    
    if temperature <= 0:
        temperature = 0.01
    
    log_probs = np.log(probs + 1e-10)
    scaled_log_probs = log_probs / temperature
    scaled_probs = np.exp(scaled_log_probs)
    return scaled_probs / scaled_probs.sum()


def compute_interpolated_probability(w3: str, w2: str, w1: str, 
                                     one_grams: dict, two_grams: dict, three_grams: dict,
                                     lambdas: tuple = (0.6, 0.3, 0.1),
                                     smoothing_k: float = 0.1) -> float:
    """Compute interpolated probability using all n-gram levels.
    
    Args:
        w3: Word to predict
        w2: Previous word (context)
        w1: Word before w2 (longer context)
        one_grams: Unigram frequency dict
        two_grams: Bigram frequency dict
        three_grams: Trigram frequency dict
        lambdas: Tuple of (lambda3, lambda2, lambda1) weights
        smoothing_k: Add-k smoothing parameter
        
    Returns:
        Interpolated probability
    """
    lambda3, lambda2, lambda1 = lambdas
    vocab_size = len(one_grams)
    total_words = sum(one_grams.values())
    
    p1 = (one_grams.get(w3, 0) + smoothing_k) / (total_words + smoothing_k * vocab_size)
    
    bigram = f"{w2} {w3}"
    bigram_count = two_grams.get(bigram, 0)
    w2_count = one_grams.get(w2, 0)
    
    if w2_count > 0:
        p2 = (bigram_count + smoothing_k) / (w2_count + smoothing_k * vocab_size)
    else:
        p2 = smoothing_k / (smoothing_k * vocab_size)
    
    trigram = f"{w1} {w2} {w3}"
    bigram_context = f"{w1} {w2}"
    trigram_count = three_grams.get(trigram, 0)
    context_count = two_grams.get(bigram_context, 0)
    
    if context_count > 0:
        p3 = (trigram_count + smoothing_k) / (context_count + smoothing_k * vocab_size)
    else:
        p3 = smoothing_k / (smoothing_k * vocab_size)
    
    return lambda3 * p3 + lambda2 * p2 + lambda1 * p1


def sample_with_interpolation(context: tuple, one_grams: dict, two_grams: dict, three_grams: dict,
                              lambdas: tuple = (0.6, 0.3, 0.1), 
                              smoothing_k: float = 0.1,
                              temperature: float = 1.0) -> str:
    """Sample next word using interpolated probabilities.
    
    Args:
        context: Tuple of (w1, w2) context words
        one_grams: Unigram frequency dict
        two_grams: Bigram frequency dict
        three_grams: Trigram frequency dict
        lambdas: Interpolation weights
        smoothing_k: Smoothing parameter
        temperature: Temperature for sampling
        
    Returns:
        Next word sampled from interpolated distribution
    """
    w1, w2 = context
    candidates = set()
    
    for trigram in three_grams.keys():
        words = trigram.split(' ')
        if len(words) == 3 and words[0] == w1 and words[1] == w2:
            candidates.add(words[2])
    
    for bigram in two_grams.keys():
        words = bigram.split(' ')
        if len(words) == 2 and words[0] == w2:
            candidates.add(words[1])
    
    if len(candidates) < 10:
        sorted_unigrams = sorted(one_grams.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_unigrams[:50]:
            candidates.add(word)
    
    if not candidates:
        return sample_random_word(one_grams)
    
    candidates_list = list(candidates)
    probs = np.array([
        compute_interpolated_probability(w, w2, w1, one_grams, two_grams, three_grams, lambdas, smoothing_k)
        for w in candidates_list
    ])
    
    probs = probs / probs.sum()
    probs = apply_temperature(probs, temperature)
    
    return np.random.choice(candidates_list, p=probs)


def make_text(all_grams, length, prefix=None, lambdas=(0.6, 0.3, 0.1), 
              temperature=1.0, smoothing_k=0.1, use_interpolation=False):
    """Generate text using n-gram language model with backoff (default) or interpolation.
    
    Args:
        all_grams: Dictionary of all n-grams with frequencies
        length: Number of words to generate
        prefix: Optional starting words (string or None)
        lambdas: Tuple of (lambda3, lambda2, lambda1) for interpolation
        temperature: Temperature for controlling randomness
        smoothing_k: Add-k smoothing parameter
        use_interpolation: If True, use interpolation (slower) instead of backoff
        
    Returns:
        Generated text string
    """
    one_grams, two_grams, three_grams, context_map_2, context_map_3 = separate_grams(all_grams)
    
    if not one_grams:
        return "Error: Model is empty"
    
    arr = []
    
    if prefix is None or prefix == "":
        if two_grams:
            random_bigram = np.random.choice(list(two_grams.keys()))
            arr = random_bigram.split(' ')
        else:
            arr = [sample_random_word(one_grams), sample_random_word(one_grams)]
    else:
        if isinstance(prefix, list):
            prefix = " ".join(prefix)
        
        prefix = clean_prefix(prefix)
        arr = [w for w in prefix.split(" ") if w]
        
        if not arr:
            print("Warning: Empty prefix after cleaning, using random start")
            random_bigram = np.random.choice(list(two_grams.keys()))
            arr = random_bigram.split(' ')
        else:
            valid_words = []
            for word in arr:
                if word in one_grams:
                    valid_words.append(word)
                else:
                    print(f"Warning: '{word}' not in vocabulary, skipping")
            
            if not valid_words:
                print("Warning: No valid words in prefix, using random start")
                random_bigram = np.random.choice(list(two_grams.keys()))
                arr = random_bigram.split(' ')
            else:
                arr = valid_words
                
                if len(arr) == 1:
                    next_word = sample_next_word_2gram(arr[0], one_grams, context_map_2)
                    if next_word:
                        arr.append(next_word)
                    else:
                        arr.append(sample_random_word(one_grams))
    
    for _ in range(length):
        if len(arr) >= 2 and use_interpolation:
            context = (arr[-2], arr[-1])
            next_word = sample_with_interpolation(
                context, one_grams, two_grams, three_grams,
                lambdas, smoothing_k, temperature
            )
            arr.append(next_word)
        elif len(arr) >= 2:
            context = (arr[-2], arr[-1])
            next_word = sample_next_word_3gram(context, one_grams, two_grams, context_map_3)
            
            if next_word:
                arr.append(next_word)
            else:
                next_word = sample_next_word_2gram(arr[-1], one_grams, context_map_2)
                
                if next_word:
                    arr.append(next_word)
                else:
                    arr.append(sample_random_word(one_grams))
        else:
            arr.append(sample_random_word(one_grams))
    
    return " ".join(arr)


if __name__ == '__main__':
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"Error: Model file '{args.model}' not found")
        print("Please train a model first using train.py")
        exit(1)
    
    print(f"Loading model from '{args.model}'...")
    try:
        with open(model_path, 'r', encoding='utf-8') as file:
            all_grams = json.load(file)
    except json.JSONDecodeError:
        print(f"Error: '{args.model}' is not a valid JSON file")
        exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    if not all_grams:
        print("Error: Model is empty")
        exit(1)
    
    print(f"Model loaded: {len(all_grams)} n-grams")
    
    if args.length <= 0:
        print("Error: Length must be positive")
        exit(1)
    
    length = args.length
    prefix = args.prefix
    
    if prefix:
        print(f"Generating {length} words with prefix: {' '.join(prefix)}")
    else:
        print(f"Generating {length} words with random start")
    
    if args.use_interpolation:
        lambda_sum = args.lambda1 + args.lambda2 + args.lambda3
        if abs(lambda_sum - 1.0) > 0.01:
            print(f"Warning: Lambda values sum to {lambda_sum:.3f}, normalizing to 1.0")
            lambdas = (args.lambda3 / lambda_sum, args.lambda2 / lambda_sum, args.lambda1 / lambda_sum)
        else:
            lambdas = (args.lambda3, args.lambda2, args.lambda1)
        
        if args.temperature <= 0:
            print("Error: Temperature must be positive")
            exit(1)
        
        print(f"Using interpolation (slower): λ3={lambdas[0]:.2f}, λ2={lambdas[1]:.2f}, λ1={lambdas[2]:.2f}")
        print(f"Temperature: {args.temperature:.2f}, Smoothing: {args.smoothing:.3f}")
    else:
        lambdas = (0.6, 0.3, 0.1)
        print("Using fast backoff strategy (default)")
    
    generated_text = make_text(all_grams, length, prefix, lambdas, 
                              args.temperature, args.smoothing, args.use_interpolation)
    
    print("\n" + "="*50)
    print("GENERATED TEXT:")
    print("="*50)
    print(generated_text)
    print("="*50)
    
    if args.output:
        try:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            print(f"\nSaved to: {args.output}")
        except Exception as e:
            print(f"\nError saving to file: {e}")
