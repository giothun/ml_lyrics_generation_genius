# ML Lyrics Generation (Genius) 

A lyrics generation project using n-gram language models trained on artist lyrics from [Genius.com](https://genius.com). Generate new song lyrics in the style of your favorite Russian hip-hop and rap artists.

Web access: https://mllyricsgenerationgenius-production.up.railway.app/

## Features

- 🎁 **Pre-trained model included** - Ready to use with 4,700+ Russian lyrics (1.5M n-grams)
- 📥 Download lyrics from Genius.com API for multiple artists
- 🧠 Train n-gram language models (configurable 1-3 grams)
- 🎵 Generate new lyrics with optional seed phrases
- 🔄 Intelligent backoff strategy (3-gram → 2-gram → 1-gram)
- 🚀 Optimized generation using efficient lookup structures
- 💾 Save generated lyrics to files
- 🌐 **Web interface** for easy lyrics generation in your browser
- 📤 Upload custom trained models through the web UI

## Pre-trained Model Included 🎁

This repository comes with a **pre-trained model** (`all_grams.txt`) ready to use:

- **Dataset**: 4,730 Russian song lyrics
- **Artists**: 40+ popular Russian hip-hop and rap artists
- **Vocabulary**: 104,522 unique words
- **N-grams**: 1,490,644 total (1-grams, 2-grams, 3-grams)
- **File Size**: ~46 MB

**You can start generating lyrics immediately without training!**

Advanced users can train custom models on their own data (see Training section below).

## Prerequisites

- Python 3.7+
- Genius API key (only needed if you want to download new lyrics)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ml_lyrics_generation_genius.git
cd ml_lyrics_generation_genius
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Genius API key:
```bash
export GENIUS_API_KEY='your_api_key_here'
```

Or add it to your `.bashrc` or `.zshrc` for persistence:
```bash
echo "export GENIUS_API_KEY='your_api_key_here'" >> ~/.zshrc
source ~/.zshrc
```

## Usage

There are two ways to use this project:
1. **Web Interface** (recommended for ease of use)
2. **Command Line** (for advanced users and automation)

---

## 🌐 Web Interface

The easiest way to generate lyrics is through the web interface.

### Starting the Web App

```bash
# Start the web server (pre-trained model included!)
python web_app.py
```

Then open your browser and navigate to: **http://localhost:5000**

### Web Features

- ✨ **Easy Generation**: Simple form to generate lyrics with customizable length and prefix
- 📊 **Model Statistics**: See n-gram counts and vocabulary size
- 📤 **Upload Models**: Upload custom trained models
- 💾 **Download Results**: Save generated lyrics as text files
- 📋 **Copy to Clipboard**: One-click copy functionality
- 🎨 **Modern UI**: Beautiful, responsive design that works on all devices

### Web Interface Screenshots

The web interface provides:
1. Model status dashboard showing n-gram statistics
2. Model upload/management section
3. Generation form with prefix and length controls
4. Results display with copy and download options

---

## 💻 Command Line Usage

For automation or advanced use cases, use the command-line scripts directly.

### 1. Download Lyrics (`create_dataset_genius.py`)

Download lyrics for artists specified in the script:

```bash
python create_dataset_genius.py
```

This will:
- Connect to the Genius API
- Download lyrics for all artists in the predefined list
- Save individual songs as text files in the `data/` directory
- Show progress and statistics

**Note:** The script includes a predefined list of Russian artists. Edit the `choose_artist()` function to add or remove artists.

### 2. Train a Custom Model (`train.py`) - Optional

**Note:** A pre-trained model is already included! This step is only for advanced users who want to:
- Train on different artists or languages
- Use their own lyrics dataset
- Experiment with different n-gram sizes

```bash
# Download new lyrics first (requires GENIUS_API_KEY)
export GENIUS_API_KEY='your_key_here'
python create_dataset_genius.py

# Train on your custom data
python train.py --input-dir data/data --model my_custom_model.txt --ngram-size 3
```

**Arguments:**
- `-indir`, `--input-dir`: Directory containing training data (default: `data`)
- `-m`, `--model`: Path to save the trained model (default: `all_grams.txt`)
- `-n`, `--ngram-size`: Maximum n-gram size, 1-3 (default: `3`)

**Example output:**
```
Found 4823 text files
Successfully loaded 4730 files
Training 3-gram model...
Generated 1490644 unique n-grams
  1-grams: 104522
  2-grams: 560326
  3-grams: 825796
Saving model to 'all_grams.txt'...
Model saved successfully!
```

**Using Custom Models:**
- Command line: `python generate.py -m my_custom_model.txt -p "prefix" -l 50`
- Web interface: Upload via the "Upload Custom Model" button

### 3. Generate Lyrics (`generate.py`)

Generate new lyrics using the trained model:

```bash
# Random start, 20 words
python generate.py

# With a seed phrase
python generate.py -p я люблю -l 30

# Save to file
python generate.py -p я хочу -l 50 -o output.txt
```

**Arguments:**
- `-m`, `--model`: Path to trained model file (default: `all_grams.txt`)
- `-p`, `--prefix`: Starting words for generation (optional)
- `-l`, `--length`: Number of words to generate (default: `20`)
- `-o`, `--output`: Output file path (optional)

**Example output:**
```
Loading model from 'all_grams.txt'...
Model loaded: 45231 n-grams
Generating 30 words with prefix: я люблю

==================================================
GENERATED TEXT:
==================================================
я люблю когда звучит музыка и ты танцуешь со мной под луной мы летим высоко над облаками забывая про всё что было раньше
==================================================
```

## Project Structure

```
ml_lyrics_generation_genius/
├── create_dataset_genius.py  # Download lyrics from Genius
├── train.py                  # Train n-gram model
├── generate.py               # Generate new lyrics (CLI)
├── web_app.py                # Flask web application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore                # Git ignore rules
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose configuration
├── railway.json              # Railway deployment config
├── DEPLOYMENT.md             # Deployment guide
├── templates/                # HTML templates for web app
│   └── index.html           # Main web interface
├── static/                   # Static files (CSS, JS)
│   └── style.css            # Web app styling
├── uploads/                  # User-uploaded models (created automatically)
├── data/                     # Downloaded lyrics (created automatically)
└── all_grams.txt            # Trained model (pre-trained, ready to use!)
```

## How It Works

### N-gram Language Model

The project uses a statistical n-gram language model:

1. **Training**: Count frequencies of word sequences (1-grams, 2-grams, 3-grams) in the corpus
2. **Generation**: 
   - Start with a seed phrase (or random words)
   - For each new word, use the previous 2 words as context
   - Calculate probability: `P(w₃|w₁,w₂) = count(w₁,w₂,w₃) / count(w₁,w₂)`
   - Sample next word from the probability distribution
   - If no 3-gram found, back off to 2-gram, then 1-gram

### Backoff Strategy

```
Try 3-gram model (w₁, w₂) → w₃
    ↓ (if fails)
Try 2-gram model w₂ → w₃
    ↓ (if fails)
Use 1-gram model (random word from corpus)
```

This ensures coherent text generation even when encountering unseen contexts.



## Examples

### Example 1: Generate with prefix
```bash
python generate.py -p я стоял на краю -l 25
```
Output: *"я стоял на краю пропасти и смотрел вниз где бушевал океан мыслей..."*

### Example 2: Longer generation
```bash
python generate.py -l 100 -o my_song.txt
```
Generates 100 words and saves to `my_song.txt`

### Example 3: Train on custom data
```bash
mkdir my_lyrics
# Add your .txt files to my_lyrics/
python train.py -indir my_lyrics -m my_model.txt
python generate.py -m my_model.txt -l 50
```

## Included Artists

The default dataset includes lyrics from popular Russian artists:
- Слава КПСС, Pyrokinesis, Aikko, Lil krystalll, oxxxymiron
- ATL, Og buda, Loqiemean, katanacss, playingtheangel
- монеточка, три дня дождя, MORGENSHTERN, NOIZE MC
- And many more...

See `create_dataset_genius.py` for the full list.

## Limitations

- **Quality**: N-gram models don't understand grammar or meaning, just statistical patterns
- **Coherence**: Generated text may lose coherence over longer sequences
- **Rhyme**: No rhyme scheme enforcement (future improvement)
- **Context**: No long-range dependencies (only looks back 2 words)

## Future Improvements

- [x] ~~Web interface (Flask)~~ ✅ **Implemented!**
- [ ] Add temperature parameter for controlling randomness
- [ ] Implement Kneser-Ney smoothing for better probability estimates
- [ ] Add rhyme scheme detection and enforcement
- [ ] Support for 4-grams and 5-grams
- [ ] Use pickle for faster model loading
- [ ] Perplexity calculation for model evaluation
- [ ] Verse structure preservation
- [ ] API endpoint for programmatic access
- [ ] Model versioning and comparison

## Quick Start Guide

### Option 1: Web Interface (Recommended)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the web app (pre-trained model included!)
python web_app.py

# 3. Open http://localhost:5000 in your browser
```

**That's it!** The repository includes a pre-trained model (`all_grams.txt`) trained on 4,700+ Russian song lyrics, so you can start generating immediately.

### Option 2: Command Line
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate lyrics with pre-trained model
python generate.py -p "я хочу" -l 30

# Optional: Train your own custom model
# python train.py -indir your_data_folder
```

## Troubleshooting

### Web App Issues

**"No model loaded"**
- Train a model first: `python train.py`
- Or upload a pre-trained model through the web interface

**Web app won't start**
- Make sure Flask is installed: `pip install Flask`
- Check if port 5000 is already in use
- Try a different port: Edit `web_app.py` and change `port=5000`

### Command Line Issues

**"Error: GENIUS_API_KEY environment variable not set"**
Set your API key: `export GENIUS_API_KEY='your_key'`

**"Error: Directory 'data' does not exist"**
Run `create_dataset_genius.py` first to download lyrics

**"Warning: 'word' not in vocabulary"**
The prefix word wasn't seen during training. The generator will skip it and use available words.

**Model file is very large**
This is normal. A model trained on 150+ songs can be 10-50 MB. Consider using pickle instead of JSON for better compression.

## License

This project is for educational purposes. Respect Genius.com's API terms of service and artist copyrights.

## Contributing

Feel free to open issues or submit pull requests with improvements!

## Acknowledgments

- [LyricsGenius](https://github.com/johnwmillr/LyricsGenius) - Python library for Genius API
- [Genius.com](https://genius.com) - Lyrics database
- Russian hip-hop and rap artists for their creative work
