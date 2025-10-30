"""
Flask web application for lyrics generation.
Allows users to generate lyrics using a trained n-gram model.
"""

import json
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import secrets

# Import generation functions from generate.py
from generate import separate_grams, make_text

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Default model path
DEFAULT_MODEL = 'all_grams.txt'


def load_model(model_path):
    """Load n-gram model from file.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dictionary of n-grams or None if failed
    """
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            all_grams = json.load(f)
        return all_grams
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def get_model_stats(all_grams):
    """Get statistics about the loaded model.
    
    Args:
        all_grams: Dictionary of n-grams
        
    Returns:
        Dictionary with model statistics
    """
    if not all_grams:
        return None
    
    gram_counts = {}
    for gram in all_grams.keys():
        size = len(gram.split(' '))
        gram_counts[size] = gram_counts.get(size, 0) + 1
    
    return {
        'total': len(all_grams),
        'unigrams': gram_counts.get(1, 0),
        'bigrams': gram_counts.get(2, 0),
        'trigrams': gram_counts.get(3, 0),
    }


@app.route('/')
def index():
    """Render the main page."""
    # Check if default model exists
    default_model_exists = Path(DEFAULT_MODEL).exists()
    
    # Get current model stats if available
    current_model = session.get('current_model', DEFAULT_MODEL if default_model_exists else None)
    model_stats = None
    
    if current_model and Path(current_model).exists():
        all_grams = load_model(current_model)
        if all_grams:
            model_stats = get_model_stats(all_grams)
    
    return render_template('index.html', 
                         default_model_exists=default_model_exists,
                         model_stats=model_stats,
                         current_model=current_model)


@app.route('/generate', methods=['POST'])
def generate():
    """Generate lyrics based on form input."""
    try:
        # Get form data
        prefix = request.form.get('prefix', '').strip()
        length = int(request.form.get('length', 20))
        
        # Get advanced parameters (with defaults)
        lambda3 = float(request.form.get('lambda3', 0.6))
        lambda2 = float(request.form.get('lambda2', 0.3))
        lambda1 = float(request.form.get('lambda1', 0.1))
        temperature = float(request.form.get('temperature', 1.0))
        smoothing = float(request.form.get('smoothing', 0.1))
        use_interpolation = request.form.get('use_interpolation', 'false').lower() == 'true'
        
        # Validate length
        if length < 1 or length > 500:
            return jsonify({'error': 'Length must be between 1 and 500'}), 400
        
        # Validate and normalize lambdas
        lambda_sum = lambda1 + lambda2 + lambda3
        if lambda_sum <= 0:
            return jsonify({'error': 'Lambda values must sum to a positive number'}), 400
        lambdas = (lambda3 / lambda_sum, lambda2 / lambda_sum, lambda1 / lambda_sum)
        
        # Validate temperature
        if temperature <= 0:
            return jsonify({'error': 'Temperature must be positive'}), 400
        
        # Determine which model to use
        current_model = session.get('current_model', DEFAULT_MODEL)
        
        # Check if model exists
        if not Path(current_model).exists():
            return jsonify({'error': 'No model loaded. Please upload a model or train one first.'}), 400
        
        # Load model
        all_grams = load_model(current_model)
        if not all_grams:
            return jsonify({'error': 'Failed to load model'}), 500
        
        # Generate text
        prefix_arg = prefix if prefix else None
        generated_text = make_text(all_grams, length, prefix_arg, lambdas, 
                                  temperature, smoothing, use_interpolation)
        
        return jsonify({
            'success': True,
            'text': generated_text,
            'prefix': prefix,
            'length': length,
            'parameters': {
                'lambda3': lambdas[0],
                'lambda2': lambdas[1],
                'lambda1': lambdas[2],
                'temperature': temperature,
                'smoothing': smoothing,
                'use_interpolation': use_interpolation
            }
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500


@app.route('/upload_model', methods=['POST'])
def upload_model():
    """Handle model file upload."""
    try:
        if 'model_file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['model_file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension
        if not file.filename.endswith(('.txt', '.json')):
            return jsonify({'error': 'Only .txt and .json files are allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Validate it's a valid model
        all_grams = load_model(filepath)
        if not all_grams:
            os.remove(filepath)
            return jsonify({'error': 'Invalid model file format'}), 400
        
        # Get stats
        stats = get_model_stats(all_grams)
        
        # Store in session
        session['current_model'] = filepath
        
        return jsonify({
            'success': True,
            'message': 'Model uploaded successfully',
            'stats': stats,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/use_default_model', methods=['POST'])
def use_default_model():
    """Switch back to the default model."""
    try:
        if not Path(DEFAULT_MODEL).exists():
            return jsonify({'error': 'Default model not found. Please train a model first.'}), 400
        
        # Load and validate
        all_grams = load_model(DEFAULT_MODEL)
        if not all_grams:
            return jsonify({'error': 'Failed to load default model'}), 500
        
        stats = get_model_stats(all_grams)
        
        # Update session
        session['current_model'] = DEFAULT_MODEL
        
        return jsonify({
            'success': True,
            'message': 'Switched to default model',
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to switch model: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    import os
    
    print("=" * 60)
    print("ML Lyrics Generation Web Interface")
    print("=" * 60)
    
    if Path(DEFAULT_MODEL).exists():
        print(f"✓ Default model found: {DEFAULT_MODEL}")
    else:
        print(f"⚠ Default model not found: {DEFAULT_MODEL}")
        print("  Please train a model first using: python train.py")
    
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') != 'production'
    
    print(f"\nStarting web server on port {port}...")
    print(f"Mode: {'Development' if debug else 'Production'}")
    if debug:
        print("Open your browser and navigate to: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=debug, host='0.0.0.0', port=port)

