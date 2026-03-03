from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from forensic_analyzer import ForensicAnalyzer
from ui_analyzer import UIAnalyzer
import tempfile

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analyze', methods=['POST'])
def analyze_screenshot():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run forensic analysis
        forensic = ForensicAnalyzer(filepath)
        forensic_results = forensic.run_full_analysis()
        
        # Run UI analysis
        ui = UIAnalyzer(filepath)
        ui_results = ui.analyze()
        
        # Combine results
        final_results = {
            'forensic': forensic_results,
            'ui_analysis': ui_results,
            'combined_score': (forensic_results['overall_score'] + ui_results['overall_ui_score']) / 2,
            'verdict': ''
        }
        
        # Generate verdict
        if final_results['combined_score'] >= 80:
            final_results['verdict'] = '✅ Likely Authentic'
        elif final_results['combined_score'] >= 50:
            final_results['verdict'] = '⚠️ Suspicious - Further Investigation Needed'
        else:
            final_results['verdict'] = '❌ Likely Fake/Manipulated'
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(final_results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)