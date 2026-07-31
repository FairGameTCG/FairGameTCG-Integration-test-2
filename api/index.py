import os
import sys

# Add the root directory to the path so we can import analyser
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import base64
import tempfile
import logging
import json
from pathlib import Path

# Import your analyser
from analyser import CardAnalyser, CenteringResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize analyser
analyser = CardAnalyser()

# ── Helper functions ─────────────────────────────────────────────────────

def pil_image_to_b64(img, quality=88, max_size=(1920, 1920)):
    """Convert PIL Image to base64 string."""
    buf = BytesIO()
    if max_size and (img.width > max_size[0] or img.height > max_size[1]):
        img.thumbnail(max_size, Image.LANCZOS)
    if img.mode in ('RGBA', 'LA', 'P'):
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = rgb_img
    img.save(buf, format='JPEG', quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def pil_image_to_b64_gray(img):
    """Convert PIL Image to grayscale base64."""
    buf = BytesIO()
    img.convert('L').convert('RGB').save(buf, format='JPEG', quality=88)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

# ── Centering thresholds ─────────────────────────────────────────────────
CENTERING_THRESHOLDS = [
    (2, 'Gem', 25.0),
    (4, 'Excellent', 22.0),
    (7, 'Good', 19.0),
    (11, 'Moderate', 15.0),
    (20, 'Poor', 10.0),
    (float('inf'), 'Poor', 6.0),
]

def calculate_manual_centering(border_widths):
    """Calculate centering from manual border measurements."""
    left_width = max(border_widths['left_border'], 1)
    right_width = max(border_widths['right_border'], 1)
    top_width = max(border_widths['top_border'], 1)
    bottom_width = max(border_widths['bottom_border'], 1)
    
    total_h = left_width + right_width
    total_v = top_width + bottom_width
    
    lr_left = round((left_width / total_h) * 100, 1)
    lr_right = round((right_width / total_h) * 100, 1)
    tb_top = round((top_width / total_v) * 100, 1)
    tb_bottom = round((bottom_width / total_v) * 100, 1)
    
    lr_dev = abs(lr_left - 50.0)
    tb_dev = abs(tb_top - 50.0)
    worst = max(lr_dev, tb_dev)
    
    grade, score = 'Poor', 6.0
    for threshold, g, s in CENTERING_THRESHOLDS:
        if worst <= threshold:
            grade, score = g, s
            break
    
    return {
        'lr_ratio': (lr_left, lr_right),
        'tb_ratio': (tb_top, tb_bottom),
        'grade': grade,
        'score': score,
        'left_border': left_width,
        'right_border': right_width,
        'top_border': top_width,
        'bottom_border': bottom_width,
    }

def process_manual_crop(image_data, crop_points):
    """Crop image based on 4 corner points."""
    points = [(int(p['x']), int(p['y'])) for p in crop_points]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    left = max(0, min(xs))
    right = min(image_data.width, max(xs))
    top = max(0, min(ys))
    bottom = min(image_data.height, max(ys))
    
    cropped = image_data.crop((left, top, right, bottom))
    if cropped.mode in ('RGBA', 'LA', 'P'):
        cropped = cropped.convert('RGB')
    
    return cropped, (left, top)

# ── Routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inventory')
@app.route('/inventory.html')
def inventory():
    return render_template('inventory.html')

@app.route('/grader')
@app.route('/grader.html')
def grader():
    return render_template('grader.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': '5.1'})

@app.route('/grade', methods=['POST'])
def grade():
    try:
        mode = request.form.get('mode', 'auto')
        
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Read image directly from memory (Vercel has read-only filesystem)
        image_bytes = file.read()
        
        # Save to a temp location Vercel allows
        tmp_path = '/tmp/upload_' + secure_filename(file.filename)
        with open(tmp_path, 'wb') as f:
            f.write(image_bytes)
        
        annotated_path = None
        
        try:
            if mode == 'manual':
                # Get crop points
                crop_points_json = request.form.get('crop_points')
                if not crop_points_json:
                    return jsonify({'success': False, 'error': 'crop_points required for manual mode'}), 400
                
                crop_points = json.loads(crop_points_json)
                
                # Open image and apply manual crop
                original_img = Image.open(tmp_path)
                cropped_img, crop_offset = process_manual_crop(original_img, crop_points)
                
                # Save cropped image
                cropped_path = '/tmp/cropped_' + secure_filename(file.filename)
                if cropped_img.mode in ('RGBA', 'LA', 'P'):
                    cropped_img = cropped_img.convert('RGB')
                cropped_img.save(cropped_path, 'JPEG', quality=95)
                
                # Run analysis on cropped image
                result = analyser.analyse(cropped_path)
                
                # Clean up cropped file
                if os.path.exists(cropped_path):
                    os.unlink(cropped_path)
            else:
                # Auto mode
                result = analyser.analyse(tmp_path)
            
            if not result.success:
                return jsonify({'success': False, 'error': result.error_msg}), 422
            
            # Load annotated image
            annotated_img = Image.open(result.annotated_path)
            annotated_b64 = pil_image_to_b64(annotated_img)
            annotated_b64_gray = pil_image_to_b64_gray(annotated_img)
            annotated_img.close()
            
            # Build response
            s = result.surface
            c = result.centering
            e = result.edges
            co = result.corners
            
            payload = {
                'success': True,
                'mode': mode,
                'centering_method': c.method if c else 'auto',
                'annotated_image': annotated_b64,
                'annotated_image_gray': annotated_b64_gray,
                'defect_images': [],
                'beckett_candidate': bool(result.beckett_candidate),
                'overall_score': result.overall_score,
                'estimated_grade': result.estimated_grade,
                'grade_band': result.grade_band,
                'recommendation': result.recommendation,
                'recommendation_reason': result.recommendation_reason,
                'overall_analysis': {
                    'beckett_candidate': bool(result.beckett_candidate),
                    'overall_score': result.overall_score,
                    'estimated_grade': result.estimated_grade,
                    'grade_band': result.grade_band,
                    'recommendation': result.recommendation,
                    'recommendation_reason': result.recommendation_reason,
                },
            }
            
            if c:
                payload['centering'] = {
                    'lr_ratio': c.lr_ratio,
                    'tb_ratio': c.tb_ratio,
                    'measurements': {'left': c.left_border, 'right': c.right_border, 'top': c.top_border, 'bottom': c.bottom_border},
                    'grade': c.grade,
                    'score': c.score,
                    'max_score': 25,
                    'method': c.method,
                }
            
            if e:
                payload['edges'] = {
                    'top': {'flag': e.top_flag, 'deviation': round(e.top_deviation, 2), 'dev': round(e.top_deviation, 2)},
                    'bottom': {'flag': e.bottom_flag, 'deviation': round(e.bottom_deviation, 2), 'dev': round(e.bottom_deviation, 2)},
                    'left': {'flag': e.left_flag, 'deviation': round(e.left_deviation, 2), 'dev': round(e.left_deviation, 2)},
                    'right': {'flag': e.right_flag, 'deviation': round(e.right_deviation, 2), 'dev': round(e.right_deviation, 2)},
                    'score': e.score,
                    'max_score': 15,
                }
            
            if co:
                payload['corners'] = {
                    'tl': {'rating': co.tl, 'whitening': round(co.tl_whitening, 1)},
                    'tr': {'rating': co.tr, 'whitening': round(co.tr_whitening, 1)},
                    'bl': {'rating': co.bl, 'whitening': round(co.bl_whitening, 1)},
                    'br': {'rating': co.br, 'whitening': round(co.br_whitening, 1)},
                    'score': co.score,
                    'max_score': 20,
                }
            
            if s:
                payload['surface'] = {
                    'grade': s.grade,
                    'score': s.score,
                    'max_score': 30,
                    'scratches_detected': bool(s.scratches_detected),
                    'print_defects_detected': bool(s.print_defects_detected),
                    'haze_detected': bool(s.haze_detected),
                    'scratch_severity': s.scratch_severity,
                    'defect_severity': s.defect_severity,
                    'haze_level': s.haze_level,
                    'deductions': [{'reason': d[0], 'points_lost': d[1]} for d in s.deductions],
                }
            
            return jsonify(payload)
            
        finally:
            # Clean up
            for path in [tmp_path, result.annotated_path if hasattr(result, 'annotated_path') and result.annotated_path else None]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception:
                        pass
    
    except Exception as e:
        logger.exception(f'Error: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Max 16MB.'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Internal server error.'}), 500

# ── Vercel handler ───────────────────────────────────────────────────────
# This is what Vercel calls
def handler(request, **kwargs):
    """Handle incoming requests for Vercel serverless function."""
    return app
