import os
import base64
import tempfile
import logging
import json
from io import BytesIO
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from analyser import CardAnalyser, CenteringResult
import numpy as np
import cv2
import requests
import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary.utils import cloudinary_url
from flask import redirect, Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}
app.config['UPLOAD_TIMEOUT'] = 30  # seconds

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME', ''),
    api_key=os.environ.get('CLOUDINARY_API_KEY', ''),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET', '')
)

# Pokédex mapping
POKEDEX_MAP = {
    'bulbasaur': 1, 'ivysaur': 2, 'venusaur': 3, 'charmander': 4,
    'charizard': 6, 'squirtle': 7, 'blastoise': 9, 'pikachu': 25,
    'raichu': 26, 'eevee': 133, 'mewtwo': 150, 'mew': 151,
}

def get_pokedex_number(card_name):
    clean = card_name.lower().split('(')[0].strip()
    for name, num in POKEDEX_MAP.items():
        if name in clean:
            return num
    return None
# Initialize analyser
analyser = CardAnalyser()

# Centering grade thresholds (matching analyser.py)
CENTERING_THRESHOLDS = [
    (2, 'Gem', 25.0),
    (4, 'Excellent', 22.0),
    (7, 'Good', 19.0),
    (11, 'Moderate', 15.0),
    (20, 'Poor', 10.0),
    (float('inf'), 'Poor', 6.0),
]

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def pil_image_to_b64(img, quality=88, max_size=(1920, 1920)):
    """Convert PIL Image to base64 string with optional resizing."""
    buf = BytesIO()
    
    # Resize large images to save bandwidth
    if max_size and (img.width > max_size[0] or img.height > max_size[1]):
        img.thumbnail(max_size, Image.LANCZOS)
    
    # Convert RGBA to RGB if necessary
    if img.mode in ('RGBA', 'LA', 'P'):
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = rgb_img
    
    img.save(buf, format='JPEG', quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def pil_image_to_b64_gray(img):
    """Convert PIL Image to grayscale base64 (for v4 compatibility)."""
    buf = BytesIO()
    img.convert('L').convert('RGB').save(buf, format='JPEG', quality=88)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def validate_image(file):
    """Validate the uploaded image file."""
    if not file or file.filename == '':
        return False, 'No file selected'
    
    if not allowed_file(file.filename):
        return False, f'Invalid file type. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
    
    # Check file size
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset position
    
    if size == 0:
        return False, 'Empty file'
    
    if size > app.config['MAX_CONTENT_LENGTH']:
        return False, f'File too large. Maximum size: {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'
    
    # Try to open as image
    try:
        img = Image.open(file)
        img.verify()  # Verify it's actually an image
        file.seek(0)  # Reset position after verify
        return True, 'OK'
    except UnidentifiedImageError:
        file.seek(0)
        return False, 'Invalid image file'
    except Exception as e:
        file.seek(0)
        return False, f'Error reading image: {str(e)}'

def process_manual_crop(image_data, crop_points, border_lines=None):
    """
    Process manual crop points and optionally border lines.
    
    Args:
        image_data: PIL Image object
        crop_points: List of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        border_lines: Optional dict with 'vertical' and 'horizontal' lists of 4 values each
    
    Returns:
        Tuple of (cropped_image, borders_dict or None)
    """
    # Convert crop points to integers
    points = [(int(p['x']), int(p['y'])) for p in crop_points]
    
    # Calculate bounding box from the 4 points
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)
    
    # Ensure coordinates are within image bounds
    left = max(0, left)
    top = max(0, top)
    right = min(image_data.width, right)
    bottom = min(image_data.height, bottom)
    
    # Crop the image and ensure RGB mode
    cropped = image_data.crop((left, top, right, bottom))
    if cropped.mode in ('RGBA', 'LA', 'P'):
        cropped = cropped.convert('RGB')
    
    # Process border lines if provided
    borders = None
    if border_lines and 'vertical' in border_lines and 'horizontal' in border_lines:
        vert = sorted(border_lines['vertical'])
        horiz = sorted(border_lines['horizontal'])
        
        if len(vert) == 4 and len(horiz) == 4:
            # Adjust border lines relative to the crop
            vert_adjusted = [v - left for v in vert]
            horiz_adjusted = [h - top for h in horiz]
            
            # Calculate border widths using the 8-line formula
            left_border_width = max(0, vert_adjusted[1] - vert_adjusted[0])
            right_border_width = max(0, vert_adjusted[3] - vert_adjusted[2])
            top_border_width = max(0, horiz_adjusted[1] - horiz_adjusted[0])
            bottom_border_width = max(0, horiz_adjusted[3] - horiz_adjusted[2])
            
            borders = {
                'left_border': left_border_width,
                'right_border': right_border_width,
                'top_border': top_border_width,
                'bottom_border': bottom_border_width,
                'vertical_lines': vert_adjusted,
                'horizontal_lines': horiz_adjusted
            }
    
    return cropped, borders

def calculate_manual_centering(border_widths):
    """
    Calculate centering ratios from manual border measurements.
    Uses the 8-line dynamic formula.
    """
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

# ── ROUTES ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Main homepage."""
    return render_template('index.html')

@app.route('/inventory')
@app.route('/inventory.html')
def inventory():
    """Inventory page."""
    return render_template('inventory.html')

@app.route('/grader')
@app.route('/grader.html')
def grader():
    """Card grader page."""
    return render_template('grader.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Pokemon Card Analysis Engine',
        'version': '5.1',
        'modes': ['auto', 'manual'],
        'features': ['8-line manual centering', 'live ratio calculation', 'consensus border detection']
    })

@app.route('/card-image/<path:card_name>')
def get_card_image(card_name):
    public_id = f"pokemon-cards/{card_name.replace(' ', '_').lower()[:100]}"
    
    try:
        cloudinary.api.resource(public_id)
        url = cloudinary_url(public_id, width=300, crop="fit", format="jpg")[0]
        return redirect(url)
    except:
        pass
    
    pokedex_id = get_pokedex_number(card_name)
    
    try:
        if pokedex_id:
            api_url = f'https://api.pokemontcg.io/v2/cards?q=nationalPokedexNumbers:{pokedex_id}&pageSize=1'
        else:
            clean = card_name.lower().split('(')[0].strip()
            api_url = f'https://api.pokemontcg.io/v2/cards?q=name:"{clean}"&pageSize=1'
        
        resp = requests.get(api_url, timeout=5)
        if resp.status_code == 200 and resp.json().get('data'):
            img_url = resp.json()['data'][0]['images']['small']
            cloudinary.uploader.upload(img_url, public_id=public_id, overwrite=True)
            url = cloudinary_url(public_id, width=300, crop="fit", format="jpg")[0]
            return redirect(url)
    except:
        pass
    
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="280"><rect width="200" height="280" fill="#FCE4EC" rx="12"/><text x="100" y="150" text-anchor="middle" fill="#B5547A" font-size="48">🃏</text></svg>'
    return Response(svg, mimetype='image/svg+xml')

@app.route('/grade', methods=['POST'])
def grade():
    """
    Grade endpoint with support for both auto and manual modes.
    
    This single endpoint handles:
    - v4 requests (auto mode only, returns simpler payload with gray image)
    - v5.1 requests (auto + manual mode, returns full payload with defect images)
    
    The response format is detected based on the mode parameter:
    - mode=auto (or no mode) → v5.1 full response
    - mode=manual → v5.1 manual response
    - For v4 compatibility, the v4 frontend ignores extra fields
    """
    try:
        # Determine mode
        mode = request.form.get('mode', 'auto')
        
        # Check if image is present
        if 'image' not in request.files:
            logger.warning('No image in request')
            return jsonify({
                'success': False, 
                'error': 'No image provided. Please upload an image file.'
            }), 400
        
        file = request.files['image']
        
        # Validate the uploaded file
        is_valid, message = validate_image(file)
        if not is_valid:
            logger.warning(f'Invalid upload: {message}')
            return jsonify({
                'success': False, 
                'error': message
            }), 400
        
        # Create temporary file
        suffix = Path(secure_filename(file.filename)).suffix or '.jpg'
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp.name
        tmp.close()
        
        annotated_path = None
        cropped_path = None
        
        try:
            # Save uploaded file
            file.save(tmp_path)
            file_size = os.path.getsize(tmp_path)
            logger.info(f'Processing file: {file.filename} ({file_size} bytes) in {mode} mode')
            
            # Process based on mode
            if mode == 'manual':
                # Get crop points from request
                crop_points_json = request.form.get('crop_points')
                if not crop_points_json:
                    return jsonify({
                        'success': False,
                        'error': 'Manual mode requires crop_points parameter'
                    }), 400
                
                try:
                    crop_points = json.loads(crop_points_json)
                    if len(crop_points) != 4:
                        return jsonify({
                            'success': False,
                            'error': 'crop_points must contain exactly 4 points'
                        }), 400
                except json.JSONDecodeError:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid crop_points JSON format'
                    }), 400
                
                # Get border lines for 8-line centering
                border_lines_json = request.form.get('border_lines')
                border_lines = None
                if border_lines_json:
                    try:
                        border_lines = json.loads(border_lines_json)
                        if 'vertical' not in border_lines or 'horizontal' not in border_lines:
                            logger.warning('border_lines missing vertical or horizontal arrays')
                            border_lines = None
                        elif len(border_lines['vertical']) != 4 or len(border_lines['horizontal']) != 4:
                            logger.warning(f'border_lines must have exactly 4 values each')
                            border_lines = None
                    except json.JSONDecodeError:
                        logger.warning('Invalid border_lines JSON')
                        border_lines = None
                
                # Open original image
                original_img = Image.open(tmp_path)
                
                # Process manual crop
                cropped_img, manual_borders = process_manual_crop(
                    original_img, crop_points, border_lines
                )
                
                # Save cropped image temporarily
                cropped_fd, cropped_path = tempfile.mkstemp(suffix='.jpg')
                os.close(cropped_fd)
                if cropped_img.mode in ('RGBA', 'LA', 'P'):
                    cropped_img = cropped_img.convert('RGB')
                cropped_img.save(cropped_path, 'JPEG', quality=95)
                
                logger.info(f'Manual crop applied. Original: {original_img.size}, Cropped: {cropped_img.size}')
                
                if manual_borders:
                    logger.info(f'8-line manual borders: L={manual_borders["left_border"]} R={manual_borders["right_border"]} T={manual_borders["top_border"]} B={manual_borders["bottom_border"]}')
                else:
                    logger.info('No border lines provided, will use auto-detection on cropped image')
                
                # Run analysis on cropped image
                result = analyser.analyse(cropped_path)
                
                # Override centering with manual 8-line borders if available
                if manual_borders and result.success:
                    centering_data = calculate_manual_centering(manual_borders)
                    
                    # Create new CenteringResult with manual measurements
                    from analyser import CenteringResult as CR
                    result.centering = CR(
                        left_border=centering_data['left_border'],
                        right_border=centering_data['right_border'],
                        top_border=centering_data['top_border'],
                        bottom_border=centering_data['bottom_border'],
                        lr_ratio=centering_data['lr_ratio'],
                        tb_ratio=centering_data['tb_ratio'],
                        grade=centering_data['grade'],
                        score=centering_data['score'],
                        method='manual_8line'
                    )
                    
                    # Recalculate overall score with manual centering
                    from analyser import Grader
                    grader = Grader()
                    result.overall_score = grader.calculate_overall_score(
                        result.centering, result.edges, result.corners, result.surface
                    )
                    
                    # Recalculate grade band and recommendation
                    grade_str, band, rec, rec_reason, beckett = grader.recommend(
                        result.overall_score, result.centering, result.edges, 
                        result.corners, result.surface
                    )
                    result.estimated_grade = grade_str
                    result.grade_band = band
                    result.recommendation = rec
                    result.recommendation_reason = rec_reason
                    result.beckett_candidate = beckett
                    
                    logger.info(f'Manual 8-line centering applied. Grade: {centering_data["grade"]} ({centering_data["score"]}/25)')
                
            else:
                # Auto mode - standard analysis
                result = analyser.analyse(tmp_path)
            
            if not result.success:
                logger.error(f'Analysis failed: {result.error_msg}')
                return jsonify({
                    'success': False, 
                    'error': f'Analysis failed: {result.error_msg}'
                }), 422
            
            annotated_path = result.annotated_path
            
            # Load and encode annotated image
            try:
                annotated_img = Image.open(annotated_path)
                annotated_b64 = pil_image_to_b64(annotated_img)
                annotated_b64_gray = pil_image_to_b64_gray(annotated_img)
                annotated_img.close()
            except Exception as e:
                logger.error(f'Failed to process annotated image: {e}')
                return jsonify({
                    'success': False,
                    'error': 'Failed to generate annotated image'
                }), 500
            
            # Build response payload (v5.1 format - v4 frontend ignores extra fields)
            s = result.surface
            c = result.centering
            e = result.edges
            co = result.corners
            
            # Calculate surface quality score
            surface_quality_score = round(s.score - s.print_lines_score - s.missing_texture_score, 2)
            
            # Calculate percentage scores for display
            centering_pct = round((c.score / 25.0) * 100, 1)
            edges_pct = round((e.score / 15.0) * 100, 1)
            corners_pct = round((co.score / 20.0) * 100, 1)
            surface_pct = round((s.score / 30.0) * 100, 1)
            surface_quality_pct = round((surface_quality_score / 20.0) * 100, 1)
            print_lines_pct = round((s.print_lines_score / 5.0) * 100, 1)
            missing_texture_pct = round((s.missing_texture_score / 5.0) * 100, 1)
            
            # v5.1 full payload (backward compatible with v4)
            payload = {
                'success': True,
                'mode': mode,
                'centering_method': c.method,
                'annotated_image': annotated_b64,
                'annotated_image_gray': annotated_b64_gray,  # v4 compatibility
                'defect_images': [dict(d) for d in result.defect_images] if result.defect_images else [],
                'beckett_candidate': bool(result.beckett_candidate),  # v4 compatibility (top-level)
                'overall_score': result.overall_score,  # v4 compatibility
                'estimated_grade': result.estimated_grade,  # v4 compatibility
                'grade_band': result.grade_band,  # v4 compatibility
                'recommendation': result.recommendation,  # v4 compatibility
                'recommendation_reason': result.recommendation_reason,  # v4 compatibility
                'overall_analysis': {
                    'beckett_candidate': bool(result.beckett_candidate),
                    'overall_score': result.overall_score,
                    'estimated_grade': result.estimated_grade,
                    'grade_band': result.grade_band,
                    'recommendation': result.recommendation,
                    'recommendation_reason': result.recommendation_reason,
                    'scoring_weights': {
                        'surface': 30,
                        'centering': 25,
                        'corners': 20,
                        'edges': 15,
                        'print_lines': 5,
                        'missing_texture': 5
                    }
                },
                'centering': {
                    'lr_ratio': c.lr_ratio,
                    'tb_ratio': c.tb_ratio,
                    'left_border': c.left_border,
                    'right_border': c.right_border,
                    'top_border': c.top_border,
                    'bottom_border': c.bottom_border,
                    'measurements': {
                        'left': c.left_border,
                        'right': c.right_border,
                        'top': c.top_border,
                        'bottom': c.bottom_border,
                    },
                    'grade': c.grade,
                    'score': c.score,
                    'max_score': 25,
                    'percentage': centering_pct,
                    'weight': 25,
                    'method': c.method,
                },
                'edges': {
                    'top': {
                        'flag': e.top_flag,
                        'deviation': round(e.top_deviation, 2),
                        'dev': round(e.top_deviation, 2)  # v4 compatibility
                    },
                    'bottom': {
                        'flag': e.bottom_flag,
                        'deviation': round(e.bottom_deviation, 2),
                        'dev': round(e.bottom_deviation, 2)
                    },
                    'left': {
                        'flag': e.left_flag,
                        'deviation': round(e.left_deviation, 2),
                        'dev': round(e.left_deviation, 2)
                    },
                    'right': {
                        'flag': e.right_flag,
                        'deviation': round(e.right_deviation, 2),
                        'dev': round(e.right_deviation, 2)
                    },
                    'score': e.score,
                    'max_score': 15,
                    'percentage': edges_pct,
                    'weight': 15,
                },
                'corners': {
                    'tl': {
                        'rating': co.tl,
                        'whitening': round(co.tl_whitening, 1)
                    },
                    'tr': {
                        'rating': co.tr,
                        'whitening': round(co.tr_whitening, 1)
                    },
                    'bl': {
                        'rating': co.bl,
                        'whitening': round(co.bl_whitening, 1)
                    },
                    'br': {
                        'rating': co.br,
                        'whitening': round(co.br_whitening, 1)
                    },
                    'score': co.score,
                    'max_score': 20,
                    'percentage': corners_pct,
                    'weight': 20,
                },
                'surface': {
                    'grade': s.grade,
                    'score': s.score,
                    'max_score': 30,
                    'percentage': surface_pct,
                    'weight': 30,
                    'scratches_detected': bool(s.scratches_detected),  # v4 compatibility
                    'print_defects_detected': bool(s.print_defects_detected),  # v4 compatibility
                    'haze_detected': bool(s.haze_detected),  # v4 compatibility
                    'scratch_severity': s.scratch_severity,  # v4 compatibility
                    'defect_severity': s.defect_severity,  # v4 compatibility
                    'haze_level': s.haze_level,  # v4 compatibility
                    'deductions': [
                        {
                            'reason': d[0],
                            'points_lost': d[1]
                        } for d in s.deductions
                    ],  # matches both v4 and v5 format
                    'breakdown': {
                        'surface_quality': {
                            'score': surface_quality_score,
                            'max_score': 20,
                            'percentage': surface_quality_pct,
                            'weight': 20,
                        },
                        'print_lines': {
                            'score': s.print_lines_score,
                            'max_score': 5,
                            'percentage': print_lines_pct,
                            'weight': 5,
                        },
                        'missing_texture': {
                            'score': s.missing_texture_score,
                            'max_score': 5,
                            'percentage': missing_texture_pct,
                            'weight': 5,
                        }
                    },
                    'issues': {
                        'scratches': bool(s.scratches_detected),
                        'print_defects': bool(s.print_defects_detected),
                        'haze': bool(s.haze_detected),
                        'texture_anomalies': bool(s.lbp_defects_detected),
                    },
                    'severity': {
                        'scratch': s.scratch_severity,
                        'defect': s.defect_severity,
                        'haze': s.haze_level,
                        'lbp': s.lbp_severity,
                    },
                },
            }
            
            logger.info(f'Analysis complete for {file.filename}: {result.estimated_grade} ({result.overall_score}/100) [{mode} mode]')
            return jsonify(payload)
            
        except Exception as e:
            logger.exception(f'Unexpected error during analysis: {e}')
            return jsonify({
                'success': False,
                'error': f'Internal server error: {str(e)}'
            }), 500
            
        finally:
            # Clean up temporary files
            for path in [tmp_path, annotated_path, cropped_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                        logger.debug(f'Cleaned up: {path}')
                    except Exception as e:
                        logger.warning(f'Failed to clean up {path}: {e}')
    
    except Exception as e:
        logger.exception(f'Unhandled exception: {e}')
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred'
        }), 500

# ── ERROR HANDLERS ──────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File is too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(400)
def bad_request(e):
    return jsonify({
        'success': False,
        'error': 'Bad request. Please check your input.'
    }), 400

@app.errorhandler(500)
def server_error(e):
    logger.error(f'Internal server error: {e}')
    return jsonify({
        'success': False,
        'error': 'Internal server error. Please try again later.'
    }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Resource not found.'
    }), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f'Starting Fair Game TCG on port {port}')
    logger.info(f'Debug mode: {debug}')
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True,
    )
