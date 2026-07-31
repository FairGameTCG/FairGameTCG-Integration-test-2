"""
Pokemon Card Analysis Engine v5.1 — Modular Architecture for Web Deployment
Optimised for top-down phone photos on dark backgrounds.
Supports both auto-detection and client-side manual centering data.
"""

from PIL import Image, ImageOps, ImageDraw
import math
import os
import sys
import argparse
import json
import tempfile
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CARD_W = 1200
CARD_H = 1675
CORNER_PATCH = 130
BORDER_SAMPLE_STEPS = 120
MAX_BORDER_FRACTION = 0.18

# Centering grade thresholds (max_deviation, grade, score)
CENTERING_THRESHOLDS = [
    (2, 'Gem', 25.0),
    (4, 'Excellent', 22.0),
    (7, 'Good', 19.0),
    (11, 'Moderate', 15.0),
    (20, 'Poor', 10.0),
    (float('inf'), 'Poor', 6.0),
]

# Edge grade thresholds (max_deviation, flag, score)
EDGE_THRESHOLDS = [
    (2.0, 'Clean', 15.0),
    (5.0, 'Minor Wear', 12.0),
    (10.0, 'Moderate Wear', 8.25),
    (float('inf'), 'Significant Wear', 4.5),
]

# Corner grade mapping
CORNER_GRADE_MAP = {
    'Sharp': 10.0,
    'Slightly Rounded': 7.5,
    'Rounded': 5.0,
    'Dinged': 2.5,
}

# Surface grade thresholds (min_score, grade)
SURFACE_THRESHOLDS = [
    (27.6, 'Pristine'),
    (22.8, 'Clean'),
    (16.8, 'Light Wear'),
    (10.8, 'Moderate Wear'),
    (float('-inf'), 'Heavy Wear'),
]

# Overall grade bands (min_score, band)
GRADE_BANDS = [
    (95, 'Gem Mint'),
    (90, 'Mint'),
    (80, 'Near Mint-Mint'),
    (70, 'Near Mint'),
    (60, 'Excellent-Mint'),
    (50, 'Excellent'),
    (40, 'Very Good-Excellent'),
    (30, 'Very Good'),
    (20, 'Good'),
    (float('-inf'), 'Poor'),
]

# Scoring weights
SCORING_WEIGHTS = {
    'centering': 0.25,
    'edges': 0.15,
    'corners': 0.20,
    'surface_quality': 0.20,
    'print_lines': 0.10,
    'missing_texture': 0.10,
}

# LBP parameters
LBP_POINTS = 16
LBP_RADIUS = 2
LBP_GRID_COLS = 6
LBP_GRID_ROWS = 8
LBP_CHI_THRESHOLD_FACTOR = 2.0
LBP_MIN_THRESHOLD = 0.08

# Surface analysis parameters
HAZE_STD_THRESHOLD = 40.0
SCRATCH_VARIANCE_THRESHOLD = 55.0
OUTLIER_STD_MULTIPLIER = 3.2
BLOCK_SIZE = 20

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def clamp(val: float, lo: float, hi: float) -> float:
    """Clamp value to range [lo, hi]."""
    return max(lo, min(hi, val))

def mean(vals: List[float]) -> float:
    """Calculate mean of values."""
    return sum(vals) / len(vals) if vals else 0.0

def stdev(vals: List[float]) -> float:
    """Calculate standard deviation."""
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))

# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CenteringResult:
    left_border: int
    right_border: int
    top_border: int
    bottom_border: int
    lr_ratio: Tuple[float, float]
    tb_ratio: Tuple[float, float]
    grade: str
    score: float
    method: str = 'gradient'
    max_score: int = 25

@dataclass
class EdgeResult:
    top_deviation: float
    bottom_deviation: float
    left_deviation: float
    right_deviation: float
    top_flag: str
    bottom_flag: str
    left_flag: str
    right_flag: str
    score: float
    max_score: int = 15

@dataclass
class CornerResult:
    tl: str
    tr: str
    bl: str
    br: str
    tl_whitening: float
    tr_whitening: float
    bl_whitening: float
    br_whitening: float
    score: float
    max_score: int = 20

@dataclass
class SurfaceResult:
    scratches_detected: bool
    print_defects_detected: bool
    haze_detected: bool
    lbp_defects_detected: bool
    scratch_severity: float
    defect_severity: float
    haze_level: float
    lbp_severity: float
    grade: str
    score: float
    deductions: List[Tuple[str, float]] = field(default_factory=list)
    lbp_flagged_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    print_lines_score: float = 0.0
    missing_texture_score: float = 0.0
    max_score: int = 30

@dataclass
class AnalysisResult:
    success: bool
    error_msg: str
    annotated_path: Optional[str] = None
    centering: Optional[CenteringResult] = None
    edges: Optional[EdgeResult] = None
    corners: Optional[CornerResult] = None
    surface: Optional[SurfaceResult] = None
    estimated_grade: str = 'Error'
    grade_band: str = 'Error'
    recommendation: str = 'Unable to analyse'
    recommendation_reason: str = ''
    beckett_candidate: bool = False
    overall_score: float = 0.0
    defect_images: List[Dict] = field(default_factory=list)

# ═══════════════════════════════════════════════════════════════════════════════
# CARD DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class CardDetector:
    """Detects card boundaries in photographs."""
    
    def find_bounds(self, img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """Find card boundaries. Returns (left, top, right, bottom) or None."""
        gray = img.convert('L')
        w, h = gray.size
        pixels = np.array(gray, dtype=np.uint8)
        
        # Sample background from corners and edges
        bg_samples = np.array([
            pixels[5, 5], pixels[5, w-5], pixels[h-5, 5], pixels[h-5, w-5],
            pixels[h//2, 5], pixels[h//2, w-5], pixels[5, h//2], pixels[w-5, h//2]
        ], dtype=float)
        
        bg = float(np.mean(bg_samples))
        dark_bg = bg < 80
        threshold = (bg + 40) if dark_bg else (bg - 40)
        
        def get_pixel(x: int, y: int) -> float:
            return float(pixels[max(0, min(y, h-1)), max(0, min(x, w-1))])
        
        def scan_to_card(axis: str, direction: str, start: int, end: int,
                        mid_range: Tuple[int, int]) -> int:
            step = 1 if direction == 'forward' else -1
            for i in range(start, end, step):
                vals = []
                for j in range(mid_range[0], mid_range[1], 4):
                    if axis == 'x':
                        vals.append(get_pixel(i, j))
                    else:
                        vals.append(get_pixel(j, i))
                m = float(np.mean(vals))
                if dark_bg and m > threshold:
                    return i
                elif not dark_bg and m < threshold:
                    return i
            return start
        
        mid_h = (h // 4, 3 * h // 4)
        mid_w = (w // 4, 3 * w // 4)
        
        left = scan_to_card('x', 'forward', 0, w // 2, mid_h)
        right = scan_to_card('x', 'backward', w - 1, w // 2, mid_h)
        top = scan_to_card('y', 'forward', 0, h // 2, mid_w)
        bottom = scan_to_card('y', 'backward', h - 1, h // 2, mid_w)
        
        card_w = right - left
        card_h = bottom - top
        
        if card_w < w * 0.15 or card_h < h * 0.15:
            logger.debug("Card too small relative to image")
            return None
        
        aspect_ratio = card_h / card_w if card_w > 0 else 0
        if aspect_ratio < 1.2 or aspect_ratio > 1.6:
            logger.debug(f"Invalid aspect ratio: {aspect_ratio:.2f}")
            return None
        
        pad = 3
        return (
            max(0, left - pad),
            max(0, top - pad),
            min(w, right + pad),
            min(h, bottom + pad)
        )

# ═══════════════════════════════════════════════════════════════════════════════
# CONSENSUS BORDER DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ConsensusBorderDetector:
    """Combines multiple border detection strategies using consensus voting."""
    
    def detect(self, img: Image.Image) -> CenteringResult:
        """Detect borders using consensus of multiple methods."""
        w, h = img.size
        gray_arr = np.array(img.convert('L'))
        
        results = []
        
        # Try Sobel edge detection
        try:
            sobel_borders = self._sobel_detect(gray_arr, w, h)
            if sobel_borders:
                results.append(sobel_borders)
        except Exception as e:
            logger.debug(f"Sobel detection failed: {e}")
        
        # Try gradient-based detection
        try:
            gradient_borders = self._gradient_detect(gray_arr, w, h)
            if gradient_borders:
                results.append(gradient_borders)
        except Exception as e:
            logger.debug(f"Gradient detection failed: {e}")
        
        if not results:
            # Fallback: assume typical borders
            default = int(min(w, h) * 0.08)
            borders = (default, default, default, default)
            method = 'fallback'
        else:
            # Average the results
            borders = tuple(int(np.mean([r[i] for r in results])) for i in range(4))
            method = 'consensus'
        
        return self._create_result(borders, w, h, method)
    
    def _sobel_detect(self, gray_arr: np.ndarray, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
        """Sobel edge detection for clean card edges."""
        try:
            sobel_x = cv2.Sobel(gray_arr, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_arr, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            mid_y_start, mid_y_end = h // 5, 4 * h // 5
            mid_x_start, mid_x_end = w // 5, 4 * w // 5
            
            left_region = magnitude[mid_y_start:mid_y_end, :w//2]
            left_profile = np.mean(left_region, axis=0)
            left_px = int(np.argmax(left_profile[10:]) + 10)
            
            right_region = magnitude[mid_y_start:mid_y_end, w//2:]
            right_profile = np.mean(right_region, axis=0)
            right_px = int(np.argmax(right_profile[10:]) + 10)
            
            top_region = magnitude[:h//2, mid_x_start:mid_x_end]
            top_profile = np.mean(top_region, axis=1)
            top_px = int(np.argmax(top_profile[10:]) + 10)
            
            bottom_region = magnitude[h//2:, mid_x_start:mid_x_end]
            bottom_profile = np.mean(bottom_region, axis=1)
            bottom_px = int(np.argmax(bottom_profile[10:]) + 10)
            
            if all(3 < x < min(w, h)//3 for x in [left_px, right_px, top_px, bottom_px]):
                return (left_px, right_px, top_px, bottom_px)
        except Exception as e:
            logger.debug(f"Sobel processing error: {e}")
        return None
    
    def _gradient_detect(self, gray_arr: np.ndarray, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
        """Simple gradient-based scan - fast and reliable fallback."""
        try:
            sample_points = 100
            max_depth = int(min(w, h) * 0.2)
            
            def find_edge(strip: np.ndarray) -> int:
                if len(strip) < 20:
                    return 0
                kernel = np.ones(5) / 5
                smoothed = np.convolve(strip, kernel, mode='valid')
                diffs = np.abs(np.diff(smoothed))
                for i in range(10, len(diffs) - 10):
                    if diffs[i] > 8 and np.mean(diffs[i-5:i+5]) > 5:
                        return i + 5
                return 0
            
            def measure_side(get_strip_func):
                widths = []
                for i in range(sample_points):
                    strip = get_strip_func(i, sample_points)
                    if strip is not None:
                        edge = find_edge(strip.astype(float))
                        if edge > 0:
                            widths.append(edge)
                return widths
            
            left_widths = measure_side(
                lambda i, sp: gray_arr[int(h*0.15 + h*0.70*i/sp), :max_depth]
            )
            right_widths = measure_side(
                lambda i, sp: gray_arr[int(h*0.15 + h*0.70*i/sp), w-max_depth:w][::-1]
            )
            top_widths = measure_side(
                lambda i, sp: gray_arr[:max_depth, int(w*0.15 + w*0.70*i/sp)]
            )
            bottom_widths = measure_side(
                lambda i, sp: gray_arr[h-max_depth:h, int(w*0.15 + w*0.70*i/sp)][::-1]
            )
            
            if all(len(w) > 10 for w in [left_widths, right_widths, top_widths, bottom_widths]):
                return (
                    int(np.median(left_widths)), int(np.median(right_widths)),
                    int(np.median(top_widths)), int(np.median(bottom_widths))
                )
        except Exception as e:
            logger.debug(f"Gradient processing error: {e}")
        return None
    
    def _create_result(self, borders: Tuple[int, int, int, int],
                      w: int, h: int, method: str) -> CenteringResult:
        """Create CenteringResult with grading."""
        left_px, right_px, top_px, bottom_px = borders
        max_border = min(w, h) // 3
        
        left_px = int(clamp(left_px, 2, max_border))
        right_px = int(clamp(right_px, 2, max_border))
        top_px = int(clamp(top_px, 2, max_border))
        bottom_px = int(clamp(bottom_px, 2, max_border))
        
        total_h = max(left_px + right_px, 1)
        total_v = max(top_px + bottom_px, 1)
        
        lr_l = round(left_px / total_h * 100, 1)
        lr_r = round(right_px / total_h * 100, 1)
        tb_t = round(top_px / total_v * 100, 1)
        tb_b = round(bottom_px / total_v * 100, 1)
        
        lr_dev = abs(lr_l - 50.0)
        tb_dev = abs(tb_t - 50.0)
        worst = max(lr_dev, tb_dev)
        
        grade, score = 'Poor', 6.0
        for threshold, g, s in CENTERING_THRESHOLDS:
            if worst <= threshold:
                grade, score = g, s
                break
        
        return CenteringResult(
            left_border=left_px, right_border=right_px,
            top_border=top_px, bottom_border=bottom_px,
            lr_ratio=(lr_l, lr_r), tb_ratio=(tb_t, tb_b),
            grade=grade, score=score, method=method
        )

# ═══════════════════════════════════════════════════════════════════════════════
# EDGE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeAnalyzer:
    """Analyzes edge straightness and wear."""
    
    def analyse(self, img: Image.Image) -> EdgeResult:
        """Analyze all four edges."""
        gray = img.convert('L')
        pixels = np.array(gray, dtype=np.uint8)
        w, h = img.size
        scan_depth = 25
        
        devs = {
            'top': self._edge_straightness(pixels, w, h, 'top', scan_depth),
            'bottom': self._edge_straightness(pixels, w, h, 'bottom', scan_depth),
            'left': self._edge_straightness(pixels, w, h, 'left', scan_depth),
            'right': self._edge_straightness(pixels, w, h, 'right', scan_depth),
        }
        
        worst = max(devs.values())
        flag, score = 'Significant Wear', 4.5
        for threshold, f, s in EDGE_THRESHOLDS:
            if worst < threshold:
                flag, score = f, s
                break
        
        return EdgeResult(
            top_deviation=round(devs['top'], 2),
            bottom_deviation=round(devs['bottom'], 2),
            left_deviation=round(devs['left'], 2),
            right_deviation=round(devs['right'], 2),
            top_flag=flag, bottom_flag=flag,
            left_flag=flag, right_flag=flag,
            score=score
        )
    
    def _edge_straightness(self, pixels: np.ndarray, w: int, h: int,
                          side: str, depth: int) -> float:
        """Measure edge straightness variation."""
        positions = []
        step = 6
        
        if side == 'top':
            for x in range(w//10, w - w//10, step):
                strip = pixels[:depth, x]
                edge = self._find_edge(strip)
                if edge > 0:
                    positions.append(edge)
        elif side == 'bottom':
            for x in range(w//10, w - w//10, step):
                strip = pixels[h-depth:h, x][::-1]
                edge = self._find_edge(strip)
                if edge > 0:
                    positions.append(edge)
        elif side == 'left':
            for y in range(h//10, h - h//10, step):
                strip = pixels[y, :depth]
                edge = self._find_edge(strip)
                if edge > 0:
                    positions.append(edge)
        elif side == 'right':
            for y in range(h//10, h - h//10, step):
                strip = pixels[y, w-depth:w][::-1]
                edge = self._find_edge(strip)
                if edge > 0:
                    positions.append(edge)
        
        return stdev(positions) if len(positions) > 4 else 0.0
    
    def _find_edge(self, strip: np.ndarray) -> int:
        """Find edge transition in a strip."""
        if len(strip) < 10:
            return 0
        diffs = np.abs(np.diff(strip.astype(float)))
        if len(diffs) > 0:
            max_idx = np.argmax(diffs)
            if diffs[max_idx] > 15:
                return int(max_idx)
        return 0

# ═══════════════════════════════════════════════════════════════════════════════
# CORNER ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class CornerAnalyzer:
    """Analyzes corner condition and whitening."""
    
    def analyse(self, img: Image.Image) -> CornerResult:
        """Analyze all four corners."""
        w, h = img.size
        p = CORNER_PATCH
        
        patches = {
            'tl': img.crop((0, 0, p, p)),
            'tr': img.crop((w-p, 0, w, p)),
            'bl': img.crop((0, h-p, p, h)),
            'br': img.crop((w-p, h-p, w, h)),
        }
        
        ratings = {}
        whitenings = {}
        
        for name, patch in patches.items():
            rating, whitening = self._analyse_corner_patch(patch, name)
            ratings[name] = rating
            whitenings[name] = whitening
        
        avg_grade = mean([CORNER_GRADE_MAP.get(ratings[k], 5.0) for k in ratings])
        score = round(avg_grade / 10.0 * 20.0, 2)
        
        return CornerResult(
            tl=ratings['tl'], tr=ratings['tr'],
            bl=ratings['bl'], br=ratings['br'],
            tl_whitening=whitenings['tl'], tr_whitening=whitenings['tr'],
            bl_whitening=whitenings['bl'], br_whitening=whitenings['br'],
            score=score
        )
    
    def _analyse_corner_patch(self, patch: Image.Image, corner: str) -> Tuple[str, float]:
        """Analyze a single corner patch."""
        p = patch.width
        gray_pixels = np.array(patch.convert('L'), dtype=np.uint8)
        rgb_pixels = np.array(patch.convert('RGB'), dtype=np.uint8)
        
        tip_size = p // 4
        tip_mask = np.zeros((p, p), dtype=bool)
        
        for y in range(p):
            for x in range(p):
                if corner == 'tl' and x < tip_size and y < tip_size and x + y < tip_size:
                    tip_mask[y, x] = True
                elif corner == 'tr' and x >= p - tip_size and y < tip_size and (p-1-x) + y < tip_size:
                    tip_mask[y, x] = True
                elif corner == 'bl' and x < tip_size and y >= p - tip_size and x + (p-1-y) < tip_size:
                    tip_mask[y, x] = True
                elif corner == 'br' and x >= p - tip_size and y >= p - tip_size and (p-1-x) + (p-1-y) < tip_size:
                    tip_mask[y, x] = True
        
        tip_gray = gray_pixels[tip_mask]
        tip_rgb = rgb_pixels[tip_mask]
        
        if len(tip_gray) == 0:
            return 'Sharp', 0.0
        
        white_count = 0
        for i, v in enumerate(tip_gray):
            if v > 210:
                r, g, b = tip_rgb[i]
                mx = max(r, g, b)
                mn = min(r, g, b)
                saturation = (mx - mn) / mx if mx > 0 else 0
                if saturation < 0.25:
                    white_count += 1
        
        whitening_pct = white_count / len(tip_gray) * 100
        
        mid_region = gray_pixels[p//4:3*p//4, p//4:3*p//4]
        mid_brightness = float(np.mean(mid_region)) if mid_region.size > 0 else 128
        tip_brightness = float(np.mean(tip_gray))
        brightness_diff = tip_brightness - mid_brightness
        
        if whitening_pct > 30 or brightness_diff > 60:
            return 'Dinged', round(whitening_pct, 1)
        elif whitening_pct > 15 or brightness_diff > 35:
            return 'Rounded', round(whitening_pct, 1)
        elif whitening_pct > 6 or brightness_diff > 18:
            return 'Slightly Rounded', round(whitening_pct, 1)
        else:
            return 'Sharp', round(whitening_pct, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# SURFACE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class SurfaceAnalyzer:
    """Analyzes surface condition including scratches, haze, and texture."""
    
    def analyse(self, img: Image.Image) -> SurfaceResult:
        """Perform complete surface analysis."""
        w, h = img.size
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        face = img.crop((pad_x, pad_y, w - pad_x, h - pad_y))
        
        gray = face.convert('L')
        gw, gh = gray.size
        gpx = np.array(gray, dtype=np.uint8).ravel()
        
        # Haze detection
        overall_std = float(np.std(gpx))
        haze_level = round(clamp((HAZE_STD_THRESHOLD - overall_std) / HAZE_STD_THRESHOLD * 100.0, 0, 100), 1)
        haze_detected = haze_level > 30.0
        
        # Scratch detection (block variance analysis)
        block = BLOCK_SIZE
        high_var_blocks = 0
        total_blocks = 0
        gray_arr = np.array(gray, dtype=np.uint8)
        
        for by in range(0, gh - block, block):
            for bx in range(0, gw - block, block):
                patch = gray_arr[by:by+block, bx:bx+block]
                if float(np.std(patch)) > SCRATCH_VARIANCE_THRESHOLD:
                    high_var_blocks += 1
                total_blocks += 1
        
        scratch_ratio = high_var_blocks / total_blocks if total_blocks else 0
        scratch_severity = round(clamp(scratch_ratio * 400, 0, 100), 1)
        scratches_detected = scratch_severity > 18.0
        
        # Print defect detection (outlier analysis)
        face_mean = float(np.mean(gpx))
        face_std = float(np.std(gpx))
        if face_std > 0:
            z_scores = np.abs((gpx - face_mean) / face_std)
            outlier_count = int(np.sum(z_scores > OUTLIER_STD_MULTIPLIER))
        else:
            outlier_count = 0
        outlier_pct = outlier_count / len(gpx) * 100 if len(gpx) > 0 else 0
        defect_severity = round(clamp(outlier_pct * 10, 0, 100), 1)
        print_defects_detected = defect_severity > 20.0
        
        # LBP texture analysis
        lbp_severity, lbp_flagged_boxes = self._analyse_lbp_texture(gray)
        lbp_defects_detected = lbp_severity > 15.0
        
        # Scoring
        surface_quality_score = 20.0
        print_lines_score = 5.0
        missing_texture_score = 5.0
        deductions = []
        
        if haze_detected:
            pts = round(min(haze_level / 100.0 * 6.0, 6.0), 2)
            surface_quality_score -= pts
            deductions.append(('Surface haze / scuffing', pts))
        
        if scratches_detected:
            pts = round(min(scratch_severity / 100.0 * 8.0, 8.0), 2)
            surface_quality_score -= pts
            deductions.append(('Scratches detected', pts))
        
        if print_defects_detected:
            pts = round(min(defect_severity / 100.0 * 6.0, 6.0), 2)
            surface_deduction = pts * 0.3
            print_lines_deduction = pts * 0.7
            surface_quality_score -= surface_deduction
            print_lines_score -= print_lines_deduction
            deductions.append(('Print / ink defects (surface)', round(surface_deduction, 2)))
            deductions.append(('Print / ink defects (print lines)', round(print_lines_deduction, 2)))
        
        if lbp_defects_detected:
            print_lines_deduction = round(min(lbp_severity / 100.0 * 2.5, 2.5), 2)
            missing_texture_deduction = round(min(lbp_severity / 100.0 * 2.5, 2.5), 2)
            print_lines_score -= print_lines_deduction
            missing_texture_score -= missing_texture_deduction
            deductions.append(('Print lines (texture anomalies)', round(print_lines_deduction, 2)))
            deductions.append(('Missing texture', round(missing_texture_deduction, 2)))
        
        surface_quality_score = round(clamp(surface_quality_score, 0.0, 20.0), 2)
        print_lines_score = round(clamp(print_lines_score, 0.0, 5.0), 2)
        missing_texture_score = round(clamp(missing_texture_score, 0.0, 5.0), 2)
        total_surface_score = surface_quality_score + print_lines_score + missing_texture_score
        
        grade = 'Heavy Wear'
        for threshold, g in SURFACE_THRESHOLDS:
            if total_surface_score >= threshold:
                grade = g
                break
        
        return SurfaceResult(
            scratches_detected=scratches_detected,
            print_defects_detected=print_defects_detected,
            haze_detected=haze_detected,
            lbp_defects_detected=lbp_defects_detected,
            scratch_severity=scratch_severity,
            defect_severity=defect_severity,
            haze_level=haze_level,
            lbp_severity=lbp_severity,
            grade=grade,
            score=total_surface_score,
            deductions=deductions,
            lbp_flagged_boxes=lbp_flagged_boxes,
            print_lines_score=print_lines_score,
            missing_texture_score=missing_texture_score
        )
    
    def _analyse_lbp_texture(self, gray_img: Image.Image) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        """LBP texture analysis for print defects."""
        try:
            from skimage.feature import local_binary_pattern
            
            gw, gh = gray_img.size
            arr = np.array(gray_img, dtype=np.uint8)
            
            P, R = LBP_POINTS, LBP_RADIUS
            n_bins = P + 2
            
            cols, rows = LBP_GRID_COLS, LBP_GRID_ROWS
            pw = gw // cols
            ph = gh // rows
            
            lbp_map = local_binary_pattern(arr, P, R, method='uniform')
            
            hists = {}
            for gy in range(rows):
                for gx in range(cols):
                    x0, y0 = gx * pw, gy * ph
                    x1, y1 = x0 + pw, y0 + ph
                    patch_lbp = lbp_map[y0:y1, x0:x1].ravel()
                    hist, _ = np.histogram(patch_lbp, bins=n_bins, range=(0, n_bins), density=True)
                    hists[(gx, gy)] = hist
            
            chi_scores = {}
            for (gx, gy), h in hists.items():
                neighbour_hists = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nb = hists.get((gx + dx, gy + dy))
                        if nb is not None:
                            neighbour_hists.append(nb)
                
                if not neighbour_hists:
                    chi_scores[(gx, gy)] = 0.0
                    continue
                
                dists = []
                for nb in neighbour_hists:
                    denom = h + nb
                    with np.errstate(divide='ignore', invalid='ignore'):
                        chi = np.where(denom > 0, (h - nb) ** 2 / denom, 0.0).sum()
                    dists.append(chi)
                chi_scores[(gx, gy)] = float(np.median(dists))
            
            all_scores = list(chi_scores.values())
            if not all_scores:
                return 0.0, []
            
            arr_scores = np.array(all_scores)
            threshold = float(arr_scores.mean() + LBP_CHI_THRESHOLD_FACTOR * arr_scores.std())
            threshold = max(threshold, LBP_MIN_THRESHOLD)
            
            flagged_boxes = []
            for (gx, gy), score in chi_scores.items():
                if score > threshold:
                    x0, y0 = gx * pw, gy * ph
                    flagged_boxes.append((x0, y0, pw, ph))
            
            flagged_ratio = len(flagged_boxes) / (cols * rows)
            lbp_severity = round(clamp(flagged_ratio * 250, 0, 100), 1)
            
            return lbp_severity, flagged_boxes
            
        except ImportError:
            logger.debug("skimage not available, skipping LBP analysis")
            return 0.0, []
        except Exception as e:
            logger.debug(f"LBP analysis failed: {e}")
            return 0.0, []

# ═══════════════════════════════════════════════════════════════════════════════
# GRADER
# ═══════════════════════════════════════════════════════════════════════════════

class Grader:
    """Calculates overall scores and generates recommendations."""
    
    def calculate_overall_score(self, centering: CenteringResult, edges: EdgeResult,
                               corners: CornerResult, surface: SurfaceResult) -> float:
        """Calculate weighted overall score out of 100."""
        centering_pct = (centering.score / centering.max_score) * 100
        edges_pct = (edges.score / edges.max_score) * 100
        corners_pct = (corners.score / corners.max_score) * 100
        
        surface_quality_score = surface.score - surface.print_lines_score - surface.missing_texture_score
        surface_quality_pct = (surface_quality_score / 20.0) * 100
        print_lines_pct = (surface.print_lines_score / 5.0) * 100
        missing_texture_pct = (surface.missing_texture_score / 5.0) * 100
        
        total = (
            centering_pct * SCORING_WEIGHTS['centering'] +
            edges_pct * SCORING_WEIGHTS['edges'] +
            corners_pct * SCORING_WEIGHTS['corners'] +
            surface_quality_pct * SCORING_WEIGHTS['surface_quality'] +
            print_lines_pct * SCORING_WEIGHTS['print_lines'] +
            missing_texture_pct * SCORING_WEIGHTS['missing_texture']
        )
        return round(total, 1)
    
    def recommend(self, overall: float, centering: CenteringResult,
                 edges: EdgeResult, corners: CornerResult,
                 surface: SurfaceResult) -> Tuple[str, str, str, str, bool]:
        """Generate grade band and recommendation."""
        # Determine grade band
        band = 'Poor'
        for threshold, b in GRADE_BANDS:
            if overall >= threshold:
                band = b
                break
        
        # Generate grade string
        if overall >= 95:
            grade_str = 'GEM-MT 10'
        elif overall >= 90:
            grade_str = 'MINT 9'
        elif overall >= 80:
            grade_str = 'NM-MT 8'
        elif overall >= 70:
            grade_str = 'NM 7'
        elif overall >= 60:
            grade_str = 'EX-MT 6'
        elif overall >= 50:
            grade_str = 'EX 5'
        elif overall >= 40:
            grade_str = 'VG-EX 4'
        elif overall >= 30:
            grade_str = 'VG 3'
        elif overall >= 20:
            grade_str = 'GOOD 2'
        elif overall >= 10:
            grade_str = 'FAIR 1'
        else:
            grade_str = 'POOR 0'
        
        # Beckett candidate check
        beckett = (
            overall >= 90 and
            centering.score >= 22.5 and
            edges.score >= 13.5 and
            corners.score >= 18.0 and
            surface.score >= 27.0
        )
        
        # Build issues list
        issues = []
        
        if centering.score < 22.5:
            lr_diff = abs(centering.lr_ratio[0] - centering.lr_ratio[1])
            tb_diff = abs(centering.tb_ratio[0] - centering.tb_ratio[1])
            if lr_diff > 10 or tb_diff > 10:
                issues.append(f"Off-center (L/R: {centering.lr_ratio[0]:.1f}/{centering.lr_ratio[1]:.1f}, T/B: {centering.tb_ratio[0]:.1f}/{centering.tb_ratio[1]:.1f})")
            else:
                issues.append("Minor centering issues")
        
        if edges.score < 13.5:
            worst_edge = max(edges.top_deviation, edges.bottom_deviation, edges.left_deviation, edges.right_deviation)
            if worst_edge > 5:
                issues.append(f"Edge wear detected (worst: {worst_edge:.1f}px deviation)")
            else:
                issues.append("Minor edge wear")
        
        if corners.score < 18.0:
            corner_ratings = [corners.tl, corners.tr, corners.bl, corners.br]
            rating_order = ['Sharp', 'Slightly Rounded', 'Rounded', 'Dinged']
            worst_corner = max(corner_ratings, key=lambda x: rating_order.index(x) if x in rating_order else 0)
            if worst_corner in ['Rounded', 'Dinged']:
                issues.append(f"Corner damage ({worst_corner})")
            else:
                issues.append("Minor corner wear")
        
        if surface.score < 27.0:
            if surface.scratches_detected:
                issues.append(f"Scratches (severity: {surface.scratch_severity:.1f}%)")
            if surface.haze_detected:
                issues.append(f"Surface haze (level: {surface.haze_level:.1f}%)")
            if surface.print_defects_detected:
                issues.append(f"Print defects (severity: {surface.defect_severity:.1f}%)")
            if surface.lbp_defects_detected:
                issues.append(f"Texture anomalies (severity: {surface.lbp_severity:.1f}%)")
        
        # Generate recommendation
        if beckett:
            rec = 'Beckett (BGS)'
            reason = 'This card shows characteristics consistent with a BGS 9.5 or Black Label 10. Beckett subgrades reward perfection in every category. Consider submitting to Beckett Grading Services.'
        elif overall >= 85:
            rec = 'PSA'
            reason = 'Gem Mint. Near-flawless across all categories — PSA 10 territory. Strong resale value and market liquidity.'
        elif overall >= 75:
            rec = 'PSA'
            reason = 'Mint. Minor imperfections keep this from a 10, but PSA 9 still commands excellent premiums and is very worth grading.'
        elif overall >= 65:
            rec = 'PSA'
            reason = 'Near Mint–Mint. Noticeable but not severe flaws. PSA 8 is a solid mid-grade with good market demand.'
        elif overall >= 55:
            rec = 'CGC'
            reason = 'Near Mint. Light play wear across multiple categories. CGC offers competitive pricing for this grade tier.'
        elif overall >= 45:
            rec = 'CGC'
            reason = 'Excellent to Excellent–Mint. Moderate wear visible. CGC is the most cost-effective option at this grade.'
        elif overall >= 35:
            rec = "Don't Grade"
            reason = 'Very Good. Significant wear on multiple surfaces. Grading fees likely exceed the value added at this grade.'
        else:
            rec = "Don't Grade"
            reason = 'Poor to Fair. Heavy wear detected across the card. Not worth grading — consider raw sale instead.'
        
        return grade_str, band, rec, reason, beckett

# ═══════════════════════════════════════════════════════════════════════════════
# ANNOTATOR
# ═══════════════════════════════════════════════════════════════════════════════

class Annotator:
    """Builds annotated result images."""
    
    def build(self, img: Image.Image, centering: CenteringResult,
             edges: EdgeResult, corners: CornerResult, surface: SurfaceResult,
             overall_score: float, out_path: str):
        """Build and save annotated image."""
        out = img.convert('RGB').copy()
        draw = ImageDraw.Draw(out, 'RGBA')
        w, h = out.size
        
        l, r, t, b = centering.left_border, centering.right_border, centering.top_border, centering.bottom_border
        
        # Draw border overlays
        overlay_thickness = max(6, min(l, r, t, b))
        
        for edge_flag, rect in [
            (edges.top_flag, [0, 0, w, t + overlay_thickness]),
            (edges.bottom_flag, [0, h - b - overlay_thickness, w, h]),
            (edges.left_flag, [0, 0, l + overlay_thickness, h]),
            (edges.right_flag, [w - r - overlay_thickness, 0, w, h]),
        ]:
            col = self._edge_color(edge_flag)
            if col:
                draw.rectangle(rect, fill=col)
        
        # Inner artwork border
        draw.rectangle([l, t, w - r, h - b], outline=(0, 220, 100, 220), width=2)
        
        # Center crosshair
        cx = l + (w - l - r) // 2
        cy = t + (h - t - b) // 2
        draw.line([(cx - 14, cy), (cx + 14, cy)], fill=(255, 100, 100, 200), width=1)
        draw.line([(cx, cy - 14), (cx, cy + 14)], fill=(255, 100, 100, 200), width=1)
        
        # Text overlay
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        
        overlay = Image.new('RGBA', out.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        text_bg = (0, 0, 0, 200)
        text_color = (255, 255, 255, 255)
        
        lines = [
            f"Overall: {overall_score:.1f}/100",
            f"Centering: {centering.score:.1f}/25 ({centering.grade})",
            f"Surface: {surface.score:.1f}/30 ({surface.grade})",
            f"Corners: {corners.score:.1f}/20",
            f"Edges: {edges.score:.1f}/15",
            f"L/R: {centering.lr_ratio[0]:.1f}% / {centering.lr_ratio[1]:.1f}%",
            f"T/B: {centering.tb_ratio[0]:.1f}% / {centering.tb_ratio[1]:.1f}%",
        ]
        
        y_offset = 10
        for line in lines:
            bbox = overlay_draw.textbbox((0, 0), line, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            overlay_draw.rectangle([10, y_offset, 10 + tw + 20, y_offset + th + 10], fill=text_bg)
            overlay_draw.text((20, y_offset + 5), line, fill=text_color, font=font)
            y_offset += th + 15
        
        # Composite and save
        out = Image.alpha_composite(out.convert('RGBA'), overlay)
        out.convert('RGB').save(out_path, 'JPEG', quality=95)
    
    def _edge_color(self, flag: str) -> Optional[Tuple[int, int, int, int]]:
        """Get color for edge wear flag."""
        if flag == 'Minor Wear':
            return (255, 200, 60, 55)
        if flag == 'Moderate Wear':
            return (230, 100, 30, 75)
        if flag == 'Significant Wear':
            return (220, 30, 30, 95)
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CARD ANALYSER (Orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

class CardAnalyser:
    """
    Main card analysis orchestrator.
    Composes all detection, scoring, and visualization components.
    """
    
    def __init__(self):
        """Initialize all analysis components."""
        self.card_detector = CardDetector()
        self.border_detector = ConsensusBorderDetector()
        self.edge_analyzer = EdgeAnalyzer()
        self.corner_analyzer = CornerAnalyzer()
        self.surface_analyzer = SurfaceAnalyzer()
        self.grader = Grader()
        self.annotator = Annotator()
        
        logger.info("CardAnalyser v5.1 initialized successfully")
    
    def analyse(self, image_path: str) -> AnalysisResult:
        """
        Perform complete card analysis.
        
        Args:
            image_path: Path to the card image file
            
        Returns:
            AnalysisResult with all grading information
        """
        # Step 1: Load and prepare image
        img = self._load_image(image_path)
        if img is None:
            return self._fail("Could not open or process image")
        
        # Step 2: Detect card boundaries and crop
        card_img = self._detect_and_crop_card(img)
        
        # Step 3: Run all analyses
        try:
            centering = self._analyse_centering(card_img)
            edges = self.edge_analyzer.analyse(card_img)
            corners = self.corner_analyzer.analyse(card_img)
            surface = self.surface_analyzer.analyse(card_img)
        except Exception as e:
            logger.exception("Analysis pipeline failed")
            return self._fail(f"Analysis error: {str(e)}")
        
        # Step 4: Build annotated image
        annotated_path = self._build_annotated_image(
            card_img, centering, edges, corners, surface
        )
        
        # Step 5: Calculate overall score
        overall_score = self.grader.calculate_overall_score(
            centering, edges, corners, surface
        )
        
        # Step 6: Generate recommendations
        grade_str, band, rec, rec_reason, beckett = self.grader.recommend(
            overall_score, centering, edges, corners, surface
        )
        
        logger.info(f"Analysis complete: {grade_str} ({overall_score}/100)")
        
        return AnalysisResult(
            success=True,
            error_msg='',
            annotated_path=annotated_path,
            centering=centering,
            edges=edges,
            corners=corners,
            surface=surface,
            estimated_grade=grade_str,
            grade_band=band,
            recommendation=rec,
            recommendation_reason=rec_reason,
            beckett_candidate=beckett,
            overall_score=overall_score,
            defect_images=[]
        )
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and preprocess image."""
        try:
            img = Image.open(image_path).convert('RGB')
            logger.info(f"Loaded image: {image_path} ({img.size})")
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            return None
        
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            logger.debug("EXIF transpose skipped")
        
        return img
    
    def _detect_and_crop_card(self, img: Image.Image) -> Image.Image:
        """Detect card boundaries and crop to standard size."""
        bounds = self.card_detector.find_bounds(img)
        
        if bounds is None:
            logger.warning("Card detection failed, using full image")
            bounds = (0, 0, img.width, img.height)
        
        card_img = img.crop(bounds)
        card_img = card_img.resize((CARD_W, CARD_H), Image.LANCZOS)
        logger.info(f"Card cropped and resized to {CARD_W}x{CARD_H}")
        
        return card_img
    
    def _analyse_centering(self, img: Image.Image) -> CenteringResult:
        """Analyze card centering using consensus border detection."""
        return self.border_detector.detect(img)
    
    def _build_annotated_image(self, img: Image.Image, centering: CenteringResult,
                              edges: EdgeResult, corners: CornerResult,
                              surface: SurfaceResult) -> str:
        """Build annotated result image and return temp file path."""
        annotated_tmp = tempfile.NamedTemporaryFile(
            suffix='_annotated.jpg', delete=False
        )
        annotated_tmp.close()
        
        overall_score = self.grader.calculate_overall_score(
            centering, edges, corners, surface
        )
        
        self.annotator.build(
            img, centering, edges, corners, surface,
            overall_score, annotated_tmp.name
        )
        
        return annotated_tmp.name
    
    def _fail(self, msg: str) -> AnalysisResult:
        """Create a failure result."""
        logger.error(f"Analysis failed: {msg}")
        return AnalysisResult(
            success=False,
            error_msg=msg,
            estimated_grade='Error',
            grade_band='Error',
            recommendation='Unable to analyse',
            recommendation_reason=msg,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pokemon Card Analysis Engine v5.1')
    parser.add_argument('image', help='Path to card image', nargs='?')
    parser.add_argument('--output', '-o', help='Output path for annotated image', default=None)
    parser.add_argument('--json', '-j', action='store_true', help='Output JSON format')
    args = parser.parse_args()
    
    if not args.image:
        print("\n" + "="*60)
        print("POKEMON CARD ANALYSIS ENGINE v5.1")
        print("="*60)
        print("\nFeatures:")
        print("  • Automatic card detection and analysis")
        print("  • Consensus border detection (Sobel + Gradient)")
        print("  • LBP texture analysis for surface defects")
        print("  • Weighted scoring (Surface 30%, Centering 25%, Corners 20%, Edges 15%, Print Lines 10%)")
        print("\nUsage:")
        print("  python analyser.py <image_path>")
        print("  python analyser.py <image_path> --json")
        print("  python analyser.py <image_path> -o result.jpg")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    analyser = CardAnalyser()
    result = analyser.analyse(args.image)
    
    if not result.success:
        print(f"\n❌ Analysis failed: {result.error_msg}")
        sys.exit(1)
    
    if args.json:
        output = {
            'success': result.success,
            'estimated_grade': result.estimated_grade,
            'grade_band': result.grade_band,
            'overall_score': result.overall_score,
            'beckett_candidate': result.beckett_candidate,
            'recommendation': result.recommendation,
            'recommendation_reason': result.recommendation_reason,
        }
        if result.centering:
            output['centering'] = {
                'lr_ratio': result.centering.lr_ratio,
                'tb_ratio': result.centering.tb_ratio,
                'grade': result.centering.grade,
                'score': result.centering.score,
                'method': result.centering.method,
            }
        if result.surface:
            output['surface'] = {
                'grade': result.surface.grade,
                'score': result.surface.score,
                'scratches_detected': result.surface.scratches_detected,
                'haze_detected': result.surface.haze_detected,
            }
        print(json.dumps(output, indent=2))
    else:
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        print(f"Overall Score: {result.overall_score:.1f}/100")
        print(f"Grade: {result.estimated_grade} ({result.grade_band})")
        print(f"Beckett Candidate: {'✅ YES' if result.beckett_candidate else '❌ NO'}")
        print(f"Recommendation: {result.recommendation}")
        if result.centering:
            print(f"\nCentering: {result.centering.score:.1f}/25 ({result.centering.grade})")
            print(f"  L/R: {result.centering.lr_ratio[0]:.1f}% / {result.centering.lr_ratio[1]:.1f}%")
            print(f"  T/B: {result.centering.tb_ratio[0]:.1f}% / {result.centering.tb_ratio[1]:.1f}%")
            print(f"  Method: {result.centering.method}")
        if result.surface:
            print(f"\nSurface: {result.surface.score:.1f}/30 ({result.surface.grade})")
            if result.surface.deductions:
                print("  Deductions:")
                for reason, pts in result.surface.deductions:
                    print(f"    - {reason}: -{pts} pts")
        print("="*60)
    
    if result.annotated_path:
        output_path = args.output if args.output else result.annotated_path
        if args.output:
            import shutil
            shutil.copy2(result.annotated_path, output_path)
        print(f"\n📁 Annotated image saved to: {output_path}")
    
    sys.exit(0)
