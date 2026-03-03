import cv2
import numpy as np
from PIL import Image
import re

class UIAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.results = {
            'text_alignment_issues': [],
            'ui_element_anomalies': [],
            'font_consistency_score': 100,
            'overall_ui_score': 100
        }
    
    def detect_text_regions(self):
        """Detect text regions in the screenshot"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 20 < w < 500 and 10 < h < 100:  # Typical text size
                text_regions.append((x, y, w, h))
        
        return text_regions
    
    def check_text_alignment(self):
        """Check if text is properly aligned"""
        text_regions = self.detect_text_regions()
        
        if len(text_regions) < 3:
            return
        
        # Group by approximate y-coordinate (same line)
        lines = {}
        for (x, y, w, h) in text_regions:
            line_key = round(y / 20)  # Group within 20 pixels
            if line_key not in lines:
                lines[line_key] = []
            lines[line_key].append((x, w))
        
        # Check alignment within each line
        issues = []
        for line_key, regions in lines.items():
            if len(regions) > 1:
                # Check if any region is misaligned
                base_x = regions[0][0]
                for x, w in regions[1:]:
                    if abs(x - base_x) > 10:  # More than 10px difference
                        issues.append(f"Misaligned text detected at line {line_key}")
                        self.results['overall_ui_score'] -= 5
                        break
        
        self.results['text_alignment_issues'] = issues
    
    def detect_whatsapp_elements(self):
        """Detect WhatsApp-specific UI elements"""
        height, width = self.image.shape[:2]
        issues = []
        
        # Check for green WhatsApp header (typical color: #075E54)
        header_region = self.image[0:60, 0:width]  # Top 60px
        avg_header_color = np.mean(header_region, axis=(0, 1))
        
        # Convert to hex-like comparison
        if not (30 < avg_header_color[0] < 100 and  # B channel
                80 < avg_header_color[1] < 150 and  # G channel
                0 < avg_header_color[2] < 60):      # R channel
            issues.append("WhatsApp header color anomaly")
            self.results['overall_ui_score'] -= 10
        
        # Check for timestamp format (e.g., "10:30 AM")
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Look for timestamp-like patterns using template matching
        # This is simplified - you'd want more sophisticated pattern matching
        timestamp_pattern = self.create_timestamp_template()
        
        result = cv2.matchTemplate(gray, timestamp_pattern, cv2.TM_CCOEFF_NORMED)
        if np.max(result) < 0.3:  # Low confidence match
            issues.append("Timestamp format doesn't match WhatsApp pattern")
            self.results['overall_ui_score'] -= 15
        
        self.results['ui_element_anomalies'] = issues
    
    def create_timestamp_template(self):
        """Create a template for timestamp pattern"""
        # Simplified - create a pattern resembling a timestamp
        template = np.zeros((20, 80), dtype=np.uint8)
        cv2.putText(template, "10:30", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        return template
    
    def check_bubble_consistency(self):
        """Check if message bubbles are consistent"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Detect circles/bubbles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=10, maxRadius=50)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Check if bubble radii are consistent
            if len(circles) > 1:
                radii = [c[2] for c in circles]
                radius_std = np.std(radii)
                
                if radius_std > 10:  # Inconsistent bubble sizes
                    self.results['ui_element_anomalies'].append("Inconsistent chat bubble sizes")
                    self.results['overall_ui_score'] -= 10
    
    def analyze(self):
        """Run all UI analyses"""
        self.check_text_alignment()
        self.detect_whatsapp_elements()
        self.check_bubble_consistency()
        
        # Normalize score
        self.results['overall_ui_score'] = max(0, min(100, self.results['overall_ui_score']))
        
        return self.results