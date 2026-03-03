import cv2
import numpy as np
from PIL import Image, ImageChops
import exifread
import os
from skimage.metrics import structural_similarity as ssim

class ForensicAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.pil_image = Image.open(image_path)
        self.results = {
            'ela_score': 0,
            'metadata_issues': [],
            'compression_anomalies': [],
            'copy_move_detected': False,
            'overall_score': 100
        }
    
    def analyze_metadata(self):
        """Check for suspicious or missing metadata"""
        with open(self.image_path, 'rb') as f:
            tags = exifread.process_file(f)
        
        issues = []
        
        # Check if metadata exists
        if not tags:
            issues.append("No metadata found - possible screenshot or stripped metadata")
            self.results['overall_score'] -= 20
        
        # Check for editing software signatures
        editing_tools = ['Photoshop', 'Lightroom', 'GIMP', 'Snapseed', 'PicsArt']
        for tag in tags.values():
            if any(tool.lower() in str(tag).lower() for tool in editing_tools):
                issues.append(f"Edited with: {tag}")
                self.results['overall_score'] -= 15
        
        self.results['metadata_issues'] = issues
        return issues
    
    def error_level_analysis(self, quality=90):
        """Perform Error Level Analysis to detect edited regions"""
        # Save image at specific quality
        temp_path = 'temp_ela.jpg'
        self.pil_image.save(temp_path, 'JPEG', quality=quality)
        
        # Reload and compare
        original = Image.open(self.image_path)
        compressed = Image.open(temp_path)
        
        # Calculate difference
        diff = ImageChops.difference(original, compressed)
        
        # Convert to numpy array for analysis
        diff_array = np.array(diff)
        
        # Calculate statistics
        mean_diff = np.mean(diff_array)
        std_diff = np.std(diff_array)
        
        # Higher variance suggests tampering
        if std_diff > 15:
            self.results['ela_score'] = min(100, std_diff * 5)
            self.results['overall_score'] -= std_diff * 0.5
        
        # Clean up
        os.remove(temp_path)
        
        return {
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'ela_image': diff_array.tolist() if std_diff > 10 else None
        }
    
    def detect_copy_move_forgery(self):
        """Detect if parts of the image were copied and pasted"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Use ORB feature detector
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        # Check for duplicate regions using feature matching
        if descriptors is not None and len(keypoints) > 10:
            # Simple check: if many similar keypoints, possible copy-move
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors, descriptors)
            
            # Filter good matches (excluding self-matches)
            good_matches = [m for m in matches if m.distance < 50 and m.queryIdx != m.trainIdx]
            
            if len(good_matches) > len(keypoints) * 0.3:
                self.results['copy_move_detected'] = True
                self.results['overall_score'] -= 30
    
    def analyze_compression_artifacts(self):
        """Check for inconsistent compression artifacts"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Calculate DCT coefficients to check blocking artifacts
        dct_coefficients = cv2.dct(np.float32(gray[:100, :100]))
        
        # High frequency components indicate JPEG artifacts
        high_freq = np.sum(np.abs(dct_coefficients[50:, 50:]))
        total = np.sum(np.abs(dct_coefficients))
        
        if total > 0:
            artifact_ratio = high_freq / total
            if artifact_ratio > 0.3:
                self.results['compression_anomalies'].append("Unusual compression patterns detected")
                self.results['overall_score'] -= 10
    
    def run_full_analysis(self):
        """Run all forensic analyses"""
        self.analyze_metadata()
        self.error_level_analysis()
        self.detect_copy_move_forgery()
        self.analyze_compression_artifacts()
        
        # Ensure score is between 0-100
        self.results['overall_score'] = max(0, min(100, self.results['overall_score']))
        
        return self.results