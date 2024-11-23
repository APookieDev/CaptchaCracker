from flask import Flask, request, jsonify
import easyocr
import cv2
import re
import numpy as np
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Initialize the OCR reader globally
reader = easyocr.Reader(['en'], gpu=False)

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def preprocess_image(image_array):
    # Convert to grayscale
    grayscale_image = grayscale(image_array)
    
    # Convert to binary
    _, bw_image = cv2.threshold(grayscale_image, 177, 255, cv2.THRESH_BINARY)
    
    # Remove line
    noline_image = bw_image.copy()
    line_y = 23
    for x in range(noline_image.shape[1]): 
        if noline_image[line_y - 1, x] == 0 and noline_image[line_y + 1, x] == 0:
            noline_image[line_y, x] = 0
        else:
            noline_image[line_y, x] = noline_image[line_y - 1, x]
    
    # Remove border
    noborder_image = noline_image.copy()
    border_thickness = 1
    noborder_image[:border_thickness, :] = 255
    noborder_image[-border_thickness:, :] = 255 
    noborder_image[:, :border_thickness] = 255 
    noborder_image[:, -border_thickness:] = 255 
    
    return noborder_image

def extract_text(image_array, min_confidence=0.2):
    try:
        # Perform OCR with adjusted parameters
        results = reader.readtext(
            image_array,
            paragraph=False,
            height_ths=0.8,
            width_ths=0.8,
            ycenter_ths=0.5,
            decoder='greedy',
            beamWidth=5,
            allowlist='0123456789abcdef'
        )
        
        # Filter for alphanumeric characters (a-f, 0-9)
        filtered_text = []
        pattern = r'[a-f0-9]+'
        
        for detection in results:
            text = detection[1]
            confidence = detection[2]
            
            if confidence >= min_confidence:
                matches = re.findall(pattern, text)
                if matches:
                    filtered_text.append({
                        'text': text,
                        'confidence': confidence
                    })
        
        return filtered_text
    
    except Exception as e:
        print(f"Error in extract_text: {str(e)}")
        return []

@app.route('/')
def home():
    return "Captcha Solver API is running!"

@app.route('/solve', methods=['POST'])
def solve_captcha():
    try:
        # Check if image file is present in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Read image file into memory
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Process image
        processed_image = preprocess_image(image)
        
        # Extract text
        detections = extract_text(processed_image)
        
        if not detections:
            return jsonify({'error': 'No text detected'}), 404
        
        # Return the detected text with highest confidence
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        return jsonify({
            'text': best_detection['text'],
            'confidence': float(best_detection['confidence'])
        })
        
    except Exception as e:
        print(f"Error in solve_captcha: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
