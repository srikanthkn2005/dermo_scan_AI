import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- 1. MEDICAL KNOWLEDGE BASE ---
disease_info = {
    'Acne': {
        'desc': 'A common skin condition where hair follicles become plugged with oil and dead skin cells.',
        'symptoms': 'Whiteheads, blackheads, small red tender bumps (papules), or large solid painful lumps.',
        'action': 'Keep the area clean. Avoid touching or popping pimples. Apply a cold compress if painful.',
        'treatment': 'Over-the-counter creams (Benzoyl Peroxide), cleansers with Salicylic Acid.',
        'severity': 'Medium',
        'red_flags': ['Painful deep cysts', 'Severe scarring', 'No improvement after 3 months'],
        'visit_doctor': 'If cysts are painful or leaving scars, or if OTC treatments fail.'
    },
    'Eczema': {
        'desc': 'Atopic dermatitis (eczema) is a condition that makes your skin red and itchy. It damages the skin barrier function.',
        'symptoms': 'Dry skin, itching (especially at night), red to brownish-gray patches, and thickened skin.',
        'action': 'Moisturize immediately. Avoid scratching. Use mild soap and lukewarm water.',
        'treatment': 'Medical moisturizers, corticosteroid creams, and antihistamines for itching.',
        'severity': 'Chronic / Manageable',
        'red_flags': ['Yellow crusting (infection)', 'Fever', 'Worsening redness'],
        'visit_doctor': 'If sleep is disrupted by itching or signs of infection appear.'
    },
    'Rosacea': {
        'desc': 'A chronic skin condition causing redness and visible blood vessels in the face.',
        'symptoms': 'Facial redness, swollen red bumps, eye problems, and an enlarged nose.',
        'action': 'Avoid triggers (spicy food, hot drinks). Protect your face from the sun. Do not scrub.',
        'treatment': 'Topical drugs to reduce redness, oral antibiotics, and laser therapy.',
        'severity': 'Chronic',
        'red_flags': ['Eye pain or vision changes', 'Thickening of nose skin'],
        'visit_doctor': 'For prescription creams or laser therapy to reduce redness.'
    },
    'Tinea Ringworm': {
        'desc': 'A contagious fungal infection of the skin causing a ring-shaped rash.',
        'symptoms': 'A scaly ring-shaped area, itchiness, and a clear or scaly area inside the ring.',
        'action': 'Keep the area clean and completely dry. Wash bedsheets and towels daily.',
        'treatment': 'Antifungal creams (Clotrimazole, Miconazole) applied for 2 to 4 weeks.',
        'severity': 'Contagious',
        'red_flags': ['Spreading rapidly', 'Swelling/warmth', 'Hair loss in area'],
        'visit_doctor': 'If rash persists after 2 weeks of OTC treatment.'
    },
    'Unknown': {
        'desc': 'The model could not identify this condition with high confidence.',
        'symptoms': 'N/A',
        'action': 'Please upload a clearer image or consult a doctor.',
        'treatment': 'N/A',
        'severity': 'Unknown',
        'red_flags': [],
        'visit_doctor': 'Consult a dermatologist.'
    }
}

# --- 2. LOAD MODEL ---
try:
    # Ensure your model file is named 'skindisease.h5'
    model = load_model("skindisease.h5", compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template("base.html", prediction=None)

@app.route('/predict', methods=['POST'])
def upload():
    prediction_text = None
    confidence = "0"
    info = disease_info['Unknown']
    filename = None

    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return render_template("base.html", prediction=None)

            f = request.files['image']
            if f.filename == '':
                return render_template("base.html", prediction=None)

            # Save Image
            basepath = os.path.dirname(__file__)
            upload_folder = os.path.join(basepath, 'uploads')
            if not os.path.exists(upload_folder): os.makedirs(upload_folder)
            
            filename = secure_filename(f.filename)
            filepath = os.path.join(upload_folder, filename)
            f.save(filepath)

            # Preprocessing (EfficientNetV2 - 300x300)
            img = image.load_img(filepath, target_size=(300, 300))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Prediction
            preds = model.predict(x)[0]
            
            # 4 Classes (Alphabetical Order)
            classes = ['Acne', 'Eczema', 'Rosacea', 'Tinea Ringworm']
            
            pred_index = np.argmax(preds)
            confidence_val = preds[pred_index] * 100
            confidence = f"{confidence_val:.2f}"
            
            if pred_index < len(classes):
                prediction_text = classes[pred_index]
                if prediction_text in disease_info:
                    info = disease_info[prediction_text]
            else:
                prediction_text = "Unknown"

        except Exception as e:
            print(f"Server Error: {e}")
            prediction_text = "Error"

    return render_template("base.html", prediction=prediction_text, confidence=confidence, info=info, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'uploads'), filename)

if __name__ == '__main__':
    app.run(debug=True)