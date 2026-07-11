
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from predict import predict_digit

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict-digit', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(full_path)

        try:
            digit, confidence = predict_digit(full_path)


            if os.path.exists(full_path):
                os.remove(full_path)

       
            return jsonify({
                'status': 'success',
                'digit': int(digit),
                'confidence': float(confidence)
            }), 200

        except Exception as e:
            if os.path.exists(full_path):
                os.remove(full_path)
            return jsonify({'error': f'Model error: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG are allowed'}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
