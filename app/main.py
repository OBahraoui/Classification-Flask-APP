from flask import Flask, request, jsonify, render_template
from torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'No file found in request'})

        if not allowed_file(file.filename):
            return jsonify({'error': 'Format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            class_name = CLASSES[prediction.item()]  
            return jsonify({'prediction': prediction.item(), 'class_name': class_name})
        except Exception as e:
            return jsonify({'error': f'Error during prediction: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)

