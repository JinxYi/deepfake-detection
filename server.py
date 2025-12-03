import os
import sys
import config
import numpy as np
import cv2
import torch
from PIL import Image
import logging
from flask_cors import CORS, cross_origin


# import local dependencies
from preprocessing.face_crop import get_cropped_face
from src.model_registry import MODEL_REGISTRRY
from src.models.resnet import ResNetClassifier
from src.transforms.fusion import get_fused_transform
from src.transforms.frequency import get_transforms
from flask import Flask, request, jsonify


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

proj_dir = os.path.dirname(os.path.abspath('__file__'))

model_id = "dwt_rgb_resnet18"
model_checkpoint_dir = f"{proj_dir}/{config.CHECKPOINTS_DIR}/{model_id}"
checkpoint = "lightning_logs/version_0/checkpoints/epoch=13-step=177520.ckpt"
CHECKPOINT_PATH = f"{model_checkpoint_dir}/{checkpoint}"

seed = config.SEED
torch.manual_seed(seed)
np.random.seed(seed)

models = {}
transforms = {}

for model_id, spec in MODEL_REGISTRRY.items():
    ckpt_path = f"{proj_dir}/{config.CHECKPOINTS_DIR}/{model_id}/{spec['checkpoint']}"
    model = ResNetClassifier.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        in_channels=spec.get("in_channels", 3),
        freeze_features=False,
        weights_only=False,
    )
    model.eval()
    models[model_id] = model
    transforms[model_id] = get_transforms(mode=spec["transforms_mode"], image_size=224)

app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.before_request
def log_request_details():
    if request.path == '/favicon.ico': # Optional: Skip logging for favicon requests
        return

    log_message = f"Incoming Request: Method={request.method}, URL={request.url}"
    
    # Add headers (optional)
    log_message += f", Headers={request.headers}" 
    
    # Add request body (optional, handle different content types)
    if request.is_json:
        log_message += f", JSON Body={request.json}"
    elif request.form:
        log_message += f", Form Data={request.form}"
    elif request.data:
        log_message += f", Raw Data={request.data}"

    logging.info(log_message)

@app.route('/predict', methods=['POST'])
def detect_deepfake():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']
    # if not request.data:
    #     return jsonify({"error": "No image file provided."}), 400

    # image_bytes = request.data
    model_id = request.form.get('model_id', 'dwt_rgb_resnet18')
    try:
        new_image_pil = Image.open(image_file).convert('RGB')
        
        new_image_np_rgb = np.array(new_image_pil)
        new_image_np_bgr = cv2.cvtColor(new_image_np_rgb, cv2.COLOR_RGB2BGR)

        # # Load a new image as PIL Image
        # new_image_pil = Image.open(proj_dir + "/samples/fake_4.jpg")

        # Process the image: get cropped face as NumPy array (BGR)
        cropped_face_np_bgr = get_cropped_face(new_image_np_bgr)

        # Check if face was detected
        if cropped_face_np_bgr is None:
            # Handle the case where no face was found
            raise ValueError("No face detected in the image.")

        # Convert the cropped NumPy array (BGR) back to PIL Image (RGB) for torchvision transforms
        cropped_face_pil_rgb = Image.fromarray(cv2.cvtColor(cropped_face_np_bgr, cv2.COLOR_BGR2RGB))

        # Apply the transformations
        transformed_input = transforms[model_id]['test'](cropped_face_pil_rgb)

        # Add a batch dimension and move to the appropriate device
        input_tensor = transformed_input.unsqueeze(0).to(device)


        with torch.no_grad(): # Disable gradient calculations for faster inference
            output = models[model_id](input_tensor)

            # Apply sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(output)

            # Get the predicted class (0 for Real, 1 for Fake, depending on your setup)
            prob = probabilities.item()
            pred_class = (probabilities > 0.5).long().item()

        return jsonify({
            "status": "success",
            "prediction_class": pred_class,
            "prediction_label": "Fake" if pred_class == 1 else "Real",
            "probability_fake": prob,
            "logits": output.item()
        })
    except Exception as e:
        # Log the detailed error on the server side
        print(f"An error occurred during inference: {e}")
        # Return a generic error to the client
        return jsonify({"error": "Internal server error during prediction.", "details": str(e)}), 500

@app.route('/')
def home():
    return "Deepfake Detection Server is running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)