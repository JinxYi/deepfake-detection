import os
import cv2

face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml'))

def get_cropped_face(img_data, crop_factor=1.1):
    """
    Detects a face in an image and returns the cropped face as a NumPy array.

    :param img_data: The input image as a NumPy array (BGR format).
    :param crop_factor: Factor to extend the bounding box.
    :return: A NumPy array representing the cropped BGR face, or None if no face is found.
    """
    img = img_data
    if img is None or img.size == 0:
        print("Error: Empty image data provided.")
        return None

    # 1. Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        print("No face detected.")
        return None

    # 3. Crop the largest face (assuming the main subject is the largest face)
    (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    # 4. Apply a 'crop factor' for padding
    center_x = x + w // 2
    center_y = y + h // 2
    size = int(max(w, h) * crop_factor)

    # Calculate the coordinates for the padded square crop
    x_start = max(0, center_x - size // 2)
    y_start = max(0, center_y - size // 2)
    x_end = min(img.shape[1], center_x + size // 2)
    y_end = min(img.shape[0], center_y + size // 2)

    # Adjust start/end if clipping occurred to ensure a square crop
    final_size = min(x_end - x_start, y_end - y_start)

    # Final Crop
    cropped_face = img[y_start : y_start + final_size, x_start : x_start + final_size]

    return cropped_face