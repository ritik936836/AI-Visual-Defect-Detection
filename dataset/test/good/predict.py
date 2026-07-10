import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("model/defect_detection_model.h5")

# Image path (test image)
img_path = "dataset/test/good/test.jpg"

# Load image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Prediction
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Prediction: GOOD")
else:
    print("Prediction: DEFECT")