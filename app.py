from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

model = tf.keras.models.load_model("model/defect_detection_model.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    if file:
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)

        if prediction[0][0] > 0.5:
            result = "GOOD"
        else:
            result = "DEFECT"

        return render_template(
            "index.html",
            prediction=result,
            image_path=filepath
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)