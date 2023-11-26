#!/bin/python3

from PIL import Image
import numpy as np
import flask
import flask_cors
import io
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image


app = flask.Flask(__name__)
flask_cors.CORS(app, resources={r'/*': {'origins': '*'}})


@app.route('/', methods=["GET"])
def index():
    return flask.make_response("Hello World!", 200)

@app.route('/predict', methods=["POST"])
def compress():
    file = flask.request.files['filedata'].read()
    try:
        cnn = tf.keras.models.load_model("model_medium_b32.h5")
        imageFile = Image.open(io.BytesIO(file))
        imageFile.save("tmp.jpg")
        img1 = image.load_img("tmp.jpg", target_size=(320, 180, 3))
        X = image.img_to_array(img1)
        X = np.expand_dims(X, axis=0)
        image_variable = np.vstack([X])
        p = cnn.predict(image_variable)
        os.remove("tmp.jpg")
        if p[0][0] == 0.0:
            return({"Prediction": "Healthy"})
        return({"Prediction": "ESCA"})
    except Exception as e:
        return flask.make_response(f"Error : {e}", 500)


def main():
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == '__main__':
    main()