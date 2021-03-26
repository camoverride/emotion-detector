# Emotion Detector

This is an application that takes a video of your face and returns information about it: gender, age, and real-time updates about your emotional state.

Video information is captured by the app and then the image is cropped to your face. Then this information is sent off to a model server with several models that perform different analyses (gender, age, emotion detection) and return the results to the client.

This repo includes a mini-XCEPTION model which was based off of [this paper](https://arxiv.org/pdf/1710.07557.pdf).

Related repos are:

- [serving](https://github.com/camoverride/tf_models_serving)
- [model development](https://github.com/camoverride/notebooks/blob/master/notebooks/Emotion_Detection_Transfer_Learning_v2.ipynb)


## Run it

Dependencies:

- `pip install -r requirements.txt`

Several options to run this locally:

- `export FLASK_APP=app.py && flask run`
- `gunicorn -b 127.0.0.1:8080 -k flask_sockets.worker app:app`
- `docker run -e PYTHONUNBUFFERED=0 -p 5000:5000 face-app`

To run the model server, which this application depends on, you need to visit [this repo](https://github.com/camoverride/tf_models_serving)


## Deploy it

Build and deploy a new version on GCP:

- `source deploy.sh`

These files are for Heroku deployment:

- `Aptfile`
- `Procfile`
- `runtime.txt`


## Tests

Test model functions that are called by the Flask server. These make calls to web API's to ensure they're still up:

`python -m unittest tests/face_utils_tests.py`


## Data pipeline

A face-identification model draws a bounding box around a face. This is a ML model (not deep learning) that I didn't develop. It requires OpenCV to run, which is a bit sluggish and hard to deploy. It's found in the [tf_models_serving repo](https://github.com/camoverride/tf_models_serving) as `haarcascade_frontalface_default.xml`

I then trained an emotion identification model by executing the code in [this notebook](https://github.com/camoverride/notebooks/blob/master/notebooks/Emotion_Detection_Transfer_Learning_v2.ipynb). This model is also saved in the [tf_models_serving repo](https://github.com/camoverride/tf_models_serving) as `XCEPTION.72.model`. If you want to run this notebook, install the requirements in `requirements-dev.txt`

`templates/index.html` imports the `socket.io.js` library for the client. It also imports three js files from `static/assets` which handle drawing the bounding-box on the image, streaming the video back to the client, and sending video data (frames sampled every few seconds) to the server for analysis.

`app.py` serves the front end HTML and also handles all the requests that get sent to the modeling servers.
