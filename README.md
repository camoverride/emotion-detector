# Emotion Detector

This is an application that takes a video of your face and returns information about it: gender, age, and real-time updates about your emotional state.

Video information is captured by the app and then the image is cropped to your face. Then this information is sent off to a model server with several models that perform different analyses (gender, age, emotion detection) and return the results to the client.

This repo includes a mini-XCEPTION model which was based off of [this paper](https://arxiv.org/pdf/1710.07557.pdf).

For more information on the model check out [this notebook](https://github.com/camoverride/notebooks/blob/master/notebooks/Emotion_Detection_Transfer_Learning_v2.ipynb)

![picture of the application](https://github.com/camoverride/emotion-detector/blob/master/app_example.png?raw=true)

## Run it

Dependencies:

- `pip install -r requirements.txt`

Several options to run this locally:

- `export FLASK_APP=app.py && flask run`

Or:

- `gunicorn -b 127.0.0.1:8080 -k flask_sockets.worker app:app`

Or pull it from Docker hub (this will automatically pull the image if it doesn't exist locally):

- `docker run -e PYTHONUNBUFFERED=0 -p 5000:5000 camoverride/face-app:v1.11`

Or build it yourself from Docker (choose the right verson #):

- `docker build -t camoverride/face-app:v1.11 .`

Then start the model server:

- `docker pull camoverride/face-models:v0.3`
- `docker run -t --rm -p 8080:8080 camoverride/face-models:v0.3`

If you want to build the model server, visit the `serving` directory and check out the README.


## Deploy it

Build and deploy a new version on GCP (modify this file with your own credentials):

- `source deploy.sh`

These files are for Heroku deployment:

- `Aptfile`
- `Procfile`
- `runtime.txt`


## Tests

Test model functions that are called by the Flask server. These make calls to web API's to ensure they're still up. Run the model server before trying this, and make sure the version numbers are correct:

`python -m unittest tests/face_utils_tests.py`


## Data pipeline

A face-identification model draws a bounding box around a face. This is a ML model (not deep learning) that I didn't develop. It requires OpenCV to run, which is a bit sluggish and hard to deploy. It's found in the [tf_models_serving repo](https://github.com/camoverride/tf_models_serving) as `haarcascade_frontalface_default.xml`

I then trained an emotion identification model by executing the code in [this notebook](https://github.com/camoverride/notebooks/blob/master/notebooks/Emotion_Detection_Transfer_Learning_v2.ipynb). This model is also saved in the [tf_models_serving repo](https://github.com/camoverride/tf_models_serving) as `XCEPTION.72.model`. If you want to run this notebook, install the requirements in `requirements-dev.txt`

`templates/index.html` imports the `socket.io.js` library for the client. It also imports three js files from `static/assets` which handle drawing the bounding-box on the image, streaming the video back to the client, and sending video data (frames sampled every few seconds) to the server for analysis.

`app.py` serves the front end HTML and also handles all the requests that get sent to the modeling servers.


## Run it online

If you want to run this online, you should have the front-end and model (both docker images) deployed on different servers. The requirements for the model server are in `serving/serving-requirements.txt`.
