# Emotion Detector

This repo creates an app that serves returns face-analysis information to a client. This repo includes a mini-XCEPTION model which was based off of [this paper](https://arxiv.org/pdf/1710.07557.pdf).

Related repos are:

- [serving](https://github.com/camoverride/tf_models_serving)
- [model development](https://github.com/camoverride/models)

See the app online [here]().

## Run it

Several options to run this locally:

- `gunicorn -b 127.0.0.1:8080 -k flask_sockets.worker app:app`
- `export FLASK_APP=app.py && flask run`
- `docker run -e PYTHONUNBUFFERED=0 -p 5000:5000 face-app`

Build and deploy a new version on GCP:

- `source deploy.sh`

These files are for Heroku deployment:

- `Aptfile`
- `Procfile`
- `runtime.txt`

## Tests

Test model functions that are called by the Flask server. These make calls to web API's to ensure they're still up:

`python -m unittest tests/face_utils_tests.py`
