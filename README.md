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

## TO-DO List

CURRENT STATE: with firewall exception, this successfully does a videoplayback and streams dummy predictions. Major next steps are setting up HTTPS so that the video playback always works and then setting up model servers.

### Modeling

- [ ] Locally test model performance.
- [ ] Add confusion matrix to tests.
- [ ] Check softmax on output layer.
- [ ] Add smoothing for video streaming.
- [ ] Xception >>> mini-Xception.
- [ ] face_detection_model (pre-compuled haarcascade) really sucks and can't handle faces at odd angles
- [ ] re-train model for low-light conditions, generate new data (tf built-in utils)

### App Development

- [ ] Don't resize entire webcam image. Instead, sample a square from the middle of the image. This will prevent distortion from resizing!
- [ ] Currently, I have a server deployed with docker/kubernetes on GCP. It is HTTP, not HTTPS, so the webcam doesn't work (Chrome blocks `getMediaDevices` from HTTP connections). To get around this, I put an exception for my app in chrome's firewall: https://stackoverflow.com/questions/34197653/getusermedia-in-chrome-47-without-using-https
- [ ] create actual tests
- [ ] Replace flask_sockets with the real deal: https://cloud.google.com/appengine/docs/flexible/python/using-websockets-and-session-affinity
- [ ] Currently, requests are sent from `setInterval` on the client -- this is very insecure. Find a better way of doing this...
- [ ] Dockerize this app so that it runs on gunicorn, not the Flask development server.
- [ ] Create better build file, `deploy.sh`
- [ ] Set up tensorflow servers for face-detection and emotion-detection models. Find out how to make this secure so only my app can send requests to the server: https://towardsdatascience.com/securing-ml-services-on-the-web-69408e8554d0 and https://cloud.google.com/docs/authentication/production?hl=en_US
- [ ] If predictions come in too quickly, things get "clogged up" as the stack of images waiting to be processed grows. Figure out how to dump the stack, instead prefering latency (in other words, wait for one response is returned before asking for another -- sync, not async!!!) -- in other words, Set models to update only after previous HTTP request is processed and returned.
- [ ] Read this: https://towardsdatascience.com/securing-ml-services-on-the-web-69408e8554d0
- [ ] one route should crop. Then the cropped face should be passed to every other route, where there is one route per model. That way each model can update without needing to wait for the others. Also: update some models (i.e. gender) less frequently.
- [ ] Make sure that websockets aren't sending the same reply to every client... each client should access independently...
- [ ] Make websockets timeout after 5 minutes.
