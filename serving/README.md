# tf_models_serving

This is where serving files live. Check the latest version of the docker image, because it changes frequently!


## Quickstart

Download the image:

- `docker pull camoverride/face-models:v0.3`

Run it locally:

- `docker run -t --rm -p 8080:8080 camoverride/face-models:v0.3`

Send it some test data:

- `pip install -r serving-requirements.txt`
- `python -m unittest tests/model_server_tests.py`


## Build it

To add additional models, take a `SavedModel` and copy it to the `models` directory. Then edit the `models/models.config` file accordingly.

Make sure that models are created with the `serving_default` signature, or they won't be able to be served.

Then build a new image (choose an appropriate version number and change your username):

- `docker build -t camoverride/face-models:v0.4 .`

Note: port `8500` is open to public. `8501` isn't.


## Deploy it

- [Build docs](https://www.tensorflow.org/tfx/serving/docker)
- [Deploy docs](https://www.tensorflow.org/tfx/serving/serving_kubernetes)

The file `build-deploy.sh` will help you get the Docker image deployed to a Kube cluster on GCP. `emotion_serve.yaml` is useful too but will be deprecated soon.


## Odds and Ends

The notebooks in `create_servables` help you get models into the required `SavedModel` format.
