FROM tensorflow/serving
COPY models /models
ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh", "--rest_api_port=8080", "--model_config_file=/models/models.config"]
