"""
This module serves the application (`index`) and has a route that generates
a bounding box that's displayed to the user (`bb_route`). Other routes are
responsible for calling models on a remote server and returning the results
to the client.
"""

from threading import Lock

from flask import Flask, render_template
from flask_socketio import SocketIO

from face_utils import decode_image, crop_face, get_model_pred


app = Flask(__name__)
socketio = SocketIO(app, async_mode=None)
thread_lock = Lock()

# Congifuration for model server.
MODEL_SERVER_URL = "localhost"
MODEL_SERVER_PORT = "8080"
EMOTION_MODEL_VERSION = "1"


@app.route("/")
def index():
    """
    Return the HTML for the front end
    """
    return render_template("index.html", async_mode=socketio.async_mode)


@socketio.on("analyze_emotion_request", namespace="/compute_emotion_route")
def emotion_route(message):
    """
    Route for getting results from the emotion model.
    """
    # Decode the image.
    image = decode_image(message["data"])

    # Crop the image to a single face.
    cropped_face, _ = crop_face(image)

    print("GETTING EMOTION")
    emotion_categories = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

    # Get the category of the face.
    emotion = get_model_pred(cropped_face, model_server_url=MODEL_SERVER_URL, model_server_port=MODEL_SERVER_PORT,
                model_version=EMOTION_MODEL_VERSION, model_name="emotion_model", categories=emotion_categories)

    # Send the category to the client.
    socketio.emit("emotion_model_response", {"data": emotion}, namespace="/compute_emotion_route")


@socketio.on("compute_bb_event", namespace="/compute_bb")
def bb_route(message):
    """
    This generates the boundingboxes that are used on the client side.
    """
    # If this fails, give default coords for the boundingbox so it won't be shown.
    try:
        # Decode the image.
        image = decode_image(message["data"])

        # Crop the image to a single face.
        _, coords = crop_face(image)

        data = {"bb_x": str(coords[0]), "bb_y": str(coords[1]), "bb_width": str(coords[2]), "bb_height": str(coords[3])}

    except IndexError:
        data = {"bb_x": str(0), "bb_y": str(0), "bb_width": str(0), "bb_height": str(0)}

    # Send the category to the client.
    socketio.emit("bb_response", data, namespace="/compute_bb")


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0")
