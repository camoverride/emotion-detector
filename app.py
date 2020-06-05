from threading import Lock

from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit

from face_utils import decode_image, crop_face, get_emotions


app = Flask(__name__)
socketio = SocketIO(app, async_mode=None)
thread_lock = Lock() # TODO: not sure if this is needed.


@app.route("/")
def index():
    return render_template("index.html", async_mode=socketio.async_mode)


@socketio.on("compute_emotion_event", namespace="/compute_emotion_route")
def emotion_route(message):
    """
    Route for getting results from the emotion model.
    """
    # Decode the image.
    image = decode_image(message["data"])

    # Crop the image to a single face.
    cropped_face, _ = crop_face(image)

    # Get the category of the face.
    emotion = get_emotions(cropped_face)

    # Send the category to the client.
    socketio.emit("emotion_model_response", {"data": emotion}, namespace="/compute_emotion_route")


@socketio.on("compute_bb_event", namespace="/compute_bb")
def bb_route(message):
    """
    This generates the boundingboxes that are used on the client side.
    """
    # If this fails, give default coords for the boundingbox so it won't be shown.
    # TODO: handle exception better. Don't catch all.
    try:
        # Decode the image.
        image = decode_image(message["data"])

        # Crop the image to a single face.
        _, coords = crop_face(image)

        data = {"bb_x": str(coords[0]), "bb_y": str(coords[1]), "bb_width": str(coords[2]), "bb_height": str(coords[3])}

    except:
        data = {"bb_x": str(0), "bb_y": str(0), "bb_width": str(0), "bb_height": str(0)}

    # Send the category to the client.
    socketio.emit("bb_response", data, namespace="/compute_bb")


if __name__ == "__main__":
    # TODO: run with Gunicorn instead.
    socketio.run(app, debug=True, host="0.0.0.0")
