from threading import Lock

from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit

from face_utils import decode_image, crop_face, get_emotions


app = Flask(__name__)
socketio = SocketIO(app, async_mode=None)
thread_lock = Lock()


@app.route("/")
def index():
    return render_template("index.html", async_mode=socketio.async_mode)


@socketio.on("my_event", namespace="/emotion_detector")
def test_message(message):
    """
    The client-side javascript generates an event every 2 seconds, sending a video frame to
    this route. The frame is processed and an emotion is returned to the client.
    """
    # Decode the image.
    image = decode_image(message["data"])

    # Crop the image to a single face.
    cropped_face, _ = crop_face(image)

    # Get the category of the face.
    emotion = get_emotions(cropped_face)

    # Send the category to the client.
    socketio.emit("emotion_response", {"data": emotion}, namespace="/emotion_detector")


@socketio.on("compute_bb_event", namespace="/compute_bb")
def bb_route(message):
    """
    This generates the boundingboxes that are used on the client side.
    """
    # Decode the image.
    image = decode_image(message["data"])

    # Crop the image to a single face.
    _, coords = crop_face(image)

    # coords = [random.randrange(10, 100), random.randrange(10, 100), random.randrange(100, 400), random.randrange(100, 400) ]

    data = {"bb_x": str(coords[0]), "bb_y": str(coords[1]), "bb_width": str(coords[2]), "bb_height": str(coords[3])}

    # Send the category to the client.
    socketio.emit("bb_response", data, namespace="/compute_bb")


if __name__ == "__main__":
    # TODO: run with Gunicorn instead.
    socketio.run(app, debug=True, host="0.0.0.0")
