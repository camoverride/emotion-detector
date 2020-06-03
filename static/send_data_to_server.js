// Capture video frames from client, send them to the server, and recieves the response.
$(document).ready(function() {
    var socket = io("/compute_emotion_route");

    // Sends video data to the server every 5 seconds.
    window.setInterval(function() {
        var capture = document.getElementById("video_canvas");
        socket.emit("compute_emotion_event", {data: capture.toDataURL("image/jpeg")});
    }, 5000);

    // Send response from server to client.
    socket.on("emotion_model_response", function(msg, cb) {
        $("#emotion_prediction").text(msg.data);
        if (cb)
            cb();
    });
});
