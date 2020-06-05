// Capture video frames from client, send them to the server, and handle the response.
$(document).ready(function() {
    var socket = io("/compute_emotion_route");

    // Send the request to analyze the emotion.
    $("form#analyze_button").submit(function(event) {
        var capture = document.getElementById("video_canvas");

        var info = capture.toDataURL("image/jpeg");
        socket.emit("analyze_emotion_request", {data: info});

        return false;
    });

    // Handle the response sent from the server.
    socket.on("emotion_model_response", function(msg, cb) {
        $("#emotion_prediction").text(msg.data);
        if (cb)
            cb();
    });
});
