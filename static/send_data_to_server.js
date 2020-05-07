// Capture video frames from client, send them to the server, and recieves the response.
$(document).ready(function() {
    var socket = io("/emotion_detector");

    // Sends video data to the server every 5 seconds.
    window.setInterval(function() {
        var cap = document.getElementById("video_canvas");
        socket.emit("my_event", {data: cap.toDataURL("image/jpeg")});
    }, 5000);

    // Send response from server to client.
    socket.on("emotion_response", function(msg, cb) {
        $("#emotion_prediction").text(msg.data);
        if (cb)
            cb();
    });
});

// Write the video data
function capture() {
    var canvas = document.getElementById("video_canvas");
    var video = document.getElementById("videoElement");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    canvas.toBlob() = (blob) => {
    const img = new Image();
    img.src = window.URL.createObjectUrl(blob);
    };
}

window.setInterval(function() {
    capture();
}, 5000);

// Make sure the video canvas stays hidden.
document.getElementById("video_canvas").style.visibility = "hidden";
