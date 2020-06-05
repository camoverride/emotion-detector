// Continuously draw the video and the saved bb parameters to the canvas.

window.setInterval(function() {
    // Get video, canvas, and drawing context information.
    var canvas = document.getElementById("video_canvas");
    var video = document.getElementById("videoElement");
    var context = canvas.getContext("2d");

    // Set the canvas to the same dimensions as the webcam image.
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Get the boundingbox dimensions that have been secretly saved on the client side.
    var bb_x = document.getElementById("bb_x").innerText;
    var bb_y = document.getElementById("bb_y").innerText;
    var bb_h = document.getElementById("bb_height").innerText;
    var bb_w = document.getElementById("bb_width").innerText;

    // Draw the webcam image.
    context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    // Draw and customize the boundingbox.
    context.beginPath();
    context.rect(bb_x, bb_y, bb_h, bb_w);
    context.strokeStyle = "purple";
    context.lineWidth = "4";
    context.stroke();
}, 10);


// Every 500ms, compute the new bb parameters and save them on the client.
// This guarantees a continuous stream. However, there may be latency if
// the face detection algorithm is ever served from a seperate server.

var socket = io("/compute_bb");

window.setInterval(function() {
    var canvas = document.getElementById("video_canvas");
    socket.emit("compute_bb_event", {data: canvas.toDataURL("image/jpeg")});
}, 500);

// Replace the boundingbox parameters on the client side.
socket.on("bb_response", function(msg, cb) {
    $("#bb_x").text(msg.bb_x);
    $("#bb_y").text(msg.bb_y);
    $("#bb_height").text(msg.bb_height);
    $("#bb_width").text(msg.bb_width);
    
    if (cb)
        cb();
});
