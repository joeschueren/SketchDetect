var canvas = document.getElementById("Canvas");
var ctx = canvas.getContext("2d");
ctx.fillStyle = "#FFFFFF";
ctx.fillRect(0, 0, canvas.width, canvas.height);

var isDrawing = false;

// Add event listeners for both touch and mouse events
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);
canvas.addEventListener("touchstart", startDrawing);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", stopDrawing);

function startDrawing(e) {
    e.preventDefault(); // prevent default touch behavior
    isDrawing = true;
    // Use the appropriate event type based on the device
    var touchEvent = (e.type === 'touchstart') ? e.touches[0] : e;
    draw(touchEvent);
}

function draw(e) {
    e.preventDefault(); // prevent default touch behavior
    if (!isDrawing) return;
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.strokeStyle = "#000000";
    // Use the appropriate coordinates based on the device
    var x = (e.type === 'touchmove' || e.type === 'touchstart') ? e.touches[0].clientX - canvas.offsetLeft : e.clientX - canvas.offsetLeft;
    var y = (e.type === 'touchmove' || e.type === 'touchstart') ? e.touches[0].clientY - canvas.offsetTop + window.pageYOffset : e.clientY - canvas.offsetTop + window.pageYOffset;
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
    var dataURL = canvas.toDataURL();
    document.getElementById("image_data").value = dataURL;
}

var clear = document.getElementById("clear");
clear.addEventListener("click", clearAll)

function clearAll()
{
    ctx.fillStyle = "#ffffff"; // set background color to white
    ctx.fillRect(0, 0, canvas.width, canvas.height); // fill the entire canvas with the new background color
    ctx.beginPath(); // reset the drawing path
    document.getElementById("image_data").value = '';
}