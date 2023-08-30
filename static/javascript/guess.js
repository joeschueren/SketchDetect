const img = document.getElementById("response")

const toFolder = "Neural-Network/static/Images/";

const response = document.getElementById("response")

const index = response.getAttribute("response")

const imagePaths = ["umbrella", "house", "apple", "envelope", "star", "heart", "lightning",
                    "cloud", "spoon", "balloon", "mug", "mountains", "fish", "bowtie",
                    "ladder", "icecream", "bow", "moon", "smiley"];

show = 0
setInterval(() => {
    if(show === 3){
        show = 1
    }
    else{
        show++;
    }
    img.src = {{ url_for('static', filename='"+ imagePaths[index] + show +".png) }}
}, 3000)