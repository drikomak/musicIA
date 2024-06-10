function addAudioWave(id, height) {
    var audio = document.getElementById("audio-"+id)
    
    const wavesurfer = WaveSurfer.create({
        container: "#spectrum-"+id,
        height: height,
        waveColor: "#999",
        progressColor: "#FFF",
        cursorWidth: 0,
        barWidth: 4,
        barGap: 5,
        barRadius: 10,
        barHeight: 0.8,
        barAlign: "",
        minPxPerSec: 1,
        fillParent: true,
        media: audio,
    })
}

function addControls(id) {
    var audio = $("#audio-"+id)
    var audioDOM = document.getElementById("audio-"+id)
    var button = $("#playbutton-"+id)

    $(button).on("click", function () {
        if (audioDOM.paused) {
            audioDOM.play()
        } else {
            audioDOM.pause()
        }
    })

    $(audio).on("pause", function() {
        $(button).removeClass("dds__icon--pause")
        $(button).addClass("dds__icon--play-cir")
    })

    $(audio).on("play", function() {
        $(button).removeClass("dds__icon--play-cir")
        $(button).addClass("dds__icon--pause")
    })
}

function lazyLoad() {
    $("input").prop("disabled", true)
    $("#generator").addClass("opa-min")
    $("#htmxresult").addClass("anim")
    $("#htmxresult").css("display","block")
    $("#caption-result").html("Résulat en cours de chargement...")

    document.getElementById("audio-result").pause()

    $("#audio-result").off("play")
    $("#audio-result").off("pause")
    $("#playbutton-result").off("click")
}

function endLoad() {
    $("#generator").removeClass("opa-min")
    $("input").prop("disabled",false)
    $("#htmxresult").removeClass("anim")
}

function getValue(param) {
    var p = document.getElementById(param)
    return p.value.toString()
}

function setupCamera() {   
    let width = 640; // We will scale the photo width to this
    let height = 480; // This will be computed based on the input stream
    
    let streaming = false;

    let video = document.getElementById("video");
    let canvas = document.getElementById("canvas");
    
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then((stream) => {
            video.srcObject = stream;
            video.play();
    })
    
    video.addEventListener("canplay", (ev) => {
        if (!streaming) {
            video.setAttribute("width", width);
            video.setAttribute("height", height);
            canvas.setAttribute("width", width);
            canvas.setAttribute("height", height);
            streaming = true;
        }
    }, false);
    
    clearphoto();
}

function clearphoto() {
    const context = canvas.getContext("2d");
    context.fillStyle = "#AAA";
    context.fillRect(0, 0, canvas.width, canvas.height);

    const data = canvas.toDataURL("image/png");
    return data
}

function takepicture() {
    const context = canvas.getContext("2d");
    width = 640; height = 480;          // NE DEVRAIT PAS ETRE EN DUR !! CODE SELON MON PC PORTABLE, A REVOIR!!!!!!!!!!!!!!!!
    if (width && height) {
        canvas.width = width;         
        canvas.height = height;
        context.drawImage(video, 0, 0, width, height);

        const data = canvas.toDataURL("image/png");
        // photo.setAttribute("src", data);

        console.log(data);

        return data
    } else {
        return clearphoto();
    }
}

$(document).ready(function () {
    $(".navig").on("click", function () {
        if (window.innerHeight > 760) {
            let nb = $(this).data("index")
            let el = document.getElementById('sn')
            let h = el.clientHeight.toFixed(0)
            console.log(nb, h, window.innerHeight);
            
            el.scroll({top:h*nb,behavior:"smooth"})
        } else {
            let nb = $(this).data("index")
            document.getElementById("sn-"+nb).scrollIntoView()
        }
        
    })
})

function closeCam() {
    $("#camera-overlay").empty().css("display","none").removeClass("htmx-swapping")
}

function validatePic() {
    closeCam()
    $("#emotion").children()
}