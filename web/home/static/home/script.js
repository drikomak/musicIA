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