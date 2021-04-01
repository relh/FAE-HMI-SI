
// Declare the 3 slider bars
$('#ex1').slider({
	tooltip: 'always',
	formatter: function(value) {
                date = yearputsfield[value].substring(0,15);
                date = date.slice(0, 4) + '/' + date.slice(4,6) + '/' + date.slice(6,8) + ' ' + date.slice(9,11) + ':' + date.slice(11,13);
		return 'Date: ' + date ;
	}
});

$('#ex2').slider({
	tooltip: 'always',
	formatter: function(value) {
                date = monthputsfield[value].substring(0,15);
                date = date.slice(0, 4) + '/' + date.slice(4,6) + '/' + date.slice(6,8) + ' ' + date.slice(9,11) + ':' + date.slice(11,13);
		return 'Date: ' + date ;
	}
});

$('#ex3').slider({
	tooltip: 'always',
	formatter: function(value) {
                date = yearputsfield[value].substring(0,15);
                date = date.slice(0, 4) + '/' + date.slice(4,6) + '/' + date.slice(6,8) + ' ' + date.slice(9,11) + ':' + date.slice(11,13);
		return 'Date: ' + date ;
	}
});

// Demo 2 -- The Zoom-ins Window -- Yearputs Initialization
var container = document.getElementById('img-container');
var containergt = document.getElementById('img-container-gt');

var pred_options = {
    width: 512, // required
    height: 512,
    img: "./full_prediction_thumb.jpg",
    // more options here
    //zoomWidth: 768,
    //scale: 5,
    offset: {
      vertical: 0,
      horizontal: 10
    },
    zoomPosition: 'right'
};
var gt_options = {
    width: 512, // required
    height: 512,
    // more options here
    img: "./inclination_pred.png",
    //zoomWidth: 768,
    //scale: 6,
    offset: {
      vertical: 0,
      horizontal: 10
    },
    zoomPosition: 'left'
};
target = document.querySelector('input[name="options"]:checked').id;
pred_options.img = 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[7] + '_pred_image.jpg';
gt_options.img = 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[7] + '_gt_image.jpg';
window.prediction_zoom = new ImageZoom(container, pred_options);
window.vfisv_zoom = new ImageZoom(containergt, gt_options);
target = document.querySelector('input[name="options"]:checked').id;

function preloadImage(url)
{
    var img = new Image();
    img.src = url;
}


//generate the shuffler

var targetNames = ["field","inclination","azimuth","vlos_mag","dop_width","eta_0","src_continuum","src_grad"];
var shuffleFlip = new Array();
for(var i=0; i< 8; i++){
    shuffleFlip.push(new Array());
    for(var j=0; j < yearputsfield.length; j++){
        shuffleFlip[i].push(j % 2);
    }
}

function setRevealBit(bit){
    var s = '';
    if(bit){ s = 'Left: VFISV; Right: Prediction'; }
    else { s = 'Left: Prediction; Right: VFISV'; }
    console.log(s)
    document.getElementById('quizReveal').onclick = function(){document.getElementById('quizReveal').innerHTML=s};
}
setRevealBit(0);

function radioClick(label, play) {
    //target = document.querySelector('input[name="options"]:checked').id;
    var input = label.getElementsByTagName('input')[0]
    var target = input.id;
    var targetIndex = targetNames.indexOf(target);

    document.getElementById('quizReveal').innerHTML = 'Reveal Answer';

	if (play == false) {
                var value = $('#ex1').slider('getValue');
		window['prediction_zoom'].kill();
		window['vfisv_zoom'].kill();

        var revealBit = shuffleFlip[targetIndex][value];
        setRevealBit(revealBit);
        if(revealBit){
            pred_options.img = 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value] + '_pred_image.jpg';
            gt_options.img = 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value] + '_gt_image.jpg';
        } else {
            gt_options.img = 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value] + '_pred_image.jpg';
            pred_options.img = 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value] + '_gt_image.jpg';
        }
		window.prediction_zoom = new ImageZoom(container, pred_options);
		window.vfisv_zoom = new ImageZoom(containergt, gt_options);
	} else{
                var value = $('#ex3').slider('getValue');
                var revealBit = shuffleFlip[targetIndex][value];
                setRevealBit(revealBit);
                if(revealBit){
                    $('#playimage').attr('src', 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value] + '_pred_image_thumb.jpg');
                    $('#playimagegt').attr('src', 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value] + '_gt_image_thumb.jpg');
                } else {
                    $('#playimagegt').attr('src', 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value] + '_pred_image_thumb.jpg');
                    $('#playimage').attr('src', 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value] + '_gt_image_thumb.jpg');
                }
        }

        var min = 0;
        var max = '3,000 Mx/cm<sup>2</sup>';
        if (target == 'field') {
                min = '&emsp;&emsp;&ensp;0 Mx/cm<sup>2</sup>'
        } else if (target == 'inclination' || target == 'azimuth') {
                min = '&emsp;&emsp;0 °';
                max = '180 °';
        } else if (target == 'vlos_mag'){
                min = '&ensp;-700,000 cm/s';
                max = '700,000 cm/s';
        } else if (target == 'eta_0') {
                min = '&emsp;&nbsp;0'
                max = '50';
        } else if (target == 'dop_width') {
                min = '&ensp;0 mÅ'
                max = '60 Å';
        } else if (target == 'src_continuum') {
                min = '&emsp;&emsp;&emsp;0 DN/s'
                max = '29,060 DN/s'; //.61;
        } else if (target == 'src_grad') {
                min = '&emsp;&emsp;&emsp;0 DN/s'
                max = '52,695 DN/s'; //.32;
        }

        if (play == false) {
                $('#cb1').attr('src', 'https://sunsite.s3.amazonaws.com/assets/' + target + '_colorbar.png');
                document.getElementById('min').innerHTML = min + '&nbsp;&nbsp;';
                document.getElementById('max').innerHTML = '&nbsp;&nbsp;' + max;
        } else {
                $('#cb3').attr('src', 'https://sunsite.s3.amazonaws.com/assets/' + target + '_colorbar.png');
                document.getElementById('min3').innerHTML = min + '&nbsp;&nbsp;';
                document.getElementById('max3').innerHTML = '&nbsp;&nbsp;' + max;
        }
}

function triggerOne(e, context) {
	var ne = new MouseEvent(e.type, e)
	container.dispatchEvent(ne, context);
}

$('#ex1').slider().on('slideStop', function(value) {
        window['prediction_zoom'].kill();
        window['vfisv_zoom'].kill();

        document.getElementById('quizReveal').innerHTML = 'Reveal Answer';

        var target = document.querySelector('input[name="options"]:checked').id;
        var targetIndex = targetNames.indexOf(target);

        var revealBit = shuffleFlip[targetIndex][value.value];
        setRevealBit(revealBit);
        if(revealBit){
            pred_options.img = 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value.value] + '_pred_image.jpg';
            gt_options.img = 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value.value] + '_gt_image.jpg';
        } else {
            gt_options.img = 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value.value] + '_pred_image.jpg';
            pred_options.img = 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value.value] + '_gt_image.jpg';
        }

        window.prediction_zoom = new ImageZoom(container, pred_options);
        window.vfisv_zoom = new ImageZoom(containergt, gt_options);

        var container_img = container.getElementsByTagName('img');
        var container_gt_img = containergt.getElementsByTagName('img');
        container_img[0].style.width = 768;
        container_img[0].style.height = 768;

        container_gt_img[0].style.width = 768;
        container_gt_img[0].style.height = 768;
});

// Demo 1 -- The Play Section
var containerplay = document.getElementById('img-container-play');
var containergtplay = document.getElementById('img-container-gt-play');

target = document.querySelector('input[name="options"]:checked').id;
preloadImage('https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[9] + '_pred_image_thumb.jpg');
preloadImage('https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[9] + '_gt_image_thumb.jpg');

var global_play = 0;
var interval;
function playMe() {
        if (global_play == 0) {
                global_play = 1;
                $('#control').attr('src', 'https://sunsite.s3.amazonaws.com/assets/pause.png');
                interval = setInterval(function() {
                    var value = $('#ex3').slider('getValue');
                    $('#ex3').slider('setValue', value+1, true, false);
                    $('#ex3').trigger({'type': 'slideStop', 'value': value+1});
                    global_play = 1;
                }, 2000);
        } else {
                $('#control').attr('src', 'https://sunsite.s3.amazonaws.com/assets/play.png');
                clearInterval(interval);
                global_play = 0;
        }
}

$('#ex3').slider().on('slideStop', function(value) {
        var target= document.querySelector('input[name="optionsPlay"]:checked').id;

        $('#playimage').attr('src', 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value.value] + '_pred_image_thumb.jpg');
        $('#playimagegt').attr('src', 'https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value.value] + '_gt_image_thumb.jpg');

        preloadImage('https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value.value+1] + '_pred_image_thumb.jpg');
        preloadImage('https://sunsite.s3.amazonaws.com/images/yearputs/' + target + '/' + yearputsfield[value.value+1] + '_gt_image_thumb.jpg');
});

// Demo 3 -- The Month Section
month_target = 'field';

function triggerOneMonth(e, context) {
	var ne = new MouseEvent(e.type, e)
	containermonth.dispatchEvent(ne, context);
}

$('#ex2').slider().on('slideStop', function(value) {
        //target = document.querySelector('input[name="optionsPlay"]:checked').id;
	var value = $('#ex2').slider('getValue');
        target = 'field';

        $('#monthimage').attr('src', 'https://sunsite.s3.amazonaws.com/images/monthputs/' + target + '/' + monthputsfield[value+12] + '_pred_image_thumb.jpg');
        $('#monthimagegt').attr('src', 'https://sunsite.s3.amazonaws.com/images/monthputs/' + target + '/' + monthputsfield[value+12] + '_gt_image_thumb.jpg');

	var bluebar = document.getElementById('bluebar');
//	new_left = 12.00 + value / 4.1;
	new_left = 6.00 + value / 7.2;
    if (new_left > 93.5) {
		new_left = 93.5;
	} 
	bluebar.style.left = new_left.toString() + '%';

        preloadImage('https://sunsite.s3.amazonaws.com/images/monthputs/' + target + '/' + monthputsfield[value+24] + '_pred_image_thumb.jpg');
        preloadImage('https://sunsite.s3.amazonaws.com/images/monthputs/' + target + '/' + monthputsfield[value+24] + '_gt_image_thumb.jpg');
});

function playMeTwo() {
        if (global_play == 0) {
                global_play = 1;
                $('#control2').attr('src', 'https://sunsite.s3.amazonaws.com/assets/pause.png');
                interval = setInterval(function() {
                    var value = $('#ex2').slider('getValue');
                    $('#ex2').slider('setValue', value+12, true, false);
                    $('#ex2').trigger({'type': 'slideStop', 'value': value+12});
                    global_play = 1;
                }, 2000);
        } else {
                $('#control2').attr('src', 'https://sunsite.s3.amazonaws.com/assets/play.png');
                clearInterval(interval);
                global_play = 0;
        }
}

/* Demo 4 -- Batchnorm bad */
var currentBNInd = 0;
function renderBN(){
    var base = 'http://fouheylab.eecs.umich.edu/~fouhey/hmiInversion/';
    for(var i = 0; i < 24; i++){
        var u = base+currentBNInd+"_5_x"+i+".jpg";
        document.getElementById("bn_"+i).src = u; 
    }
    document.getElementById("bn_crop").src = base+currentBNInd+"_5_y.jpg";
    document.getElementById("bn_full").src = base+currentBNInd+"_5_y_full.jpg";
    s = ""+currentBNInd;
    s = (s.length == 1) ? "0"+s : s
    document.getElementById("bn_date").innerHTML = s+":00";
}

function adjustBN(di){
    currentBNInd = (currentBNInd+24+di)%24;
    renderBN();
}


