document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    var formData = new FormData();
    var videoInput = document.getElementById('videoInput');
    
    if (videoInput.files.length > 0) {
        formData.append('video', videoInput.files[0]);

        fetch('/encode', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('response').innerHTML = 'Processing complete: ' + data.message;
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('response').innerHTML = 'An error occurred during processing.';
        });
    } else {
        alert('Please upload a video.');
    }
});

// Start webcam capture
document.getElementById('startWebcam').addEventListener('click', function() {
    const video = document.getElementById('webcam');

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(error => {
            console.error('Error accessing webcam: ', error);
        });

        fetch('/real_time_encode', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            document.getElementById('response').innerHTML = 'Real-time processing complete: ' + data.message;
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('response').innerHTML = 'An error occurred during real-time processing.';
        });
    } else {
        alert('Webcam not supported in your browser.');
    }
});
