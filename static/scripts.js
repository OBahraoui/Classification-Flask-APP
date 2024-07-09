document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();
    let fileInput = document.getElementById('file-upload');
    let file = fileInput.files[0];
    
    let formData = new FormData();
    formData.append('file', file);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        let resultDiv = document.getElementById('result');
        if (data.error) {
            resultDiv.textContent = `Error: ${data.error}`;
        } else {
            resultDiv.textContent = `Prediction: ${data.class_name}`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        let resultDiv = document.getElementById('result');
        resultDiv.textContent = 'Error during prediction';
    });
});
