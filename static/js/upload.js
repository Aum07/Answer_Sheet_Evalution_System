// Initialize Dropzone for file uploads
Dropzone.autoDiscover = false;

let referenceFile = null;
let studentFiles = [];
let progressInterval;

document.addEventListener('DOMContentLoaded', function() {
    // Reference file dropzone
    const referenceDropzone = new Dropzone("#referenceDropzone", {
        url: "/evaluate/",
        autoProcessQueue: false,
        maxFiles: 1,
        acceptedFiles: ".pdf",
        addRemoveLinks: true,
        dictDefaultMessage: "<i class='fas fa-file-pdf fa-2x mb-3'></i><br>Drop reference answer sheet here<br><span class='text-muted'>or click to upload</span>",
    });

    // Student files dropzone
    const studentDropzone = new Dropzone("#studentDropzone", {
        url: "/evaluate/",
        autoProcessQueue: false,
        maxFiles: 10,
        acceptedFiles: ".pdf",
        addRemoveLinks: true,
        dictDefaultMessage: "<i class='fas fa-file-pdf fa-2x mb-3'></i><br>Drop student answer sheets here<br><span class='text-muted'>or click to upload</span>",
    });

    // Handle reference file upload
    referenceDropzone.on("addedfile", function(file) {
        if (referenceDropzone.files.length > 1) {
            referenceDropzone.removeFile(referenceDropzone.files[0]);
        }
        referenceFile = file;
        updateSubmitButton();
        
        // Add animation to the file preview
        setTimeout(() => {
            file.previewElement.classList.add('animate__animated', 'animate__bounceIn');
        }, 100);
    });

    referenceDropzone.on("removedfile", function() {
        referenceFile = null;
        updateSubmitButton();
    });

    // Handle student files upload
    studentDropzone.on("addedfile", function(file) {
        studentFiles.push(file);
        updateSubmitButton();
        
        // Add animation to the file preview
        setTimeout(() => {
            file.previewElement.classList.add('animate__animated', 'animate__bounceIn');
        }, 100);
    });

    studentDropzone.on("removedfile", function(file) {
        const index = studentFiles.indexOf(file);
        if (index !== -1) {
            studentFiles.splice(index, 1);
        }
        updateSubmitButton();
    });

    // Enable/disable submit button based on file uploads
    function updateSubmitButton() {
        const submitBtn = document.getElementById('submitBtn');
        submitBtn.disabled = !referenceFile || studentFiles.length === 0;
        
        if (!referenceFile || studentFiles.length === 0) {
            submitBtn.classList.remove('btn-pulse');
        } else {
            submitBtn.classList.add('btn-pulse');
        }
    }

    // Handle form submission
    document.getElementById('submitBtn').addEventListener('click', async function() {
        const processingModal = new bootstrap.Modal(document.getElementById('processingModal'));
        processingModal.show();
        
        // Simulate progress bar
        let progress = 0;
        const progressBar = document.getElementById('progressBar');
        
        progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += (90 - progress) / 10;
                progressBar.style.width = `${progress}%`;
            }
        }, 500);

        const formData = new FormData();
        formData.append('reference_pdf', referenceFile);
        
        studentFiles.forEach(file => {
            formData.append('student_pdfs', file);
        });

        try {
            const response = await fetch('http://127.0.0.1:8000/evaluate/', {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);
            progressBar.style.width = '100%';

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            
            // Store the results in localStorage for the results page
            localStorage.setItem('evaluationResults', JSON.stringify(data));
            
            // Wait a moment to show 100% completion
            setTimeout(() => {
                // Redirect to results page
                window.location.href = '/static/results.html';
            }, 500);
        } catch (error) {
            console.error('Error:', error);
            clearInterval(progressInterval);
            processingModal.hide();
            
            const errorModal = new bootstrap.Modal(document.getElementById('errorModal'));
            document.getElementById('errorMessage').textContent = error.message;
            errorModal.show();
        }
    });
});
