// Main JavaScript for the single solution evaluation page

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const solutionForm = document.getElementById('solution-form');
    const imageInput = document.getElementById('solution-image');
    const imagePreviewCard = document.getElementById('image-preview-card');
    const imagePreview = document.getElementById('image-preview');
    const resultCard = document.getElementById('result-card');
    const resultContent = document.getElementById('result-content');
    const temperatureSlider = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperature-value');
    const submitBtn = document.getElementById('submit-btn');

    // Bootstrap modal
    const loadingModal = new bootstrap.Modal(document.getElementById('loading-modal'));

    // Update temperature value display
    temperatureSlider.addEventListener('input', function() {
        temperatureValue.textContent = this.value;
    });

    // Image preview
    imageInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const reader = new FileReader();

            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreviewCard.classList.remove('d-none');
            };

            reader.readAsDataURL(this.files[0]);
        } else {
            imagePreviewCard.classList.add('d-none');
        }
    });

    // Form submission
    solutionForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Hide previous results
        resultCard.classList.add('d-none');

        // Validate form
        // Make sure required fields are filled
        const taskType = document.getElementById('task-type');
        const modelId = document.getElementById('model-id');
        const solutionImage = document.getElementById('solution-image');

        if (!taskType.value || !modelId.value || !solutionImage.files.length) {
            solutionForm.reportValidity();
            return;
        }

        // Show loading modal
        loadingModal.show();

        // Create FormData
        const formData = new FormData(solutionForm);

        try {
            // Send request to API
            const response = await fetch('/api/v1/solutions/evaluate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();

            // Display result
            displayResult(data);

        } catch (error) {
            console.error('Error:', error);
            resultContent.innerHTML = `
                <div class="alert alert-danger">
                    <h5>Ошибка при проверке решения</h5>
                    <p>${error.message || 'Произошла ошибка при обработке запроса.'}</p>
                </div>
            `;
            resultCard.classList.remove('d-none');
        } finally {
            // Hide loading modal
            loadingModal.hide();
        }
    });

    // Function to display the evaluation result
    function displayResult(data) {
        // Extract score from the result
        let score = 'Не определено';
        let scoreClass = '';

        const scoreMatch = data.result.match(/Оценка:\s*(\d+)\s*балл/i);
        if (scoreMatch && scoreMatch[1]) {
            score = scoreMatch[1];
            scoreClass = `score-${score}`;
        }

        // Format the result with Markdown-like formatting
        let formattedResult = data.result
            .replace(/#{3}\s*(.*?)$/gm, '<h5>$1</h5>')  // ### headers
            .replace(/#{2}\s*(.*?)$/gm, '<h4>$1</h4>')  // ## headers
            .replace(/#{1}\s*(.*?)$/gm, '<h3>$1</h3>')  // # headers
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>')  // italic
            .split('\n').join('<br>');  // line breaks

        // Create result HTML
        resultContent.innerHTML = `
            <div class="mb-3">
                <span class="badge bg-primary">Модель: ${data.model}</span>
                <span class="badge bg-secondary">Время: ${(data.processing_time / 1000).toFixed(2)} сек</span>
            </div>

            <div class="alert alert-info">
                <div class="d-flex justify-content-between align-items-center">
                    <h4 class="mb-0">Итоговая оценка</h4>
                    <span class="result-score ${scoreClass}">${score} балл(ов)</span>
                </div>
            </div>

            <div class="result-analysis">
                ${formattedResult}
            </div>
        `;

        // Show result card
        resultCard.classList.remove('d-none');

        // Scroll to result
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});
