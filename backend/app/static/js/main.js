// Main JavaScript for the single solution evaluation page

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const solutionForm = document.getElementById('solution-form');
    const studentImageInput = document.getElementById('student-solution-image');
    const correctImageInput = document.getElementById('correct-solution-image');
    const imagePreviewCard = document.getElementById('image-preview-card');
    const imagePreview = document.getElementById('image-preview');
    const resultCard = document.getElementById('result-card');
    const resultContent = document.getElementById('result-content');
    const temperatureSlider = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperature-value');

    // Bootstrap modal
    const loadingModal = new bootstrap.Modal(document.getElementById('loading-modal'));

    // Update temperature value display
    temperatureSlider.addEventListener('input', function() {
        temperatureValue.textContent = this.value;
    });

    // Image preview for student solution
    studentImageInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const reader = new FileReader();

            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreviewCard.classList.remove('d-none');
                imagePreviewCard.querySelector('.card-title').textContent = 'Предпросмотр решения ученика';
            };

            reader.readAsDataURL(this.files[0]);
        }
    });

    // Image preview for correct solution
    correctImageInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const reader = new FileReader();

            reader.onload = function(e) {
                // If student image is not yet uploaded, show this image
                if (!studentImageInput.files || !studentImageInput.files[0]) {
                    imagePreview.src = e.target.result;
                    imagePreviewCard.classList.remove('d-none');
                    imagePreviewCard.querySelector('.card-title').textContent = 'Предпросмотр правильного решения';
                }
            };

            reader.readAsDataURL(this.files[0]);
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

        if (!taskType.value || !modelId.value || !studentImageInput.files.length) {
            solutionForm.reportValidity();
            return;
        }

        // Show loading modal
        loadingModal.show();

        // Create FormData manually to ensure correct field names
        const formData = new FormData();
        formData.append('task_type', taskType.value);
        formData.append('task_description', document.getElementById('task-description').value || '');
        formData.append('model_id', modelId.value);
        formData.append('student_solution_image', studentImageInput.files[0]);

        // Add correct solution image if available
        if (correctImageInput.files.length > 0) {
            formData.append('correct_solution_image', correctImageInput.files[0]);
        }

        // Add other form fields
        const includeExamples = document.getElementById('include-examples');
        if (includeExamples && includeExamples.checked) {
            formData.append('include_examples', 'true');
        } else {
            formData.append('include_examples', 'false');
        }

        const promptVariant = document.getElementById('prompt-variant');
        if (promptVariant && promptVariant.value) {
            formData.append('prompt_variant', promptVariant.value);
        }

        const temperature = document.getElementById('temperature');
        if (temperature) {
            formData.append('temperature', temperature.value);
        }

        const maxTokens = document.getElementById('max-tokens');
        if (maxTokens && maxTokens.value) {
            formData.append('max_tokens', maxTokens.value);
        }

        try {
            // Send request to API
            const response = await fetch('/api/v1/solutions/evaluate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                // Try to get error details from response
                let errorMessage = `HTTP error! Status: ${response.status}`;
                try {
                    const errorData = await response.json();
                    if (errorData.detail) {
                        errorMessage = `Error: ${errorData.detail}`;
                    }
                } catch (e) {
                    // If we can't parse the JSON, just use the status
                    console.error('Could not parse error response:', e);
                }
                throw new Error(errorMessage);
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
        console.log('Received data:', data);

        // Extract score from the result
        let score = 'Не определено';
        let scoreClass = '';
        let explanation = '';

        // Check if result is an object with the expected structure
        if (data.result && typeof data.result === 'object') {
            // Use the score directly from the result object
            if (typeof data.result.score === 'number') {
                score = data.result.score.toString();
                scoreClass = `score-${score}`;
            }

            // Get the explanation text
            if (typeof data.result.explanation === 'string') {
                explanation = data.result.explanation;
            }
        } else if (typeof data.result === 'string') {
            // Fallback for string result (old format)
            const scoreMatch = data.result.match(/Оценка:\s*(\d+)\s*балл/i);
            if (scoreMatch && scoreMatch[1]) {
                score = scoreMatch[1];
                scoreClass = `score-${score}`;
            }
            explanation = data.result;
        }

        // Format the result with Markdown-like formatting
        let formattedResult = explanation
            .replace(/#{3}\s*(.*?)$/gm, '<h5>$1</h5>')  // ### headers
            .replace(/#{2}\s*(.*?)$/gm, '<h4>$1</h4>')  // ## headers
            .replace(/#{1}\s*(.*?)$/gm, '<h3>$1</h3>')  // # headers
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>')  // italic
            .split('\n').join('<br>');  // line breaks

        // Get model ID and processing time
        const modelId = data.model_id || data.model || 'Неизвестно';
        const processingTime = data.result?.evaluation_time || data.processing_time || 0;
        // Time is already in seconds, no need to divide by 1000
        const processingTimeInSeconds = processingTime.toFixed(2);

        // Create result HTML
        resultContent.innerHTML = `
            <div class="mb-3">
                <span class="badge bg-primary">Модель: ${modelId}</span>
                <span class="badge bg-secondary">Время: ${processingTimeInSeconds} сек</span>
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
