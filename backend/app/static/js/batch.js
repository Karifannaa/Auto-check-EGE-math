// JavaScript for the batch processing page

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const tasksContainer = document.getElementById('tasks-container');
    const noTasksMessage = document.getElementById('no-tasks-message');
    const addTaskBtn = document.getElementById('add-task-btn');
    const startBatchBtn = document.getElementById('start-batch-btn');
    const batchResultsCard = document.getElementById('batch-results-card');
    const batchResultsContainer = document.getElementById('batch-results-container');
    const taskTemplate = document.getElementById('task-template');
    const batchTemperatureSlider = document.getElementById('batch-temperature');
    const batchTemperatureValue = document.getElementById('batch-temperature-value');

    // Bootstrap modal
    const batchLoadingModal = new bootstrap.Modal(document.getElementById('batch-loading-modal'));

    // Progress tracking
    const progressBar = document.getElementById('batch-progress-bar');
    const processedCount = document.getElementById('processed-count');
    const totalCount = document.getElementById('total-count');

    // Task counter
    let taskCounter = 0;
    let tasks = [];

    // Update temperature value display
    batchTemperatureSlider.addEventListener('input', function() {
        batchTemperatureValue.textContent = this.value;
    });

    // Add task button
    addTaskBtn.addEventListener('click', function() {
        addNewTask();
        updateStartButtonState();
    });

    // Start batch processing button
    startBatchBtn.addEventListener('click', function() {
        processBatch();
    });

    // Function to add a new task
    function addNewTask() {
        // Increment counter
        taskCounter++;

        // Clone template
        const taskNode = document.importNode(taskTemplate.content, true);
        const taskItem = taskNode.querySelector('.task-item');

        // Set task number
        taskItem.querySelector('.task-number').textContent = taskCounter;

        // Set up remove button
        const removeBtn = taskItem.querySelector('.remove-task-btn');
        removeBtn.addEventListener('click', function() {
            taskItem.remove();
            updateTasksList();
            updateStartButtonState();
        });

        // Set up image preview
        const studentImageInput = taskItem.querySelector('.task-student-images');
        const correctImageInput = taskItem.querySelector('.task-correct-image');
        const previewContainer = taskItem.querySelector('.task-images-preview');

        // Preview for student images
        studentImageInput.addEventListener('change', function() {
            // Clear previous previews
            previewContainer.innerHTML = '';

            if (this.files && this.files.length > 0) {
                // Add a label for student solutions
                const labelDiv = document.createElement('div');
                labelDiv.className = 'col-12';
                labelDiv.innerHTML = '<h6 class="text-muted">Решения учеников:</h6>';
                previewContainer.appendChild(labelDiv);

                // Add previews for each student solution
                for (let i = 0; i < this.files.length; i++) {
                    const file = this.files[i];
                    const reader = new FileReader();

                    reader.onload = function(e) {
                        const col = document.createElement('div');
                        col.className = 'col-4 col-md-3 col-lg-2';

                        const img = document.createElement('img');
                        img.src = e.target.result;
                        img.className = 'img-fluid';
                        img.alt = `Решение ученика ${i + 1}`;

                        col.appendChild(img);
                        previewContainer.appendChild(col);
                    };

                    reader.readAsDataURL(file);
                }
            }
        });

        // Preview for correct solution image
        correctImageInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const reader = new FileReader();

                reader.onload = function(e) {
                    // Check if there's already a correct solution preview
                    const existingLabel = previewContainer.querySelector('.correct-solution-label');
                    if (!existingLabel) {
                        // Add a label for correct solution
                        const labelDiv = document.createElement('div');
                        labelDiv.className = 'col-12 correct-solution-label';
                        labelDiv.innerHTML = '<h6 class="text-muted">Правильное решение:</h6>';

                        // Insert at the beginning of the container
                        if (previewContainer.firstChild) {
                            previewContainer.insertBefore(labelDiv, previewContainer.firstChild);
                        } else {
                            previewContainer.appendChild(labelDiv);
                        }

                        // Add the image
                        const col = document.createElement('div');
                        col.className = 'col-4 col-md-3 col-lg-2 correct-solution-preview';

                        const img = document.createElement('img');
                        img.src = e.target.result;
                        img.className = 'img-fluid';
                        img.alt = 'Правильное решение';

                        col.appendChild(img);
                        previewContainer.insertBefore(col, previewContainer.querySelector('.correct-solution-label').nextSibling);
                    } else {
                        // Update existing preview
                        const existingPreview = previewContainer.querySelector('.correct-solution-preview img');
                        if (existingPreview) {
                            existingPreview.src = e.target.result;
                        }
                    }
                };

                reader.readAsDataURL(this.files[0]);
            }
        });

        // Add task to container
        tasksContainer.appendChild(taskItem);

        // Hide "no tasks" message
        noTasksMessage.classList.add('d-none');

        // Update tasks list
        updateTasksList();
    }

    // Function to update the tasks list
    function updateTasksList() {
        const taskItems = tasksContainer.querySelectorAll('.task-item');

        if (taskItems.length === 0) {
            noTasksMessage.classList.remove('d-none');
        } else {
            noTasksMessage.classList.add('d-none');
        }

        // Update tasks array
        tasks = Array.from(taskItems);
    }

    // Function to update start button state
    function updateStartButtonState() {
        const taskItems = tasksContainer.querySelectorAll('.task-item');
        startBatchBtn.disabled = taskItems.length === 0;
    }

    // Function to process the batch
    async function processBatch() {
        // Validate all tasks
        const taskItems = tasksContainer.querySelectorAll('.task-item');
        let isValid = true;

        taskItems.forEach(item => {
            const taskType = item.querySelector('.task-type');
            const studentImages = item.querySelector('.task-student-images');

            if (!taskType.value || studentImages.files.length === 0) {
                isValid = false;
            }
        });

        if (!isValid) {
            alert('Пожалуйста, заполните все поля для всех задач.');
            return;
        }

        // Get batch settings
        const modelId = document.getElementById('batch-model-id').value;
        console.log('Using model ID:', modelId);
        const includeExamples = document.getElementById('batch-include-examples').checked;
        const promptVariant = document.getElementById('batch-prompt-variant').value;
        const temperature = document.getElementById('batch-temperature').value;

        if (!modelId) {
            alert('Пожалуйста, выберите модель для проверки.');
            return;
        }

        // Reset results
        batchResultsContainer.innerHTML = '';
        batchResultsCard.classList.add('d-none');

        // Set up progress tracking
        const totalTasks = taskItems.length;
        let processedTasks = 0;

        totalCount.textContent = totalTasks;
        processedCount.textContent = processedTasks;
        progressBar.style.width = '0%';

        // Show loading modal
        batchLoadingModal.show();

        // Process each task
        for (const taskItem of taskItems) {
            const taskType = taskItem.querySelector('.task-type').value;
            const taskDescription = taskItem.querySelector('.task-description').value;
            const taskImages = taskItem.querySelector('.task-student-images').files;
            const taskNumber = taskItem.querySelector('.task-number').textContent;

            // Get correct solution image if available
            const correctImage = taskItem.querySelector('.task-correct-image').files[0];

            // Process each student image for this task
            for (let i = 0; i < taskImages.length; i++) {
                const studentImage = taskImages[i];

                try {
                    // Create form data
                    const formData = new FormData();
                    formData.append('task_type', taskType);
                    formData.append('task_description', taskDescription);
                    formData.append('model_id', modelId);
                    formData.append('student_solution_image', studentImage);

                    // Add correct solution image if available
                    if (correctImage) {
                        formData.append('correct_solution_image', correctImage);
                    }

                    formData.append('include_examples', includeExamples);

                    if (promptVariant) {
                        formData.append('prompt_variant', promptVariant);
                    }

                    formData.append('temperature', temperature);

                    // Send request
                    const response = await fetch('/api/v1/solutions/evaluate', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }

                    const data = await response.json();

                    // Add result to results container
                    addResultToContainer(taskNumber, i + 1, taskType, taskDescription, data);

                } catch (error) {
                    console.error('Error processing task:', error);

                    // Add error to results container
                    addErrorToContainer(taskNumber, i + 1, taskType, error.message);
                }
            }

            // Update progress
            processedTasks++;
            processedCount.textContent = processedTasks;
            progressBar.style.width = `${(processedTasks / totalTasks) * 100}%`;
        }

        // Hide loading modal
        batchLoadingModal.hide();

        // Show results
        batchResultsCard.classList.remove('d-none');
        batchResultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // Function to add a result to the results container
    function addResultToContainer(taskNumber, imageNumber, taskType, taskDescription, data) {
        console.log('Received data for batch:', data);

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

        // Create result item
        const resultId = `result-${taskNumber}-${imageNumber}`;
        const resultItem = document.createElement('div');
        resultItem.className = 'accordion-item';
        resultItem.innerHTML = `
            <h2 class="accordion-header">
                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#${resultId}">
                    <div class="d-flex justify-content-between align-items-center w-100 me-3">
                        <span>Задача #${taskNumber} - Решение ${imageNumber}</span>
                        <span class="badge ${scoreClass} bg-primary">${score} балл(ов)</span>
                    </div>
                </button>
            </h2>
            <div id="${resultId}" class="accordion-collapse collapse">
                <div class="accordion-body">
                    ${taskDescription ? `
                    <div class="mb-3">
                        <h6>Комментарий к проверке:</h6>
                        <p>${taskDescription}</p>
                    </div>
                    ` : ''}

                    <div class="mb-3">
                        <span class="badge bg-primary">Модель: ${modelId}</span>
                        <span class="badge bg-secondary">Время: ${processingTimeInSeconds} сек</span>
                    </div>

                    <div class="result-analysis">
                        ${formattedResult}
                    </div>
                </div>
            </div>
        `;

        // Add to container
        batchResultsContainer.appendChild(resultItem);
    }

    // Function to add an error to the results container
    function addErrorToContainer(taskNumber, imageNumber, taskType, errorMessage) {
        const resultId = `result-error-${taskNumber}-${imageNumber}`;
        const resultItem = document.createElement('div');
        resultItem.className = 'accordion-item';
        resultItem.innerHTML = `
            <h2 class="accordion-header">
                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#${resultId}">
                    <div class="d-flex justify-content-between align-items-center w-100 me-3">
                        <span>Задача #${taskNumber} - Решение ${imageNumber}</span>
                        <span class="badge bg-danger">Ошибка</span>
                    </div>
                </button>
            </h2>
            <div id="${resultId}" class="accordion-collapse collapse">
                <div class="accordion-body">
                    <div class="alert alert-danger">
                        <h5>Ошибка при проверке решения</h5>
                        <p>${errorMessage || 'Произошла ошибка при обработке запроса.'}</p>
                    </div>
                </div>
            </div>
        `;

        // Add to container
        batchResultsContainer.appendChild(resultItem);
    }
});
