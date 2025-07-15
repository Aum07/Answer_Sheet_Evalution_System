document.addEventListener('DOMContentLoaded', function() {
    // Get evaluation results from localStorage
    const resultsData = JSON.parse(localStorage.getItem('evaluationResults'));
    
    if (!resultsData || !resultsData.evaluations || resultsData.evaluations.length === 0) {
        alert('No evaluation results found. Please upload files first.');
        window.location.href = '/static/upload.html';
        return;
    }
    
    // Store the request ID for downloading reports
    const requestId = resultsData.request_id;
    
    // Populate summary view
    populateSummaryView(resultsData);
    
    // Initialize DataTable
    const dataTable = $('#studentsTable').DataTable({
        responsive: true,
        order: [[3, 'desc']], // Sort by percentage by default
        language: {
            search: "_INPUT_",
            searchPlaceholder: "Search students..."
        }
    });
    
    // Handle "Back to Summary" button click
    document.getElementById('backToSummary').addEventListener('click', function() {
        document.getElementById('detailView').style.display = 'none';
        document.getElementById('summaryView').style.display = 'block';
    });
    
    // Handle "Download All" button click
    document.getElementById('downloadAllBtn').addEventListener('click', function() {
        window.location.href = `/download-multiple/${requestId}`;
    });
    
    // Handle "Export CSV" button click
    document.getElementById('exportCsvBtn').addEventListener('click', function() {
        exportToCSV(resultsData);
    });
});

function populateSummaryView(data) {
    const evaluations = data.evaluations;
    
    // Calculate summary statistics
    const totalStudents = evaluations.length;
    const totalScores = evaluations.map(e => e.percentage);
    const avgScore = totalScores.reduce((a, b) => a + b, 0) / totalStudents;
    
    let highestScore = 0;
    let highestScoreStudent = '';
    let lowestScore = 100;
    let lowestScoreStudent = '';
    
    evaluations.forEach(evaluation => {
        if (evaluation.percentage > highestScore) {
            highestScore = evaluation.percentage;
            highestScoreStudent = evaluation.student_name;
        }
        if (evaluation.percentage < lowestScore) {
            lowestScore = evaluation.percentage;
            lowestScoreStudent = evaluation.student_name;
        }
    });
    
    // Update summary cards
    document.getElementById('totalStudents').textContent = totalStudents;
    document.getElementById('averageScore').textContent = `${avgScore.toFixed(2)}%`;
    document.getElementById('highestScore').textContent = `${highestScore.toFixed(2)}%`;
    document.getElementById('highestScoreStudent').textContent = highestScoreStudent;
    document.getElementById('lowestScore').textContent = `${lowestScore.toFixed(2)}%`;
    document.getElementById('lowestScoreStudent').textContent = lowestScoreStudent;
    
    // Populate students table
    const tableBody = document.getElementById('studentsTableBody');
    tableBody.innerHTML = '';
    
    evaluations.forEach((evaluation, index) => {
        const row = document.createElement('tr');
        
        const scoreClass = getScoreColorClass(evaluation.percentage);
        
        row.innerHTML = `
            <td>${evaluation.student_name}</td>
            <td>${evaluation.total_score}</td>
            <td>${evaluation.max_possible_score}</td>
            <td><span class="badge ${scoreClass}">${evaluation.percentage.toFixed(2)}%</span></td>
            <td>
                <button class="btn btn-sm btn-primary view-details-btn" data-index="${index}">
                    <i class="fas fa-eye me-1"></i> View Details
                </button>
                <a href="/download/${data.request_id}?student=${encodeURIComponent(evaluation.student_name)}" 
                   class="btn btn-sm btn-outline-secondary">
                   <i class="fas fa-download me-1"></i> Download
                </a>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
    
    // Add event listeners to the newly created buttons
    document.querySelectorAll('.view-details-btn').forEach(button => {
        button.addEventListener('click', function() {
            const studentIndex = this.getAttribute('data-index');
            showStudentDetails(evaluations[studentIndex], studentIndex, data.request_id);
        });
    });
    
    // Create charts
    createScoreDistributionChart(evaluations);
    createQuestionPerformanceChart(evaluations);
}

function getScoreColorClass(score) {
    if (score >= 80) return 'bg-success text-white';
    if (score >= 60) return 'bg-info text-white';
    if (score >= 40) return 'bg-warning text-dark';
    return 'bg-danger text-white';
}

function showStudentDetails(student, index, requestId) {
    // Hide summary view and show detail view
    document.getElementById('summaryView').style.display = 'none';
    const detailView = document.getElementById('detailView');
    detailView.style.display = 'block';
    
    // Update student details
    document.getElementById('studentDetailName').textContent = student.student_name;
    document.getElementById('detailTotalScore').textContent = `${student.total_score}/${student.max_possible_score}`;
    document.getElementById('detailPercentage').textContent = `${student.percentage.toFixed(2)}%`;
    
    // Set download button data
    document.getElementById('downloadReportBtn').setAttribute('data-student', student.student_name);
    document.getElementById('downloadReportBtn').onclick = function() {
        window.location.href = `/download/${requestId}?student=${encodeURIComponent(student.student_name)}`;
    };
    
    // Create student score chart
    createStudentScoreChart(student);
    
    // Populate questions
    const questionsContainer = document.getElementById('questionsContainer');
    questionsContainer.innerHTML = '';
    
    student.results.forEach((result, qIndex) => {
        const scoreClass = getScoreColorClass(result.Predicted_Score * 10);
        
        const questionCard = document.createElement('div');
        questionCard.className = 'card mb-3 shadow-sm';
        
        questionCard.innerHTML = `
            <div class="card-header d-flex justify-content-between align-items-center bg-light">
                <h5 class="mb-0">Question ${qIndex + 1}</h5>
                <span class="badge ${scoreClass}">${result.Predicted_Score}/10</span>
            </div>
            <div class="card-body">
                <h6 class="fw-bold">Question:</h6>
                <p>${result.Question}</p>
                
                <div class="row mt-4">
                    <div class="col-md-6">
                        <h6 class="fw-bold">Student Answer:</h6>
                        <div class="student-answer mb-3">
                            <p>${result.Student_Answer || 'No answer provided'}</p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h6 class="fw-bold">Feedback:</h6>
                        <div class="feedback">
                            <p>${result.Feedback}</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        questionsContainer.appendChild(questionCard);
    });
}

function createScoreDistributionChart(evaluations) {
    const ctx = document.getElementById('scoreDistributionChart').getContext('2d');
    
    // Create score ranges
    const ranges = ['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'];
    const counts = [0, 0, 0, 0, 0];
    
    evaluations.forEach(evaluation => {
        const score = evaluation.percentage;
        if (score <= 20) counts[0]++;
        else if (score <= 40) counts[1]++;
        else if (score <= 60) counts[2]++;
        else if (score <= 80) counts[3]++;
        else counts[4]++;
    });
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ranges,
            datasets: [{
                label: 'Number of Students',
                data: counts,
                backgroundColor: [
                    'rgba(220, 53, 69, 0.7)',
                    'rgba(255, 193, 7, 0.7)',
                    'rgba(255, 205, 86, 0.7)',
                    'rgba(23, 162, 184, 0.7)',
                    'rgba(40, 167, 69, 0.7)'
                ],
                borderColor: [
                    'rgb(220, 53, 69)',
                    'rgb(255, 193, 7)',
                    'rgb(255, 205, 86)',
                    'rgb(23, 162, 184)',
                    'rgb(40, 167, 69)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            },
            animation: false
        }
    });
}

function createQuestionPerformanceChart(evaluations) {
    if (evaluations.length === 0 || !evaluations[0].results || evaluations[0].results.length === 0) {
        return;
    }
    
    const ctx = document.getElementById('questionPerformanceChart').getContext('2d');
    const questionCount = evaluations[0].results.length;
    const labels = Array.from({length: questionCount}, (_, i) => `Q${i+1}`);
    
    // Calculate average score for each question
    const averageScores = [];
    
    for (let q = 0; q < questionCount; q++) {
        let totalScore = 0;
        let validCount = 0;
        
        evaluations.forEach(evaluation => {
            if (evaluation.results[q]) {
                totalScore += evaluation.results[q].Predicted_Score;
                validCount++;
            }
        });
        
        averageScores.push(validCount > 0 ? totalScore / validCount : 0);
    }
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Average Score (out of 10)',
                data: averageScores,
                backgroundColor: 'rgba(54, 162, 235, 0.7)',
                borderColor: 'rgb(54, 162, 235)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 10
                }
            },
            animation: false
        }
    });
}

function createStudentScoreChart(student) {
    const ctx = document.getElementById('studentScoreChart').getContext('2d');
    
    const labels = student.results.map((_, index) => `Q${index+1}`);
    const scores = student.results.map(result => result.Predicted_Score);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Score (out of 10)',
                data: scores,
                backgroundColor: scores.map(score => {
                    if (score >= 8) return 'rgba(40, 167, 69, 0.7)';
                    if (score >= 5) return 'rgba(255, 193, 7, 0.7)';
                    return 'rgba(220, 53, 69, 0.7)';
                }),
                borderColor: scores.map(score => {
                    if (score >= 8) return 'rgb(40, 167, 69)';
                    if (score >= 5) return 'rgb(255, 193, 7)';
                    return 'rgb(220, 53, 69)';
                }),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 10
                }
            },
            animation: false
        }
    });
}

function exportToCSV(data) {
    const evaluations = data.evaluations;
    let csvContent = "Student Name,Total Score,Max Score,Percentage\n";
    
    evaluations.forEach(evaluation => {
        csvContent += `"${evaluation.student_name}",${evaluation.total_score},${evaluation.max_possible_score},${evaluation.percentage.toFixed(2)}\n`;
    });
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "evaluation_results.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
