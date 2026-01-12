// ========================================
// AI CLINICAL DECISION SUPPORT - PRO EDITION
// Enhanced with Animations & AI Visualization
// ========================================

let currentPatientData = null;
let currentTriageResult = null;
let xrayImageData = null;

// ========================================
// UTILITY FUNCTIONS
// ========================================

function showLoading(status = 'Processing patient data') {
    const overlay = document.getElementById('loadingOverlay');
    const statusText = document.getElementById('loadingStatus');
    overlay.classList.add('active');
    statusText.textContent = status;
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

function showError(message) {
    alert(`❌ Error: ${message}`);
}

function getSelectedSymptoms() {
    const checkboxes = document.querySelectorAll('input[name="symptoms"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

// ========================================
// ANIMATED COUNTER
// ========================================

function animateCounter(element, start, end, duration) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}

// ========================================
// TRIAGE FORM SUBMISSION
// ========================================

document.getElementById('triageForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    showLoading('AI analyzing patient vitals...');
    
    try {
        // Collect form data
        const formData = {
            age: parseInt(document.getElementById('age').value),
            symptoms: getSelectedSymptoms(),
            heart_rate: parseInt(document.getElementById('heart_rate').value),
            bp_systolic: parseInt(document.getElementById('bp_systolic').value),
            bp_diastolic: parseInt(document.getElementById('bp_diastolic').value),
            spo2: parseInt(document.getElementById('spo2').value),
            temperature: parseFloat(document.getElementById('temperature').value)
        };
        
        // Validate
        if (!formData.age || !formData.heart_rate || !formData.bp_systolic || 
            !formData.bp_diastolic || !formData.spo2 || !formData.temperature) {
            throw new Error('Please fill in all required fields');
        }
        
        currentPatientData = formData;
        
        // Simulate AI processing delay for effect
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Send to backend
        const response = await fetch('/api/triage', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error('Failed to calculate triage score');
        }
        
        const result = await response.json();
        currentTriageResult = result;
        
        // Display with animation
        displayTriageResults(result);
        
        // Scroll smoothly
        setTimeout(() => {
            document.getElementById('resultsContainer').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }, 300);
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
});

// ========================================
// DISPLAY TRIAGE RESULTS WITH ANIMATIONS
// ========================================

function displayTriageResults(result) {
    const resultsContainer = document.getElementById('resultsContainer');
    const urgencyBadge = document.getElementById('urgencyBadge');
    const urgencyLevel = document.getElementById('urgencyLevel');
    const riskScore = document.getElementById('riskScore');
    const confidenceBar = document.getElementById('confidenceBar');
    const actionBox = document.getElementById('actionBox');
    const actionText = document.getElementById('actionText');
    const aiSummary = document.getElementById('aiSummary');
    const riskFactorsList = document.getElementById('riskFactorsList');
    const criticalFlags = document.getElementById('criticalFlags');
    
    // Show results
    resultsContainer.style.display = 'block';
    
    // Animate urgency level
    urgencyLevel.textContent = result.urgency_level;
    
    // Remove previous classes
    urgencyBadge.className = 'urgency-badge';
    
    // Add urgency class
    switch(result.urgency_level) {
        case 'EMERGENCY':
            urgencyBadge.classList.add('urgency-emergency');
            break;
        case 'HIGH':
            urgencyBadge.classList.add('urgency-high');
            break;
        case 'MEDIUM':
            urgencyBadge.classList.add('urgency-medium');
            break;
        case 'LOW':
            urgencyBadge.classList.add('urgency-low');
            break;
    }
    
    // Animate risk score counter
    animateCounter(riskScore, 0, result.risk_score, 1500);
    
    // Animate confidence bar
    confidenceBar.style.width = `${result.risk_score}%`;
    
    // Update action box
    actionBox.style.borderColor = result.urgency_color;
    actionText.textContent = result.action;
    
    // Display AI summary
    if (result.ai_summary) {
        aiSummary.textContent = result.ai_summary;
    } else {
        aiSummary.textContent = result.clinical_note;
    }
    
    // Display risk factors with staggered animation
    riskFactorsList.innerHTML = '';
    if (result.risk_factors && result.risk_factors.length > 0) {
        result.risk_factors.forEach((factor, index) => {
            setTimeout(() => {
                const li = document.createElement('li');
                li.textContent = factor;
                li.style.animationDelay = `${index * 0.1}s`;
                riskFactorsList.appendChild(li);
            }, index * 100);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = 'No significant risk factors identified';
        li.style.background = '#f0fdf4';
        li.style.borderColor = '#16a34a';
        riskFactorsList.appendChild(li);
    }
    
    // Display critical flags
    criticalFlags.innerHTML = '';
    if (result.critical_flags && result.critical_flags.length > 0) {
        result.critical_flags.forEach(flag => {
            const span = document.createElement('span');
            span.className = 'critical-flag';
            span.textContent = flag;
            criticalFlags.appendChild(span);
        });
    }
}

// ========================================
// FILE UPLOAD HANDLING
// ========================================

document.getElementById('xrayFile').addEventListener('change', function(e) {
    const fileName = document.getElementById('fileName');
    if (e.target.files.length > 0) {
        fileName.textContent = e.target.files[0].name;
        
        // Preview image
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onload = function(event) {
            xrayImageData = event.target.result;
        };
        reader.readAsDataURL(file);
    } else {
        fileName.textContent = 'No file selected';
    }
});

// ========================================
// X-RAY ANALYSIS FORM
// ========================================

document.getElementById('xrayForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    if (!currentPatientData) {
        showError('Please complete the patient assessment first');
        return;
    }
    
    const fileInput = document.getElementById('xrayFile');
    if (!fileInput.files.length) {
        showError('Please select an X-ray image');
        return;
    }
    
    showLoading('AI analyzing chest X-ray...');
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('xray', fileInput.files[0]);
        
        // Add patient data
        formData.append('age', currentPatientData.age);
        formData.append('symptoms', currentPatientData.symptoms.join(','));
        formData.append('heart_rate', currentPatientData.heart_rate);
        formData.append('bp_systolic', currentPatientData.bp_systolic);
        formData.append('bp_diastolic', currentPatientData.bp_diastolic);
        formData.append('spo2', currentPatientData.spo2);
        formData.append('temperature', currentPatientData.temperature);
        
        // Simulate AI processing
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Send to backend
        const response = await fetch('/api/analyze-xray', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Failed to analyze X-ray');
        }
        
        const result = await response.json();
        
        // Update triage results
        currentTriageResult = result;
        displayTriageResults(result);
        displayXrayResults(result.xray_analysis);
        
        // Scroll to X-ray results
        setTimeout(() => {
            document.getElementById('xrayResults').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'nearest' 
            });
        }, 300);
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
});

// ========================================
// DISPLAY X-RAY RESULTS WITH HEATMAP
// ========================================

function displayXrayResults(xrayAnalysis) {
    const xrayResults = document.getElementById('xrayResults');
    const xrayFinding = document.getElementById('xrayFinding');
    const xrayConfidence = document.getElementById('xrayConfidence');
    const xrayConfidenceFill = document.getElementById('xrayConfidenceFill');
    const xrayExplanation = document.getElementById('xrayExplanation');
    const canvas = document.getElementById('xrayCanvas');
    const ctx = canvas.getContext('2d');
    
    xrayResults.style.display = 'block';
    
    // Display finding
    if (xrayAnalysis.detected) {
        xrayFinding.innerHTML = `
            <span style="color: #dc2626; font-weight: 700;">
                ⚠️ ${xrayAnalysis.message}
            </span>
        `;
        xrayConfidenceFill.style.width = `${xrayAnalysis.confidence}%`;
        xrayConfidenceFill.style.background = 'linear-gradient(90deg, #dc2626, #ea580c)';
    } else {
        xrayFinding.innerHTML = `
            <span style="color: #16a34a; font-weight: 700;">
                ✓ ${xrayAnalysis.message}
            </span>
        `;
        xrayConfidenceFill.style.width = `${xrayAnalysis.confidence}%`;
        xrayConfidenceFill.style.background = 'linear-gradient(90deg, #16a34a, #10b981)';
    }
    
    xrayConfidence.textContent = `${xrayAnalysis.confidence.toFixed(1)}%`;
    
    // Display AI explanation
    if (xrayAnalysis.ai_explanation) {
        xrayExplanation.textContent = xrayAnalysis.ai_explanation;
    }
    
    // Draw X-ray image on canvas
    if (xrayImageData) {
        const img = new Image();
        img.onload = function() {
            // Draw image
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            // Draw heatmap overlay if pneumonia detected
            if (xrayAnalysis.detected && xrayAnalysis.heatmap) {
                drawHeatmap(ctx, xrayAnalysis.heatmap, canvas.width, canvas.height);
            }
        };
        img.src = xrayImageData;
    }
}

// ========================================
// HEATMAP VISUALIZATION
// ========================================

function drawHeatmap(ctx, heatmapData, width, height) {
    if (!heatmapData || !heatmapData.regions) return;
    
    // Draw affected regions
    heatmapData.regions.forEach(region => {
        const x = (region.x / 224) * width;
        const y = (region.y / 224) * height;
        const size = (region.size / 224) * width;
        
        // Create gradient
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, size);
        gradient.addColorStop(0, 'rgba(220, 38, 38, 0.6)');
        gradient.addColorStop(0.5, 'rgba(220, 38, 38, 0.3)');
        gradient.addColorStop(1, 'rgba(220, 38, 38, 0)');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x - size, y - size, size * 2, size * 2);
        
        // Draw border
        ctx.strokeStyle = 'rgba(220, 38, 38, 0.8)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.stroke();
    });
    
    // Add legend
    ctx.fillStyle = 'rgba(220, 38, 38, 0.8)';
    ctx.font = '12px sans-serif';
    ctx.fillText('● AI-Detected Regions', 10, height - 10);
}

// ========================================
// REAL-TIME VALIDATION
// ========================================

const inputs = document.querySelectorAll('input[type="number"]');
inputs.forEach(input => {
    input.addEventListener('blur', function() {
        validateInput(this);
    });
    
    input.addEventListener('input', function() {
        // Remove error styling on input
        this.style.borderColor = '';
    });
});

function validateInput(input) {
    const value = parseFloat(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    
    if (isNaN(value) || value < min || value > max) {
        input.style.borderColor = '#dc2626';
        input.style.boxShadow = '0 0 0 3px rgba(220, 38, 38, 0.1)';
    } else {
        input.style.borderColor = '#16a34a';
        input.style.boxShadow = '0 0 0 3px rgba(22, 163, 74, 0.1)';
    }
}

// ========================================
// KEYBOARD SHORTCUTS
// ========================================

document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const triageForm = document.getElementById('triageForm');
        if (document.activeElement.form === triageForm) {
            e.preventDefault();
            triageForm.dispatchEvent(new Event('submit'));
        }
    }
});

// ========================================
// DEMO QUICK FILL (for testing)
// ========================================

// Uncomment for quick demo during presentation
window.quickFillEmergency = function() {
    document.getElementById('age').value = 68;
    document.getElementById('heart_rate').value = 115;
    document.getElementById('bp_systolic').value = 165;
    document.getElementById('bp_diastolic').value = 98;
    document.getElementById('spo2').value = 88;
    document.getElementById('temperature').value = 101.2;
    document.querySelector('input[value="chest_pain"]').checked = true;
    document.querySelector('input[value="difficulty_breathing"]').checked = true;
    console.log('✓ Emergency scenario loaded');
};

window.quickFillLow = function() {
    document.getElementById('age').value = 32;
    document.getElementById('heart_rate').value = 72;
    document.getElementById('bp_systolic').value = 118;
    document.getElementById('bp_diastolic').value = 76;
    document.getElementById('spo2').value = 98;
    document.getElementById('temperature').value = 98.6;
    document.querySelector('input[value="fatigue"]').checked = true;
    console.log('✓ Low urgency scenario loaded');
};

// ========================================
// INITIALIZATION
// ========================================

console.log('🏥 AI Clinical Decision Support System - Pro Edition');
console.log('✓ Enhanced with AI visualization and animations');
console.log('Quick fill commands available:');
console.log('  - quickFillEmergency()');
console.log('  - quickFillLow()');

// Add smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});