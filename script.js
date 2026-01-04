// Insurance Cost Prediction - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    const submitText = document.getElementById('submitText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    
    // BMI Calculator
    const bmiCalculatorBtn = document.getElementById('bmiCalculatorBtn');
    const bmiCalculator = document.getElementById('bmiCalculator');
    const calculateBmiBtn = document.getElementById('calculateBmi');
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    const bmiInput = document.getElementById('bmi');
    
    // Toggle BMI calculator
    if (bmiCalculatorBtn) {
        bmiCalculatorBtn.addEventListener('click', function() {
            if (bmiCalculator.style.display === 'none') {
                bmiCalculator.style.display = 'block';
            } else {
                bmiCalculator.style.display = 'none';
            }
        });
    }
    
    // Calculate BMI
    if (calculateBmiBtn) {
        calculateBmiBtn.addEventListener('click', function() {
            const height = parseFloat(heightInput.value);
            const weight = parseFloat(weightInput.value);
            
            if (height && weight && height > 0 && weight > 0) {
                // Convert height from cm to meters and calculate BMI
                const heightInMeters = height / 100;
                const bmi = weight / (heightInMeters * heightInMeters);
                bmiInput.value = bmi.toFixed(1);
                
                // Close calculator
                bmiCalculator.style.display = 'none';
                heightInput.value = '';
                weightInput.value = '';
            } else {
                alert('Please enter valid height (cm) and weight (kg) values.');
            }
        });
    }
    
    // Form validation
    if (form) {
        const inputs = form.querySelectorAll('input[required], select[required]');
        
        // Real-time validation
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                validateField(input);
            });
            
            input.addEventListener('input', function() {
                clearFieldError(input);
            });
        });
        
        // Form submission
        form.addEventListener('submit', function(e) {
            let isValid = true;
            
            // Validate all fields
            inputs.forEach(input => {
                if (!validateField(input)) {
                    isValid = false;
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                return false;
            }
            
            // Show loading state
            if (submitBtn) {
                submitBtn.disabled = true;
                if (submitText) submitText.textContent = 'Predicting...';
                if (loadingSpinner) loadingSpinner.style.display = 'inline-block';
            }
        });
    }
    
    // Validate individual field
    function validateField(field) {
        const value = field.value.trim();
        const fieldName = field.name;
        let isValid = true;
        let errorMessage = '';
        
        clearFieldError(field);
        
        // Required field check
        if (field.hasAttribute('required') && !value) {
            errorMessage = `${getFieldLabel(fieldName)} is required`;
            isValid = false;
        }
        
        // Specific validations
        if (value) {
            switch(fieldName) {
                case 'age':
                    const age = parseInt(value);
                    if (isNaN(age) || age < 18 || age > 100) {
                        errorMessage = 'Age must be between 18 and 100';
                        isValid = false;
                    }
                    break;
                    
                case 'bmi':
                    const bmi = parseFloat(value);
                    if (isNaN(bmi) || bmi < 15 || bmi > 50) {
                        errorMessage = 'BMI must be between 15.0 and 50.0';
                        isValid = false;
                    }
                    break;
                    
                case 'children':
                    const children = parseInt(value);
                    if (isNaN(children) || children < 0 || children > 5) {
                        errorMessage = 'Children must be between 0 and 5';
                        isValid = false;
                    }
                    break;
                    
                case 'sex':
                    if (value !== 'male' && value !== 'female') {
                        errorMessage = 'Please select a valid option';
                        isValid = false;
                    }
                    break;
                    
                case 'smoker':
                    if (value !== 'yes' && value !== 'no') {
                        errorMessage = 'Please select a valid option';
                        isValid = false;
                    }
                    break;
                    
                case 'region':
                    const validRegions = ['northeast', 'northwest', 'southeast', 'southwest'];
                    if (!validRegions.includes(value)) {
                        errorMessage = 'Please select a valid region';
                        isValid = false;
                    }
                    break;
            }
        }
        
        if (!isValid) {
            showFieldError(field, errorMessage);
        }
        
        return isValid;
    }
    
    // Show field error
    function showFieldError(field, message) {
        field.style.borderColor = '#ef4444';
        
        // Remove existing error message
        const existingError = field.parentElement.querySelector('.field-error');
        if (existingError) {
            existingError.remove();
        }
        
        // Add error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error';
        errorDiv.style.color = '#ef4444';
        errorDiv.style.fontSize = '0.875rem';
        errorDiv.style.marginTop = '5px';
        errorDiv.textContent = message;
        field.parentElement.appendChild(errorDiv);
    }
    
    // Clear field error
    function clearFieldError(field) {
        field.style.borderColor = '';
        const errorDiv = field.parentElement.querySelector('.field-error');
        if (errorDiv) {
            errorDiv.remove();
        }
    }
    
    // Get field label
    function getFieldLabel(fieldName) {
        const labels = {
            'age': 'Age',
            'sex': 'Sex',
            'bmi': 'BMI',
            'children': 'Children',
            'smoker': 'Smoker',
            'region': 'Region'
        };
        return labels[fieldName] || fieldName;
    }
});

