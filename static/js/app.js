/**
 * IPL Win Predictor - Main Application Logic for Flask Backend
 * Handles form interactions, API calls, and result display
 */

class IPLPredictor {
    constructor() {
        // Team data with official colors and information
        this.teams = [
            'Sunrisers Hyderabad',
            'Mumbai Indians', 
            'Royal Challengers Bengaluru',
            'Kolkata Knight Riders',
            'Punjab Kings',
            'Chennai Super Kings',
            'Rajasthan Royals',
            'Delhi Capitals',
            'Lucknow Super Giants',
            'Gujarat Titans'
        ];

        // Team colors for visual representation
        this.teamColors = {
            'Sunrisers Hyderabad': '#FF822A',
            'Mumbai Indians': '#004B8D',
            'Royal Challengers Bengaluru': '#D41C24',
            'Kolkata Knight Riders': '#3A225D',
            'Punjab Kings': '#DD1F2D',
            'Chennai Super Kings': '#FDB913',
            'Rajasthan Royals': '#004C93',
            'Delhi Capitals': '#00008B',
            'Lucknow Super Giants': '#00AEEF',
            'Gujarat Titans': '#1B2133'
        };

        // IPL venues/cities
        this.cities = [
            'Ahmedabad', 'Bangalore', 'Bengaluru', 'Bloemfontein', 'Cape Town',
            'Centurion', 'Chandigarh', 'Chennai', 'Cuttack', 'Delhi',
            'Dharamsala', 'Durban', 'East London', 'Hyderabad', 'Indore',
            'Jaipur', 'Johannesburg', 'Kimberley', 'Kolkata', 'Lucknow',
            'Mohali', 'Mumbai', 'Nagpur', 'Port Elizabeth', 'Pune',
            'Raipur', 'Ranchi', 'Sharjah', 'Visakhapatnam'
        ];

        // API endpoint - Updated for local Flask server
        this.apiEndpoint = 'http://127.0.0.1:5000/predict';

        // DOM elements
        this.elements = {
            battingTeamSelect: document.getElementById('batting-team'),
            bowlingTeamSelect: document.getElementById('bowling-team'),
            citySelect: document.getElementById('city'),
            targetInput: document.getElementById('target'),
            scoreInput: document.getElementById('score'),
            oversInput: document.getElementById('overs'),
            wicketsInput: document.getElementById('wickets'),
            predictBtn: document.getElementById('predict-btn'),
            resultContainer: document.getElementById('result-container'),
            errorMessage: document.getElementById('error-message'),
            loadingOverlay: document.getElementById('loading-overlay')
        };

        this.init();
    }

    /**
     * Initialize the application
     */
    init() {
        this.populateSelects();
        this.attachEventListeners();
        this.setupFormValidation();
        console.log('IPL Win Predictor initialized successfully!');
        console.log('Connected to Flask server at:', this.apiEndpoint);
    }

    /**
     * Populate select dropdowns with teams and cities
     */
    populateSelects() {
        // Populate team selects
        this.teams.forEach(team => {
            const battingOption = document.createElement('option');
            battingOption.value = team;
            battingOption.textContent = team;
            this.elements.battingTeamSelect.appendChild(battingOption);

            const bowlingOption = document.createElement('option');
            bowlingOption.value = team;
            bowlingOption.textContent = team;
            this.elements.bowlingTeamSelect.appendChild(bowlingOption);
        });

        // Populate city select (sorted alphabetically)
        this.cities.sort().forEach(city => {
            const option = document.createElement('option');
            option.value = city;
            option.textContent = city;
            this.elements.citySelect.appendChild(option);
        });

        console.log('Dropdowns populated with teams and cities');
    }

    /**
     * Attach event listeners to form elements
     */
    attachEventListeners() {
        // Predict button click handler
        this.elements.predictBtn.addEventListener('click', () => {
            this.handlePrediction();
        });

        // Team selection change handlers
        this.elements.battingTeamSelect.addEventListener('change', () => {
            this.validateTeamSelection();
        });

        this.elements.bowlingTeamSelect.addEventListener('change', () => {
            this.validateTeamSelection();
        });

        // Enter key handler for inputs
        const inputs = [
            this.elements.targetInput,
            this.elements.scoreInput,
            this.elements.oversInput,
            this.elements.wicketsInput
        ];

        inputs.forEach(input => {
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.handlePrediction();
                }
            });

            // Real-time validation
            input.addEventListener('input', () => {
                this.clearError();
            });
        });

        console.log('Event listeners attached');
    }

    /**
     * Setup form validation rules
     */
    setupFormValidation() {
        // Add input constraints
        this.elements.targetInput.setAttribute('min', '1');
        this.elements.targetInput.setAttribute('max', '300');
        
        this.elements.scoreInput.setAttribute('min', '0');
        this.elements.scoreInput.setAttribute('max', '300');
        
        this.elements.oversInput.setAttribute('min', '0');
        this.elements.oversInput.setAttribute('max', '20');
        this.elements.oversInput.setAttribute('step', '0.1');
        
        this.elements.wicketsInput.setAttribute('min', '0');
        this.elements.wicketsInput.setAttribute('max', '10');
    }

    /**
     * Validate team selection to prevent same team selection
     */
    validateTeamSelection() {
        const battingTeam = this.elements.battingTeamSelect.value;
        const bowlingTeam = this.elements.bowlingTeamSelect.value;

        if (battingTeam && bowlingTeam && battingTeam === bowlingTeam) {
            this.showError('Batting and bowling teams cannot be the same!');
            return false;
        }

        this.clearError();
        return true;
    }

    /**
     * Main prediction handler
     */
    async handlePrediction() {
        try {
            // Clear previous errors
            this.clearError();

            // Validate form data
            const formData = this.validateForm();
            if (!formData) return;

            // Show loading state
            this.showLoading(true);

            // Calculate derived statistics
            const matchStats = this.calculateMatchStats(formData);

            // Make prediction API call to Flask server
            const prediction = await this.makePredictionRequest(matchStats);

            // Display results
            this.displayResults(prediction, formData.battingTeam, formData.bowlingTeam);

        } catch (error) {
            console.error('Prediction error:', error);
            this.showError('Failed to connect to the prediction server. Make sure your Flask server is running on port 5000.');
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Validate form inputs and return form data
     */
    validateForm() {
        const battingTeam = this.elements.battingTeamSelect.value.trim();
        const bowlingTeam = this.elements.bowlingTeamSelect.value.trim();
        const city = this.elements.citySelect.value.trim();
        const target = parseInt(this.elements.targetInput.value) || 0;
        const score = parseInt(this.elements.scoreInput.value) || 0;
        const overs = parseFloat(this.elements.oversInput.value) || 0;
        const wickets = parseInt(this.elements.wicketsInput.value) || 0;

        // Check required fields
        if (!battingTeam || !bowlingTeam || !city) {
            this.showError('Please select both teams and a venue.');
            return null;
        }

        // Validate team selection
        if (battingTeam === bowlingTeam) {
            this.showError('Batting and bowling teams must be different.');
            return null;
        }

        // Validate numerical inputs
        if (target < 1) {
            this.showError('Target score must be positive.');
            return null;
        }

        if (score < 0 ) {
            this.showError('Current score must be non negative.');
            return null;
        }

        if (overs < 0 || overs > 20) {
            this.showError('Overs must be between 0 and 20.');
            return null;
        }

        if (wickets < 0 || wickets > 10) {
            this.showError('Wickets must be between 0 and 10.');
            return null;
        }

        // Logical validations
        if (score >= target) {
            this.showError('Current score must be less than the target.');
            return null;
        }

        const ballsBowled = Math.floor(overs) * 6 + Math.round((overs % 1) * 10);
        if (ballsBowled > 120) {
            this.showError('Invalid overs format. Maximum is 20 overs.');
            return null;
        }

        return {
            battingTeam,
            bowlingTeam,
            city,
            target,
            score,
            overs,
            wickets,
            ballsBowled
        };
    }

    /**
     * Calculate match statistics for prediction
     */
    calculateMatchStats(formData) {
        const runsLeft = formData.target - formData.score;
        const ballsLeft = 120 - formData.ballsBowled;
        const wicketsLeft = 10 - formData.wickets;
        
        // Current Run Rate (runs per over)
        const crr = formData.ballsBowled === 0 ? 0 : (formData.score * 6) / formData.ballsBowled;
        
        // Required Run Rate (runs per over needed)
        const rrr = ballsLeft === 0 ? 999 : (runsLeft * 6) / ballsLeft;

        return {
            batting_team: formData.battingTeam,
            bowling_team: formData.bowlingTeam,
            city: formData.city,
            runs_left: runsLeft,
            balls_left: ballsLeft,
            wickets_left: wicketsLeft,
            runs_target: formData.target,
            crr: crr,
            rrr: rrr
        };
    }

    /**
     * Make API request for prediction to Flask server
     */
    async makePredictionRequest(matchStats) {
        // Handle edge cases first
        if (matchStats.balls_left <= 0 && matchStats.runs_left > 0) {
            return this.createMockResult(0, 0, 0, 0); // Bowling team wins
        }

        if (matchStats.runs_left <= 0) {
            return this.createMockResult(1, 1, 1, 1); // Batting team wins
        }

        try {
            console.log('Sending prediction request to Flask server:', matchStats);
            
            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(matchStats),
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server response error:', errorText);
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('Received prediction from Flask server:', result);
            return result;

        } catch (error) {
            console.error('Flask API request failed:', error);
            
            // Show more specific error message
            if (error.message.includes('Failed to fetch')) {
                throw new Error('Cannot connect to Flask server. Please ensure the server is running on http://127.0.0.1:5000');
            } else {
                throw new Error(`Server error: ${error.message}`);
            }
        }
    }

    /**
     * Create mock result for edge cases
     */
    createMockResult(final, rf, ann, lr) {
        return {
            final_prediction: final,
            model_breakdown: {
                random_forest: rf,
                ann: ann,
                logistic_regression: lr
            }
        };
    }

    /**
     * Display prediction results with animations
     */
    displayResults(prediction, battingTeam, bowlingTeam) {
        const battingWinProb = Math.round(prediction.final_prediction * 100);
        const bowlingWinProb = 100 - battingWinProb;

        // Update team labels
        document.getElementById('batting-team-result').textContent = battingTeam;
        document.getElementById('bowling-team-result').textContent = bowlingTeam;

        // Update probability bars with animation
        const battingBar = document.getElementById('batting-probability');
        const bowlingBar = document.getElementById('bowling-probability');

        // Apply team colors
        battingBar.style.background = `linear-gradient(45deg, ${this.teamColors[battingTeam]}, ${this.adjustBrightness(this.teamColors[battingTeam], 20)})`;
        bowlingBar.style.background = `linear-gradient(45deg, ${this.teamColors[bowlingTeam]}, ${this.adjustBrightness(this.teamColors[bowlingTeam], 20)})`;

        // Animate width changes
        setTimeout(() => {
            battingBar.style.width = `${battingWinProb}%`;
            bowlingBar.style.width = `${bowlingWinProb}%`;
        }, 100);

        // Update percentage text
        document.getElementById('batting-percentage').textContent = `${battingWinProb}%`;
        document.getElementById('bowling-percentage').textContent = `${bowlingWinProb}%`;

        // Update winner declaration
        const winner = battingWinProb > bowlingWinProb ? battingTeam : bowlingTeam;
        const winnerElement = document.getElementById('predicted-winner');
        winnerElement.textContent = winner;
        winnerElement.style.color = this.teamColors[winner];

        // Update individual model predictions
        document.getElementById('rf-prediction').textContent = `${(prediction.model_breakdown.random_forest * 100).toFixed(1)}%`;
        document.getElementById('ann-prediction').textContent = `${(prediction.model_breakdown.ann * 100).toFixed(1)}%`;
        document.getElementById('lr-prediction').textContent = `${(prediction.model_breakdown.logistic_regression * 100).toFixed(1)}%`;

        // Show results container with animation
        this.elements.resultContainer.classList.remove('hidden');
        
        console.log(`Prediction complete: ${winner} (${Math.max(battingWinProb, bowlingWinProb)}% probability)`);
    }

    /**
     * Adjust color brightness for gradient effects
     */
    adjustBrightness(hexColor, percent) {
        const num = parseInt(hexColor.replace("#", ""), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) + amt;
        const G = (num >> 8 & 0x00FF) + amt;
        const B = (num & 0x0000FF) + amt;
        return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
            (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
            (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
    }

    /**
     * Show/hide loading overlay
     */
    showLoading(show) {
        if (show) {
            this.elements.loadingOverlay.classList.remove('hidden');
            this.elements.predictBtn.disabled = true;
            this.elements.predictBtn.textContent = 'Analyzing...';
        } else {
            this.elements.loadingOverlay.classList.add('hidden');
            this.elements.predictBtn.disabled = false;
            this.elements.predictBtn.innerHTML = '<span class="button-text">Calculate Win Probability</span><div class="button-glow"></div>';
        }
    }

    /**
     * Display error message
     */
    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.errorMessage.style.display = 'block';
        
        // Auto-hide error after 7 seconds
        setTimeout(() => {
            this.clearError();
        }, 7000);
    }

    /**
     * Clear error message
     */
    clearError() {
        this.elements.errorMessage.textContent = '';
        this.elements.errorMessage.style.display = 'none';
    }

    /**
     * Test server connection
     */
    async testServerConnection() {
        try {
            const response = await fetch('http://127.0.0.1:5000/', {
                method: 'GET'
            });
            
            if (response.ok) {
                console.log('✅ Flask server connection successful!');
                return true;
            } else {
                console.log('❌ Flask server responded with error:', response.status);
                return false;
            }
        } catch (error) {
            console.log('❌ Cannot connect to Flask server:', error.message);
            return false;
        }
    }
}

// Initialize the application when DOM is fully loaded
document.addEventListener('DOMContentLoaded', async () => {
    const predictor = new IPLPredictor();
    
    // Test server connection
    await predictor.testServerConnection();
    
    // Make it globally accessible for debugging
    window.iplPredictor = predictor;
    
    console.log('🏏 IPL Win Predictor is ready to make predictions!');
    console.log('🔗 Make sure your Flask server is running on http://127.0.0.1:5000');
});