# IPL Win Probability Predictor

This is a full-stack web application that predicts the win probability of a team chasing in the second innings of an Indian Premier League (IPL) T20 cricket match. The prediction is made in real-time based on the current match state and is powered by an ensemble of three different machine learning models.

The application features a dynamic and interactive 3D frontend that communicates with a Python-based backend server to deliver accurate, weighted-average predictions.

## Features

* **Real-Time Prediction**: Enter the current match situation (target score, current score, overs completed, wickets down) to get an instant win probability.
* **Ensemble Model**: The prediction is not based on a single model but on a weighted average of three different models (Random Forest, Artificial Neural Network, and Logistic Regression) for higher accuracy and robustness.
* **Individual Model Breakdown**: The interface shows the individual prediction from each model, providing insight into how the final weighted average is calculated.
* **Interactive 3D Frontend**: A visually appealing user interface built with Three.js, featuring an animated 3D background and a responsive "glassmorphism" design.
* **Full-Stack Architecture**: A clear separation between the frontend (HTML/CSS/JS) and the backend (Python/Flask), making the project scalable and maintainable.

## Tech Stack

This project is built with a combination of data science and web development technologies:

* **Frontend**:
    * HTML5
    * Tailwind CSS for styling
    * JavaScript
    * **Three.js** for the 3D animated background
* **Backend**:
    * **Python**
    * **Flask** for the web server and API
    * Gunicorn as the production WSGI server
* **Machine Learning**:
    * **Scikit-learn** for the Random Forest and Logistic Regression models
    * **TensorFlow (Keras)** for the Artificial Neural Network (ANN)
    * **Pandas** and **NumPy** for data manipulation
    * **Joblib** for model serialization

## Project Structure

The project directory is organized as follows:

```
IPL WIN predictor/
├── templates/
│   └── index.html         # The main HTML file for the frontend
├── app.py                 # The Python Flask server (backend logic)
├── requirements.txt       # A list of all required Python libraries
├── Procfile               # Command for the deployment service (Render)
├── ann_model.h5           # The saved and trained ANN model file
├── rf_model.joblib        # The saved and trained Random Forest model file
├── preprocessor.joblib    # The saved scikit-learn preprocessor
├── scaler.joblib          # The saved scikit-learn scaler
└── ipl-win-pred.ipynb     # The original Jupyter Notebook for model training
```

## Local Setup and Installation

To run this project on your local machine, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/IPL-win-percentage-predictor.git](https://github.com/your-username/IPL-win-percentage-predictor.git)
    cd IPL-win-percentage-predictor
    ```

2.  **Install Python**: Ensure you have a compatible version of Python installed (this project was built with **Python 3.11**).

3.  **Install Required Libraries**: Open a terminal in the project directory and install all the necessary packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Backend Server**: Start the Flask server by running the `app.py` file:
    ```bash
    python app.py
    ```
    The server will start and be running at `http://127.0.0.1:5000`.

5.  **View the Application**: Open your web browser and navigate to `http://127.0.0.1:5000`. The website should load and be fully functional.

## Machine Learning Models

The core of this application is the ensemble of three machine learning models trained on historical IPL match data.

1.  **Logistic Regression**: A simple and interpretable linear model that provides a baseline prediction.
2.  **Random Forest Classifier**: A powerful ensemble model consisting of many decision trees. It is highly accurate and robust against overfitting.
3.  **Artificial Neural Network (ANN)**: A deep learning model built with Keras/TensorFlow, capable of learning complex, non-linear patterns in the data.

### Weighted Average (Ensembling)

The final prediction is a weighted average of the outputs from these three models. The weights were assigned based on their respective accuracies during testing in the notebook:

* **Random Forest**: 45% weight (Accuracy: ~98%)
* **ANN**: 45% weight (Accuracy: ~98%)
* **Logistic Regression**: 10% weight (Accuracy: ~80%)

This approach ensures that the more accurate models have a greater influence on the final outcome, leading to a more reliable prediction.
