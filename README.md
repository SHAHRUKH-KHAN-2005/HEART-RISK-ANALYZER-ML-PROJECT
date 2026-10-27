❤️ HeartRisk Analyzer ML — Heart Disease Prediction

HeartRisk Analyzer ML (project folder: HeartCare AI) is a machine learning project that predicts whether a person may be classified as having heart disease based on selected health-related inputs. It includes a comparison of multiple classification algorithms and an interactive Streamlit dashboard powered by the selected Logistic Regression model.

Disclaimer: This project is for educational and demonstration purposes only. It is not a medical device and must not be used to diagnose, treat, or rule out heart disease. Always consult a qualified healthcare professional for medical advice.

📌 Project Overview

The project uses a heart disease dataset containing 918 records and 11 input features, with HeartDisease as the target column. The workflow covers data preprocessing, categorical encoding, feature scaling, model training and evaluation, model selection, and deployment through a web-based dashboard.

Five classification algorithms were compared:

Logistic Regression
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Decision Tree
Support Vector Machine (SVM) with an RBF kernel

In the experiments conducted for this project, Logistic Regression achieved the highest accuracy among the evaluated models and was selected for the final dashboard. The exact accuracy value is not listed here because it was not provided in the project description.

📸 Screenshots
Patient Information Form

Patient Information Form

Prediction Result

Prediction Result

✨ Dashboard Features
Predicts a Yes/No result based on the selected model.
Displays the model's predicted class probabilities as percentages.
Visualizes the estimated heart-disease probability using a circular (donut) chart.
Provides an interactive interface for entering patient features.
Includes light/dark appearance options.
Supports automatic predictions when inputs change.
Includes page-printing functionality.
Uses a saved model and preprocessing artifacts for inference.

The probability shown is the model's output for the supplied inputs. It should not be interpreted as a clinically validated estimate of a person's actual risk.

🧰 Tech Stack
Python — application and machine learning code
Pandas — data handling and preprocessing
Scikit-learn — preprocessing, model training, and evaluation
Streamlit — interactive dashboard
Plotly — probability visualization
Joblib — saving and loading trained artifacts
📊 Dataset Features

The dataset includes the following columns:

Feature	Description
Age	Age in years
Sex	Sex category in the dataset
ChestPainType	Chest pain category
RestingBP	Resting blood pressure
Cholesterol	Serum cholesterol value
FastingBS	Fasting blood sugar indicator
RestingECG	Resting electrocardiogram category
MaxHR	Maximum heart rate
ExerciseAngina	Exercise-induced angina indicator
Oldpeak	ST depression value
ST_Slope	Slope of the peak exercise ST segment
HeartDisease	Target label: 0 or 1

Important: Feature definitions, units, categorical values, and target-label meanings should be checked against the exact dataset source used in the notebook.

⚙️ Machine Learning Workflow
Load the dataset and inspect its columns and values.
Prepare the features and target, using HeartDisease as the target column.
Encode categorical features so the models can process them.
Scale numerical features with StandardScaler where appropriate.
Train and evaluate multiple classifiers using a consistent evaluation process.
Compare model performance and select Logistic Regression based on the results from this experiment.
Save the artifacts needed for prediction.
Load the saved artifacts in Streamlit and display the prediction and probability visualization.
Feature scaling note

The notebook uses StandardScaler for the numerical features:

python
numerical_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR"]

For a reliable evaluation, split the data before fitting preprocessing steps, and fit the scaler only on the training data. For cross-validation and hyperparameter tuning, use a Scikit-learn Pipeline so scaling is fitted separately inside each training fold. Apply the already-fitted scaler to validation, test, and user-input data.

📁 Project Structure

The application folder is expected to contain the following files:

HeartCare-AI-Project/
├── app.py
├── LogisticRegression_Heart.pkl
├── scaler.pkl
├── columns.pkl
├── assets/
│   ├── input-form.png
│   └── prediction-result.png
└── README.md
app.py — Streamlit dashboard and prediction interface.
LogisticRegression_Heart.pkl — trained Logistic Regression model.
scaler.pkl — fitted feature scaler used during training.
columns.pkl — saved feature-column order/schema used to prepare prediction input.
assets/ — screenshots used in this README.
README.md — project documentation.

If your notebook or dataset is stored separately, add its filename and location to this structure.

🚀 Getting Started
1. Clone or download the repository

If the project is hosted on GitHub, clone it with:

bash
git clone <your-repository-url>
cd HeartCare-AI-Project

Alternatively, download the project files and open the project folder in VS Code.

2. Create and activate a virtual environment (recommended)

Windows:

bash
python -m venv .venv
.venv\Scripts\activate

macOS/Linux:

bash
python -m venv .venv
source .venv/bin/activate
3. Install dependencies

Install the packages used by the application:

bash
pip install streamlit pandas scikit-learn plotly joblib

Use the versions from your development environment if you have a requirements.txt file. Pickle/joblib model files may not be compatible across all Scikit-learn versions.

4. Run the dashboard

From the directory containing app.py and the saved model files, run:

bash
streamlit run app.py

Streamlit should open the dashboard in your default browser (usually at http://localhost:8501). If it does not, follow the local URL shown in the terminal.

🧪 Model Artifacts

The app loads the saved files:

LogisticRegression_Heart.pkl
scaler.pkl
columns.pkl

Keep these files in the expected location relative to app.py, or update the paths in the application code.

Security note: Only load .pkl or Joblib files from a trusted source. Deserializing an untrusted model file can execute malicious code.

📈 Model Evaluation

The following models were compared in the project:

Model	Evaluated
Logistic Regression	Yes
K-Nearest Neighbors (KNN)	Yes
Gaussian Naive Bayes	Yes
Decision Tree	Yes
SVM (RBF kernel)	Yes

Logistic Regression gave the highest accuracy in the experiments performed and was selected for the dashboard. For reproducible reporting, add the actual test-set accuracy, precision, recall, F1-score, confusion matrix, and evaluation setup from your notebook. Results should be based on a held-out test set or an appropriate cross-validation procedure, with preprocessing fitted only on training data.

🛠️ Troubleshooting
ModuleNotFoundError: Activate the correct environment and install the missing package.
Model file not found: Ensure all three .pkl files are in the location expected by app.py.
Feature-name or column-order error: Ensure columns.pkl, the saved scaler, and the trained model were created from the same preprocessing pipeline.
Prediction errors after changing package versions: Recreate the environment with compatible versions of Python, Pandas, and Scikit-learn.
Streamlit ScriptRunContext warning: Run the app with streamlit run app.py rather than executing it as a plain Python script.
🔮 Future Improvements
Add a requirements.txt file with tested package versions.
Include the measured model metrics and confusion matrix.
Add automated tests for input validation and prediction output.
Track preprocessing and model versions together in a reproducible pipeline.
Improve accessibility and mobile responsiveness.
Evaluate the model on suitable external data before making any claims about generalization.
Add clear explanations of model limitations and the meaning of probability outputs.
👨‍💻 Author

Created by SHAHRUKH KHAN

This project was developed as part of my learning journey in Machine Learning, Python, and model deployment.