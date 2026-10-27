
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit.components.v1 as components
from pathlib import Path
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HeartCare AI - BY SHAHRUKH KHAN | Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# LOAD SAVED FILES
# Keep all three files in the same folder as app.py
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

try:
    model = joblib.load(BASE_DIR / "LogisticRegression_Heart.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    feature_names = list(joblib.load(BASE_DIR / "columns.pkl"))

except FileNotFoundError as e:
    st.error(
        f"Missing file: {e.filename}. Keep app.py, "
        "LogisticRegression_Heart.pkl, scaler.pkl, and "
        "columns.pkl in the same folder."
    )
    st.stop()

# =========================================================
# THEME
# =========================================================
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

theme_choice = st.sidebar.radio(
    "🎨 Appearance",
    ["Light", "Dark"],
    index=0 if st.session_state.theme == "Light" else 1
)
st.session_state.theme = theme_choice

dark = theme_choice == "Dark"

if dark:
    BG = "#101820"
    CARD = "#192630"
    TEXT = "#F0F6FA"
    MUTED = "#A8BAC7"
    BORDER = "#2D414E"
    INPUT_BG = "#21313D"
    ACCENT = "#35C9B5"
else:
    BG = "#F2F8FB"
    CARD = "#FFFFFF"
    TEXT = "#173042"
    MUTED = "#637A89"
    BORDER = "#DFEAF0"
    INPUT_BG = "#FFFFFF"
    ACCENT = "#087F8C"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BG};
        color: {TEXT};
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stSidebar"] {{
        background: {CARD};
    }}

    .block-container {{
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }}

    h1, h2, h3, p, label {{
        color: {TEXT};
    }}

    .hero {{
        padding: 25px 28px;
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 22px;
        margin-bottom: 20px;
    }}

    .hero-title {{
        color: {ACCENT};
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 6px;
    }}

    .hero-subtitle {{
        color: {MUTED};
        font-size: 15px;
    }}

    .panel {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
    }}

    .panel-title {{
        font-size: 21px;
        font-weight: 750;
        color: {ACCENT};
        margin-bottom: 15px;
    }}

    .result-box {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 22px;
        padding: 25px 20px;
        text-align: center;
        margin-bottom: 18px;
    }}

    .result-label {{
        font-size: 14px;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 1.4px;
    }}

    .result-value {{
        font-size: clamp(25px, 3vw, 38px);
        font-weight: 850;
        margin: 12px 0 5px 0;
    }}

    .result-note {{
        color: {MUTED};
        font-size: 14px;
    }}

    .metric-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 17px;
        margin-bottom: 12px;
    }}

    .metric-label {{
        font-size: 13px;
        color: {MUTED};
        margin-bottom: 7px;
    }}

    .metric-value {{
        color: {ACCENT};
        font-size: 23px;
        font-weight: 800;
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background: {INPUT_BG};
    }}

    div.stButton > button {{
        border-radius: 12px;
        min-height: 45px;
        font-weight: 700;
        border: 1px solid {BORDER};
    }}

    .footer {{
        text-align: center;
        color: {MUTED};
        padding: 20px 5px 8px 5px;
        border-top: 1px solid {BORDER};
        margin-top: 24px;
        font-size: 13px;
    }}

    @media print {{
        [data-testid="stSidebar"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        .stButton,
        [data-testid="stDecoration"] {{
            display: none !important;
        }}

        .stApp {{
            background: white !important;
            color: black !important;
        }}

        .panel, .result-box, .metric-card, .hero {{
            background: white !important;
            color: black !important;
            box-shadow: none !important;
            break-inside: avoid;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HEADER
# =========================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">❤️ HeartRisk Analyzer ML - BY SHAHRUKH KHAN</div>
        <div class="hero-subtitle">
            Heart disease prediction • Intelligent health insights
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("Demo prediction tool — not a medical diagnosis.")

# =========================================================
# PATIENT INPUTS
# No form is used, so Streamlit reruns the prediction whenever
# an input changes (automatic prediction).
# =========================================================
st.markdown(
    '<div class="panel-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=45,
        step=1,
        key="age"
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
        key="gender"
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=60,
        max_value=250,
        value=120,
        step=1,
        key="resting_bp"
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=200,
        step=1,
        key="cholesterol"
    )

with input_col2:
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1],
        format_func=lambda x: "0 - No" if x == 0 else "1 - Yes",
        key="fasting_bs"
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ASY", "ATA", "NAP", "TA"],
        format_func=lambda x: {
            "ASY": "ASY - Asymptomatic",
            "ATA": "ATA - Atypical Angina",
            "NAP": "NAP - Non-Anginal Pain",
            "TA": "TA - Typical Angina"
        }[x],
        key="chest_pain"
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"],
        format_func=lambda x: {
            "Normal": "Normal",
            "ST": "ST-T Wave Abnormality",
            "LVH": "Left Ventricular Hypertrophy"
        }[x],
        key="resting_ecg"
    )

    max_hr = st.number_input(
        "Maximum Heart Rate (beats/min)",
        min_value=50,
        max_value=250,
        value=150,
        step=1,
        key="max_hr"
    )

with input_col3:
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["No", "Yes"],
        key="exercise_angina"
    )

    oldpeak = st.number_input(
        "Oldpeak (ST Depression)",
        min_value=-3.0,
        max_value=7.0,
        value=0.0,
        step=0.1,
        format="%.1f",
        key="oldpeak"
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"],
        key="st_slope"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Prediction mode</div>
            <div class="metric-value" style="font-size:18px;">
                ⚡ Automatic
            </div>
            <div class="metric-label">
                The result updates when you change an input.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# PREPARE FEATURES
# Important: match the exact encoding used during training.
# =========================================================
numeric_values = {
    "Age": age,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs,
    "MaxHR": max_hr,
    "Oldpeak": oldpeak
}

selected_categories = {
    "Sex": "M" if gender == "Male" else "F",
    "ChestPainType": chest_pain,
    "RestingECG": resting_ecg,
    "ExerciseAngina": "Y" if exercise_angina == "Yes" else "N",
    "ST_Slope": st_slope
}

# Start with zeroes for all saved training features.
input_data = {feature: 0 for feature in feature_names}

# Fill numeric features.
for feature, value in numeric_values.items():
    if feature in input_data:
        input_data[feature] = value

# Fill one-hot categorical features.
for original_column, selected_value in selected_categories.items():
    encoded_column = f"{original_column}_{selected_value}"

    if encoded_column in input_data:
        input_data[encoded_column] = 1

# Keep the exact saved feature order.
input_df = pd.DataFrame(
    [[input_data[feature] for feature in feature_names]],
    columns=feature_names
)

# =========================================================
# AUTOMATIC PREDICTION
# =========================================================
try:
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    classes = list(model.classes_)

    # Assumption: training target 1 = disease, 0 = no disease.
    disease_class_index = next(
        (i for i, label in enumerate(classes)
         if label == 1 or str(label) == "1"),
        None
    )

    if disease_class_index is None:
        st.error(
            "The saved model does not contain class 1. "
            "Check how the target was encoded during training."
        )
        st.stop()

    if not hasattr(model, "predict_proba"):
        st.error(
            "This saved model does not support predict_proba(), "
            "so a probability percentage cannot be displayed."
        )
        st.stop()

    probabilities = model.predict_proba(scaled_input)[0]
    disease_probability = float(probabilities[disease_class_index])
    no_disease_probability = 1.0 - disease_probability

    # The predicted class should agree with the model's class labels.
    predicted_is_disease = (
        prediction == 1 or str(prediction) == "1"
    )

    result_text = "YES — Disease Predicted" if predicted_is_disease else "NO — Disease Not Predicted"
    result_color = "#E05260" if predicted_is_disease else "#17A673"
    result_icon = "⚠️" if predicted_is_disease else "✓"

    st.markdown("---")

    # =====================================================
    # MAIN RESULT
    # =====================================================
    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Prediction Result</div>
            <div class="result-value" style="color:{result_color};">
                {result_icon} {result_text}
            </div>
            <div class="result-note">
                Based on the information entered and the saved ML model.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # PROBABILITY + CIRCULAR CHART
    # =====================================================
    chart_col, detail_col = st.columns([1.1, 1])

    with chart_col:
        st.markdown(
            '<div class="panel-title">🟢 Disease Probability</div>',
            unsafe_allow_html=True
        )

        # Donut chart with the disease percentage in the center.
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Disease probability", "Remaining probability"],
                    values=[
                        disease_probability * 100,
                        no_disease_probability * 100
                    ],
                    hole=0.72,
                    sort=False,
                    direction="clockwise",
                    textinfo="none",
                    marker=dict(
                        colors=["#E05260", "#DCE7ED"],
                        line=dict(color=BG, width=3)
                    ),
                    hovertemplate="%{label}: %{value:.2f}%<extra></extra>"
                )
            ]
        )

        fig.update_layout(
            showlegend=False,
            height=310,
            margin=dict(l=5, r=5, t=5, b=5),
            paper_bgcolor=CARD,
            plot_bgcolor=CARD,
            annotations=[
                dict(
                    text=f"<b>{disease_probability * 100:.1f}%</b>",
                    x=0.5,
                    y=0.53,
                    font=dict(size=30, color=TEXT),
                    showarrow=False
                ),
                dict(
                    text="Disease probability",
                    x=0.5,
                    y=0.42,
                    font=dict(size=12, color=MUTED),
                    showarrow=False
                )
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with detail_col:
        st.markdown(
            '<div class="panel-title">📋 Key Information</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Disease probability</div>
                <div class="metric-value">
                    {disease_probability * 100:.2f}%
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">No-disease class probability</div>
                <div class="metric-value">
                    {no_disease_probability * 100:.2f}%
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Prediction status</div>
                <div class="metric-value" style="font-size:20px;">
                    {"YES" if predicted_is_disease else "NO"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "These percentages are model outputs, not confirmed medical "
            "risk estimates. A high or low value does not establish "
            "whether someone has heart disease."
        )

    # =====================================================
    # PATIENT SUMMARY
    # =====================================================
    st.markdown("---")
    st.markdown(
        '<div class="panel-title">📝 Patient Summary</div>',
        unsafe_allow_html=True
    )

    summary = pd.DataFrame(
        {
            "Feature": [
                "Age",
                "Gender",
                "Resting Blood Pressure",
                "Cholesterol",
                "Fasting Blood Sugar",
                "Chest Pain Type",
                "Resting ECG",
                "Maximum Heart Rate",
                "Exercise-Induced Angina",
                "Oldpeak",
                "ST Slope"
            ],
            "Value": [
                f"{age} years",
                gender,
                f"{resting_bp} mm Hg",
                f"{cholesterol} mg/dL",
                fasting_bs,
                chest_pain,
                resting_ecg,
                f"{max_hr} bpm",
                exercise_angina,
                f"{oldpeak:.1f}",
                st_slope
            ]
        }
    )

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )

    # =====================================================
    # PRINT / SAVE REPORT
    # =====================================================
    st.markdown("---")
    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        if st.button("🖨️ Print / Save as PDF", use_container_width=True):
            components.html(
                """
                <script>
                window.parent.print();
                </script>
                """,
                height=0
            )

    with action_col2:
        report_text = f"""
HEARTCARE AI — PREDICTION REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PREDICTION: {"YES — Disease Predicted" if predicted_is_disease else "NO — Disease Not Predicted"}

Disease probability: {disease_probability * 100:.2f}%
No-disease class probability: {no_disease_probability * 100:.2f}%

PATIENT INFORMATION
Age: {age}
Gender: {gender}
Resting Blood Pressure: {resting_bp} mm Hg
Cholesterol: {cholesterol} mg/dL
Fasting Blood Sugar: {fasting_bs}
Chest Pain Type: {chest_pain}
Resting ECG: {resting_ecg}
Maximum Heart Rate: {max_hr}
Exercise-Induced Angina: {exercise_angina}
Oldpeak: {oldpeak}
ST Slope: {st_slope}

IMPORTANT:
This is an educational machine-learning prediction, not a medical
diagnosis. Consult a qualified healthcare professional for assessment.
        """

        st.download_button(
            "⬇️ Download Report",
            data=report_text,
            file_name="HeartCare_AI_Report.txt",
            mime="text/plain",
            use_container_width=True
        )

    with action_col3:
        if st.button("🔄 Reset Inputs", use_container_width=True):
            for key in [
                "age", "gender", "resting_bp", "cholesterol",
                "fasting_bs", "chest_pain", "resting_ecg",
                "max_hr", "exercise_angina", "oldpeak", "st_slope"
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

except Exception as error:
    st.error(
        "Prediction could not be completed. Verify that the saved "
        "model, scaler, feature names, and categorical encoding match "
        "the preprocessing used during training."
    )
    st.exception(error)

# =========================================================
# SCREEN RECORDING INFORMATION
# =========================================================
with st.expander("🎥 How to record the dashboard screen"):
    st.write(
        "On Windows, press **Windows + Alt + R** to start or stop "
        "recording if Xbox Game Bar recording is available and enabled. "
        "You can also use a screen recorder such as OBS Studio."
    )
    st.caption(
        "Streamlit does not provide a reliable built-in screen-recording "
        "button in this script. Recording depends on your operating "
        "system or a separately configured recording tool."
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        ❤️ HeartCare AI<br>
        Created by <b>SHAHRUKH KHAN</b><br>
        For educational use only — not a medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)
