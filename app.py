import base64
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Health Risk Predictor", layout="centered")

# ================== BACKGROUND IMAGE ==================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(f"""
    <style>
    /* Main app background */
    .stApp {{
        background: url("data:image/jpg;base64,{encoded}") no-repeat center center fixed;
        background-size: cover;
        color: black !important;
    }}

    /* Force all text to black */
    html, body, [class*="css"], label, p, span, div {{
        color: black !important;
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: black !important;
        text-align: center;
    }}

    /* Inputs and textareas + CURSOR FIX */
    input, textarea, .stTextInput input, .stNumberInput input {{
        background-color: #f0f2f6 !important;
        color: black !important;
        border-radius: 10px !important;
        caret-color: black !important;  /* ✅ BLACK CURSOR */
    }}

    /* Selectboxes */
    div[data-baseweb="select"] > div {{
        background-color: #f0f2f6 !important;
        border-radius: 10px !important;
        border: 2px solid transparent !important;
    }}
    div[data-baseweb="select"] * {{
        color: black !important;
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 12px;
        font-size: 18px;
    }}
    </style>
    """, unsafe_allow_html=True)

# CALL BACKGROUND IMAGE
set_bg("eg.jpg")

# ================== SESSION ==================
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ================== MODEL CACHE ==================
@st.cache_resource
def load_models():
    lung_df = pd.read_excel("lungcancer.xlsx")
    dia_df = pd.read_excel("diabetes_prediction_dataset.xlsx")
    heart_df = pd.read_excel("HeartDiseaseTrain-Test.xlsx")
    bp_df = pd.read_excel("hypertension_dataset.xlsx")
    liver_df = pd.read_excel("Indian Liver Patient Dataset (ILPD).xlsx")

    lung_df["LUNG_CANCER"] = lung_df["LUNG_CANCER"].astype(str).str.upper()
    bp_df["Has_Hypertension"] = bp_df["Has_Hypertension"].astype(str).str.upper()

    lung_X = lung_df[["GENDER","AGE","SMOKING","ALCOHOL_CONSUMING","SHORTNESS_OF_BREATH","CHEST_PAIN"]].copy()
    lung_X["GENDER"] = lung_X["GENDER"].map({"M":1,"F":0})
    lung_model = RandomForestClassifier().fit(lung_X, lung_df["LUNG_CANCER"].map({"YES":1,"NO":0}))

    dia_X = dia_df[["gender","age","hypertension","heart_disease","bmi"]].copy()
    dia_X["gender"] = dia_X["gender"].map({"Male":1,"Female":0})
    dia_model = RandomForestClassifier().fit(dia_X, dia_df["diabetes"])

    heart_X = heart_df[["age","sex","cholestoral","Max_heart_rate","thalassemia"]].copy()
    heart_X["sex"] = heart_X["sex"].map({"Male":1,"Female":0})
    heart_X["thalassemia"] = heart_X["thalassemia"].map({"Normal":0,"Fixed Defect":1,"Reversable Defect":2})
    heart_model = RandomForestClassifier().fit(heart_X, heart_df["target"])

    bp_X = bp_df[["Age","BMI","Salt_Intake","BP_History","Smoking_Status"]].copy()
    bp_X["BP_History"] = bp_X["BP_History"].map({"Normal":0,"High":1})
    bp_X["Smoking_Status"] = bp_X["Smoking_Status"].map({"Non-Smoker":0,"Smoker":1})
    bp_model = RandomForestClassifier().fit(bp_X, bp_df["Has_Hypertension"].map({"YES":1,"NO":0}))

    liver_X = liver_df[["age","gender","tot_bilirubin","direct_bilirubin","tot_proteins","albumin","ag_ratio"]].copy()
    liver_X["gender"] = liver_X["gender"].map({"Male":1,"Female":0})
    liver_model = RandomForestClassifier().fit(liver_X, liver_df["is_patient"])

    return lung_model, dia_model, heart_model, bp_model, liver_model

# ================== WELCOME PAGE ==================
if st.session_state.page == "welcome":

    st.title("💊 Health Risk Predictor")
    st.markdown("### 🧠 Smart AI-Based Health Analysis System")

    st.markdown("""
    <br>
    <div style='text-align:center; font-size:18px;'>
    🔍 Predict multiple health risks <br><br>
    ❤️ Heart | 🩸 Diabetes | 🫁 Lung | 💉 BP | 🧪 Liver <br><br>
    📊 Accurate + Fast + User Friendly
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1.5,1])
    with col2:
        if st.button("🚀 Start Health Check"):
            st.session_state.page = "predict"
            st.rerun()

# ================== PREDICT PAGE ==================
elif st.session_state.page == "predict":

    lung_model, dia_model, heart_model, bp_model, liver_model = load_models()

    st.title("📝 Enter Patient Details")

    name = st.text_input("👤 Enter Patient Name")

    def yn(x): return 1 if x=="Yes" else 0
    def gender_map(x): return 1 if x=="Male" else 0

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🎂 Age (1–120)", 1, 120)
        gender = st.selectbox("👤 Gender", ["Male","Female"])
        smoking = st.selectbox("🚬 Smoking", ["Yes","No"])
        alcohol = st.selectbox("🍺 Alcohol", ["Yes","No"])
        breath = st.selectbox("😮‍💨 Breath Problem", ["Yes","No"])

    with col2:
        chest = st.selectbox("💔 Chest Pain", ["Yes","No"])
        hypertension = st.selectbox("💉 Hypertension", ["Yes","No"])
        heart_hist = st.selectbox("❤️ Heart History", ["Yes","No"])
        bmi = st.number_input("⚖️ BMI (10–50)", 10.0, 50.0)
        chol = st.number_input("🧪 Cholesterol (100–400)", 100.0, 400.0)

    st.subheader("❤️ Heart Details")
    max_hr = st.number_input("💓 Max Heart Rate (60–200)", 60, 200)
    thal = st.selectbox("Thalassemia", ["Normal","Fixed Defect","Reversable Defect"])

    st.subheader("💉 BP Details")
    salt = st.number_input("Salt Intake (5–15)", 5.0, 15.0)
    bp_hist = st.selectbox("BP History", ["Yes","No"])

    st.subheader("🧪 Liver Details")
    tot_bil = st.number_input("Total Bilirubin (0.1–20)", 0.1, 20.0)
    dir_bil = st.number_input("Direct Bilirubin (0–10)", 0.0, 10.0)
    prot = st.number_input("Proteins (100–800)", 100.0, 800.0)
    alb = st.number_input("Albumin (10–100)", 10.0, 100.0)
    ratio = st.number_input("A/G Ratio (0.5–2.5)", 0.5, 2.5)

    if st.button("🔍 Predict"):

        if name.strip() == "":
            st.warning("⚠️ Please enter patient name")
        else:
            g = gender_map(gender)
            thal_map = {"Normal":0,"Fixed Defect":1,"Reversable Defect":2}

            st.session_state.results = {
                "name": name,
                "lung": lung_model.predict_proba([[g,age,yn(smoking),yn(alcohol),yn(breath),yn(chest)]])[0][1],
                "dia": dia_model.predict_proba([[g,age,yn(hypertension),yn(heart_hist),bmi]])[0][1],
                "heart": heart_model.predict_proba([[age,g,chol,max_hr,thal_map[thal]]])[0][1],
                "bp": bp_model.predict_proba([[age,bmi,salt,yn(bp_hist),yn(smoking)]])[0][1],
                "liver": liver_model.predict_proba([[age,g,tot_bil,dir_bil,prot,alb,ratio]])[0][1]
            }

            st.session_state.page = "result"
            st.rerun()

# ================== RESULT PAGE ==================
elif st.session_state.page == "result":

    st.title("📊 Results")

    r = st.session_state.results
    st.subheader(f"👤 Patient Name: {r['name']}")

    def risk_label(p):
        if p > 0.6:
            return "🔴 High"
        elif p > 0.3:
            return "🟡 Moderate"
        else:
            return "🟢 Low"

    st.write(f"🫁 Lung: {round(r['lung']*100,2)}% → {risk_label(r['lung'])}")
    st.progress(r['lung'])

    st.write(f"🩸 Diabetes: {round(r['dia']*100,2)}% → {risk_label(r['dia'])}")
    st.progress(r['dia'])

    st.write(f"❤️ Heart: {round(r['heart']*100,2)}% → {risk_label(r['heart'])}")
    st.progress(r['heart'])

    st.write(f"💉 BP: {round(r['bp']*100,2)}% → {risk_label(r['bp'])}")
    st.progress(r['bp'])

    st.write(f"🧪 Liver: {round(r['liver']*100,2)}% → {risk_label(r['liver'])}")
    st.progress(r['liver'])

    if st.button("➡️ Finish"):
        st.session_state.page = "thankyou"
        st.rerun()

# ================== THANK YOU ==================
elif st.session_state.page == "thankyou":

    st.title("🙏 Thank You for Visiting")

    st.markdown("""
    <div style='text-align:center; font-size:20px;'>
    💙 Stay Healthy <br><br>
    🧠 Your health matters!
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Restart"):
        st.session_state.page = "welcome"
        st.rerun()
