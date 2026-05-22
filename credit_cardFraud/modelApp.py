import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="💳",
    layout="wide"
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================
model = joblib.load("fraud_detection_model.pkl")

# ==========================================
# CUSTOM CSS STYLING
# ==========================================
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    color: #1f77b4;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
}

.stButton>button {
    width: 100%;
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
    font-size: 20px;
    height: 3em;
    border: none;
}

.stButton>button:hover {
    background-color: #125d98;
    color: white;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}

.safe {
    background-color: #d4edda;
    color: #155724;
}

.fraud {
    background-color: #f8d7da;
    color: #721c24;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("💳 Fraud Detection System")

st.sidebar.info("""
This application predicts whether a transaction is:

✅ Legitimate  
⚠️ Fraudulent

using Machine Learning.
""")

st.sidebar.markdown("---")
st.sidebar.write("Built with Streamlit 🚀")

# ==========================================
# HEADER
# ==========================================
st.markdown(
    "<h1 class='title'>💳 Credit Card Fraud Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Analyze transactions using AI-powered fraud detection.</p>",
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# INPUT SECTION
# ==========================================
col1, col2 = st.columns(2)

with col1:

    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    oldbalanceOrg = st.number_input(
        "Sender Old Balance",
        min_value=0.0,
        value=0.0
    )

with col2:

    newbalanceOrig = st.number_input(
        "Sender New Balance",
        min_value=0.0,
        value=0.0
    )

    oldbalanceDest = st.number_input(
        "Receiver Old Balance",
        min_value=0.0,
        value=0.0
    )

    newbalanceDest = st.number_input(
        "Receiver New Balance",
        min_value=0.0,
        value=0.0
    )

st.divider()

# ==========================================
# MANUAL ONE-HOT ENCODING
# ==========================================
type_mapping = {
    "PAYMENT":  [1, 0, 0, 0],
    "TRANSFER": [0, 1, 0, 0],
    "CASH_OUT": [0, 0, 1, 0],
    "DEPOSIT":  [0, 0, 0, 1]
}

encoded_type = type_mapping[transaction_type]

# ==========================================
# PREDICTION BUTTON
# ==========================================
if st.button("🔍 Analyze Transaction"):

    # ==========================================
    # PREPARE INPUT DATA
    # ==========================================
    final_input = [[
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,

        # Encoded transaction type
        encoded_type[0],
        encoded_type[1],
        encoded_type[2],
        encoded_type[3]
    ]]

    # Convert to DataFrame
    input_df = pd.DataFrame(final_input)

    # ==========================================
    # MAKE PREDICTION
    # ==========================================
    prediction = model.predict(input_df)[0]

    # Prediction probability
    prediction_proba = model.predict_proba(input_df)[0]

    fraud_probability = round(prediction_proba[1] * 100, 2)

    # ==========================================
    # DISPLAY RESULT
    # ==========================================
    st.subheader("Prediction Result")

    if prediction == 1:

        st.markdown("""
        <div class="result-box fraud">
            ⚠️ Fraudulent Transaction Detected
        </div>
        """, unsafe_allow_html=True)

        st.error(
            f"Fraud Probability: {fraud_probability}%"
        )

    else:

        st.markdown("""
        <div class="result-box safe">
            ✅ Legitimate Transaction
        </div>
        """, unsafe_allow_html=True)

        st.success(
            f"Fraud Probability: {fraud_probability}%"
        )

    # ==========================================
    # TRANSACTION SUMMARY
    # ==========================================
    st.markdown("### 📋 Transaction Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Transaction Type",
            "Amount",
            "Sender Old Balance",
            "Sender New Balance",
            "Receiver Old Balance",
            "Receiver New Balance"
        ],
        "Value": [
            transaction_type,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest
        ]
    })

    st.dataframe(summary, use_container_width=True)

# ==========================================
# FOOTER
# ==========================================
st.divider()

st.caption("© 2026 Fraud Detection System | Built with Streamlit & Scikit-Learn")