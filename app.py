import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import shap

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(layout="wide")

# ---------------- STYLING ---------------- #
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 3rem;}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 8px;
}

.big-text {font-size: 18px; line-height: 1.7;}
.section-title {font-size: 22px; font-weight: bold; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD FILES ---------------- #
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
X_columns = pd.read_pickle("X_columns.pkl")

# ---------------- FEATURE EXPLANATIONS ---------------- #
feature_explanations = {
    "src_bytes": "High amount of data sent from source",
    "dst_bytes": "High amount of data received from destination",
    "count": "Too many connections in a short time",
    "srv_count": "High number of connections to same service",
    "serror_rate": "Frequent connection errors detected",
    "same_srv_rate": "Repeated connections to same service",
    "dst_host_count": "High number of connections to host",
    "dst_host_srv_count": "High number of service requests to host",
    "flag_SF": "Connection completed normally",
    "flag_REJ": "Connection was rejected",
    "flag_S0": "Connection attempt with no response"
}

# ---------------- HEADER ---------------- #
st.title("Network Anomaly Detection Dashboard")
st.caption("Machine Learning based intrusion detection system")

# ---------------- OVERVIEW ---------------- #
st.markdown("### Overview")
st.write("""
This system analyzes network traffic and classifies it as normal or malicious.
It also explains WHY the prediction was made using SHAP.
""")

# ---------------- STATUS ---------------- #
col1, col2, col3 = st.columns(3)
col1.metric("Model", "Ready")
col2.metric("Dataset", "NSL-KDD")
col3.metric("Mode", "Detection")

st.divider()

# ---------------- DROPDOWN ---------------- #
choice = st.selectbox("Select Traffic Type to Test", ["Normal", "Attack"])

# Centered button
col_btn = st.columns([2,1,2])[1]
with col_btn:
    run = st.button("Run Detection")

# ---------------- MAIN ---------------- #
if run:

    with st.spinner("Running detection..."):
        time.sleep(1)

        df = pd.read_csv("KDDTrain+.txt", header=None)

        columns = [
            "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
            "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
            "root_shell","su_attempted","num_root","num_file_creations","num_shells",
            "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
            "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
            "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
            "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
            "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate",
            "dst_host_srv_serror_rate","dst_host_rerror_rate",
            "dst_host_srv_rerror_rate","label","difficulty_level"
        ]

        df.columns = columns
        df = df.drop("difficulty_level", axis=1)

        if choice == "Normal":
            sample_row = df[df["label"] == "normal"].sample(1)
        else:
            sample_row = df[df["label"] != "normal"].sample(1)

        df_encoded = pd.get_dummies(df, columns=['protocol_type','service','flag'])
        df_encoded = df_encoded.reindex(columns=X_columns, fill_value=0)

        sample = df_encoded.loc[sample_row.index].values.reshape(1, -1)
        sample_scaled = scaler.transform(sample)

        pred = model.predict(sample_scaled)
        proba = model.predict_proba(sample_scaled)
        confidence = max(proba[0]) * 100

    # ---------------- RESULT ---------------- #
    st.write("Actual Label:", sample_row["label"].values[0])

    if pred[0] == 0:
        st.success("Prediction: Normal Traffic")
    else:
        st.error("Prediction: Attack Detected")

    # ---------------- METRICS ---------------- #
    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction", "Normal" if pred[0]==0 else "Attack")
    col2.metric("Confidence", f"{confidence:.2f}%")
    col3.metric("Status", "Safe" if pred[0]==0 else "Suspicious")

    st.divider()

    # ---------------- SHAP ---------------- #
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_scaled)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    shap_array = shap_values[0]

    feature_df = pd.DataFrame({
        "Feature": X_columns,
        "Importance": np.abs(shap_array)
    })

    top_features = feature_df.sort_values(by="Importance", ascending=False).head(5)

    col1, col2 = st.columns([1.2, 1.8])

    # -------- LEFT -------- #
    with col1:
        st.subheader("Key Influencing Factors")

        st.dataframe(top_features.reset_index(drop=True), use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Insights")

        for _, row in top_features.iterrows():
            st.markdown(f"""
            **{row['Feature']}**  
            - Impact: {row['Importance']:.3f}  
            - {feature_explanations.get(row['Feature'], "Important feature")}
            """)

    # -------- RIGHT -------- #
    with col2:
        st.subheader("Model Explanation (Top Features)")

        top_feature_names = top_features["Feature"].values
        shap_top = shap_array[[X_columns.get_loc(f) for f in top_feature_names]]

        # Emphasize first bar slightly
        bar_values = shap_top.copy()
        bar_values[0] = bar_values[0] * 1.25

        fig, ax = plt.subplots(figsize=(5, 2.8))

        ax.barh(top_feature_names, bar_values)
        ax.invert_yaxis()

        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=8)

        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    # ---------------- EXPLANATION ---------------- #
    st.markdown("## Explanation of Prediction")

    explanations = [feature_explanations.get(f, f) for f in top_features["Feature"]]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Key Observations</div>', unsafe_allow_html=True)
        for exp in explanations:
            st.markdown(f'<div class="big-text">• {exp}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Overall Interpretation</div>', unsafe_allow_html=True)

        if pred[0] == 1:
            st.markdown("""
            <div class="big-text">
            • Traffic shows abnormal behavior<br>
            • Possible intrusion detected
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="big-text">
            • Traffic follows normal patterns<br>
            • No suspicious activity detected
            </div>
            """, unsafe_allow_html=True)

# ---------------- SPACING ---------------- #
st.markdown("<br><br>", unsafe_allow_html=True)