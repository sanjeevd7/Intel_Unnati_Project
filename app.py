import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and column headers
model = joblib.load('model/rf_model.pkl')
scaler = joblib.load('model/scaler.pkl')
model_columns = joblib.load('model/rf_features.pkl').columns.tolist()

# Define original KDD feature names
columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
           'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
           'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count',
           'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
           'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
           'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
           'dst_host_srv_rerror_rate','attack','level']

# App UI
st.title("🚨 Network Packet Malicious Detection")

uploaded_file = st.file_uploader("Upload a CSV file to check for malicious packets", type=["csv"])

if uploaded_file is not None:
    try:
        # Load data with header
        df = pd.read_csv(uploaded_file)

        # Optional: Drop unused columns
        df = df.drop(columns=['attack', 'level'], errors='ignore')

        # One-hot encode categorical features
        df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'], prefix="", prefix_sep="")

        # Align with training features
        df = df.reindex(columns=model_columns, fill_value=0)

        # Scale
        X_scaled = scaler.transform(df)

        # Predict
        predictions = model.predict(X_scaled)

        # Show results
        df['Prediction'] = predictions
        df['Prediction_Label'] = df['Prediction'].map({0: 'Normal', 1: 'Malicious'})
        st.success("✅ Prediction complete.")
        st.dataframe(df[['Prediction_Label']])

        # Optional download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", data=csv, file_name='prediction_output.csv', mime='text/csv')

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
