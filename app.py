import streamlit as st
import joblib
import numpy as np
import pickle
import pandas as pd

st.markdown(
    """
    <style>
    .stApp {
        background-color: #90EE90;
        color: brown;   /* default text color */
    }

    h1 {
        color: #1f77b4;  /* blue title */
    }

    label {
        color: #1f77b4 !important;
        font-weight: 600;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label span p {
        color: brown !important;
        font-weight: 600;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

le = pickle.load(open("le_predictivemaintenance.pkl", "rb"))   # LabelEncoder
pt = pickle.load(open("powertransformer_predictivemaintenance.pkl", "rb")) #Power Transformer
scaler = pickle.load(open("scaler_predictivemaintenance.pkl", "rb")) #MinMaxScaler
model = pickle.load(open("best_model_predictivemaintenance.pkl", "rb"))  # best XGB model

#st.title("Machine Failure Predictor")
st.markdown(
    "<h1 style='text-align: center;'>Machine Failure Predictor</h1>",
    unsafe_allow_html=True
)
st.image("SS_PREVENTIVE_MAINTENANCE.png", 
             #caption="Income Classifier", 
              use_container_width=True)
st.write("This app predicts the probability of machine failure based on input parameters.")

type_map = {
    "Low": "L",
    "Medium": "M",
    "High": "H"
}

selected = st.radio(
    "Type of Machine",
    options=list(type_map.keys()),
    index=0,
    horizontal=True
)

air_temperature = st.number_input(
    "Air Temperature (K)",
    min_value=273.0,
    max_value=350.0,
    value=300.0,
    step=0.1,
    format="%.1f"
)

process_temperature = st.number_input(
    "Process Temperature (K)",
    min_value=273.0,
    max_value=350.0,    
    value=300.0,
    step=0.1,   
    format="%.1f"
)

rotational_speed = st.number_input(
    "Rotational Speed (RPM)",   
    min_value=0,
    max_value=3000,
    value=1000,
    step=1
)

torque = st.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=0.1,
    format="%.1f"
)

tool_wear = st.number_input(
    "Tool Wear (min)", 
    min_value=0,
    max_value=300,
    value=100,
    step=1
)


#type = st.radio(
#    "Type of Machine",
#    options=["L", "M", "H"],
#    index=0,
#    horizontal=True
#)



type = type_map[selected]

if st.button("Classify Macine Failure Possibility"):
    input_data = np.array([[type, air_temperature, process_temperature, rotational_speed, torque, tool_wear]])
    input_df = pd.DataFrame(input_data, columns=['type', 'air_temperature', 'process_temperature',
       'rotational_speed', 'torque', 'tool_wear'])
    
    # Encode the categorical variable
    input_df['type'] = le.transform(input_df['type'])

    skewed_cols = ['rotational_speed']
    
    # Apply power transformation
    input_transformed = input_df.copy()
    input_transformed[skewed_cols] = pt.transform(input_df[skewed_cols])
    

    feature_order = ['type', 'air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']

    input_data = input_transformed[feature_order]
    # Scale the features
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    #st.write(f"Prediction: {prediction[0]}")
    if prediction[0] == 1:
        Prediction_text = "High possibility of machine failure."
    else:
        Prediction_text = "Low possibility of machine failure."


    st.markdown(
                    f"""
                    <div style="
                        background-color:#e8f0ff;
                        padding:15px;
                        border-radius:10px;
                        border:2px solid #0B3D91;
                        color:#0B3D91;
                        font-size:18px;
                        font-weight:bold;
                        text-align:center;
                    ">
                        Predicted Machine Failure Possibility: {Prediction_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )