import gradio as gr
import joblib
import pandas as pd
import numpy as np

model = joblib.load('catboost_model.pkl')
scaler = joblib.load('scaler.pkl')
le_dict = joblib.load('label_encoders.pkl')

def predict(age, gender, bp, chol, na_to_k, hr, bs, bmi, liver, kidney):
    input_data = pd.DataFrame([[age, gender, bp, chol, na_to_k, hr, bs, bmi, liver, kidney]], 
                              columns=['Age', 'Gender', 'Blood_Pressure', 'Cholesterol', 'Na_to_K_Ratio', 'Heart_Rate', 'Blood_Sugar', 'BMI', 'Liver_Function', 'Kidney_Function'])
    
    for col in input_data.columns:
        if col in le_dict and col != 'Drug_Class':
            input_data[col] = le_dict[col].transform(input_data[col])
            
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    pred_val = int(np.array(prediction).flatten()[0])
    drug_class = le_dict['Drug_Class'].inverse_transform([pred_val])[0]
    return drug_class

theme = gr.themes.Monochrome(
    primary_hue="green", 
    secondary_hue="stone"
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    button_primary_background_fill="#ccff00",
    button_primary_text_color="#000000"
)

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Radio(["Low", "Normal", "High"], label="Blood Pressure"),
        gr.Radio(["Normal", "High"], label="Cholesterol"),
        gr.Number(label="Na to K Ratio"),
        gr.Number(label="Heart Rate"),
        gr.Radio(["Normal", "High"], label="Blood Sugar"),
        gr.Number(label="BMI"),
        gr.Radio(["Normal", "Abnormal"], label="Liver Function"),
        gr.Radio(["Normal", "Abnormal"], label="Kidney Function")
    ],
    outputs=gr.Textbox(label="Rekomendasi Obat Optimal"),
    title="Clinical-AI Ultimate",
    theme=theme
)

if __name__ == "__main__":
    interface.launch()