import joblib
import pandas as pd
import gradio as gr

model = joblib.load("insurance_model.joblib")

def predict_insurance(age, sex, bmi, children, smoker, region):
    row = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "is_smoker": 1 if str(smoker).lower()=="yes" else 0,
        "bmi_category": ("underweight" if bmi < 18.5 else
                         "normal" if bmi < 25 else
                         "overweight" if bmi < 30 else "obese")
    }])
    p = model.predict(row)[0]
    return f"Predicted insurance charge: ${p:,.2f}"

demo = gr.Interface(
    fn=predict_insurance,
    inputs=[
        gr.Number(label="Age", value=30),
        gr.Dropdown(["male","female"], label="Sex", value="male"),
        gr.Number(label="BMI", value=27.5),
        gr.Number(label="Children", value=0),
        gr.Dropdown(["yes","no"], label="Smoker", value="no"),
        gr.Dropdown(["northeast","northwest","southeast","southwest"], label="Region", value="northeast"),
    ],
    outputs="text",
    title="Medical Insurance Cost Prediction"
)

if __name__ == "__main__":
    demo.launch()
