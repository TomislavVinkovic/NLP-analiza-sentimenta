import gradio as gr
import numpy as np
from predictor import Predictor

# Inicijaliziram svoj objekt koji u sebi sadrži sve moje modele
predictor = Predictor()

# Ova funkcija se poziva prilikom submita
def predict_text(text, model_name):
    result = None
    if(text == ""):
        return "Error: Please select a model from the dropdown"
    if model_name == "Bernoulli Naive Bayes":
        result = predictor.bernoulliPredict(text)
    elif model_name == "Bernoulli Naive Bayes (No Stops)":
        result = predictor.bernoulliNoStopsPredict(text)
    elif model_name == "Multinomial Naive Bayes":
        result = predictor.multinomialPredict(text)
    elif model_name == "Gaussian Naive Bayes":
        result = predictor.gaussianPredict(text)
    elif model_name == "Logistic Regression":
        result = predictor.logisticRegressionPredict(text)
    elif model_name == "BERT Model":
        result = predictor.bertPredict(text)
    
    if result == None:
        return "Error: Invalid model name"

    return "Positive" if result == 1 else "Negative"
    



# Kreiranje gradio interfejsa
input_text = gr.components.Textbox(label="Input Text")
model_selector = gr.components.Dropdown(choices=["Bernoulli Naive Bayes", "Bernoulli Naive Bayes (No Stops)",
                                             "Multinomial Naive Bayes", "Gaussian Naive Bayes",
                                             "Logistic Regression", "BERT Model"],
                                   label="Select Model")
                                   
output_text = gr.components.Textbox(label="Prediction")

interface = gr.Interface(fn=predict_text, inputs=[input_text, model_selector], outputs=output_text,
                         title="NLP Model Predictor", description="Select a model and enter text for prediction.")

# Pokretanje interfejsa sa share opcijom
interface.launch(share=True)
