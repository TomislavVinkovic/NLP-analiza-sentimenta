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
    elif model_name == "Bernoulli Naive Bayes (strip stopwords mode)":
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
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown(
        "# Sentiment analysis using NLP models"
    )
    gr.Markdown(
        """
        This app uses various NLP models to predict the sentiment of a given text. The models used are: Bernoulli Naive Bayes (with and without stopword exclusion), Multinomial Naive Bayes, Gaussian Naive Bayes, Logistic Regression and Google BERT.\n
        All models were trained on a 20000 tweet subset of the Sentiment 140 dataset.\n
        This demo was created as a part of the course 'Natural Language Processing' at the Faculty of Applied Mathematics and Information Technology in Osijek.
        """
    )

    with gr.Row():
        with gr.Column():
            model_selector = gr.components.Dropdown(choices=["Bernoulli Naive Bayes", "Bernoulli Naive Bayes (strip stopwords mode)",
                                                    "Multinomial Naive Bayes", "Gaussian Naive Bayes",
                                                    "Logistic Regression", "BERT Model"],
                                        label="Select Model")
            input_text = gr.components.Textbox(label="Input Text")
        with gr.Column(scale=2):
            output = gr.components.Textbox(label="Result", )
    submit = gr.components.Button(value="Submit")
    submit.click(predict_text, inputs=[input_text, model_selector], outputs=output)
    

    gr.Markdown(
        "#### How to use?"
    )
    gr.Markdown(
        """
            1. Select a model from the dropdown menu\n
            2. Enter a short piece of text in the text input\n
            3. Click submit
        """
    )
    gr.Markdown(
        "#### Output"
    )
    gr.Markdown(
        "The output will be either 'Positive' or 'Negative', depending on the sentiment of the text."
    )
    gr.Markdown(
        "#### Notes:"
    )
    gr.Markdown(
        """
        * When using any model for the first time, it may take a few seconds to give back the response as the model needs to be loaded.
        * The models should only be used on short pieces of text (tweets, comments, etc.) in the english language.
        * The output of the model may not be accurate.
        """
    )
    gr.HTML(
        """
            <div style="text-align:center">
                <strong>This project was created by Tomislav Vinkovic and David Matej Vnuk.</strong>
            </div>
        """
    )



# Pokretanje interfejsa sa share opcijom
app.launch(share=True)