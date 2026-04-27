import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "Nikhil-Vaishnav-17/distilbert-imdb-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

def predict_sentiment(review):
    inputs = tokenizer(
        review, 
        return_tensors="pt", 
        truncation=True, 
        max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_id = torch.argmax(probs).item()
    labels = ["Negative", "Positive"]
    confidence = probs[0][pred_id].item()
    
    return f"{labels[pred_id]} ({confidence:.1%} confidence)"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Enter a movie review..."),
    outputs=gr.Text(label="Sentiment"),
    title="Movie Review Sentiment Analyzer",
    description="DistilBERT fine-tuned on IMDB dataset"
)

demo.launch()