# model_components.py
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.pipeline = pipeline("sentiment-analysis", model=model_name)
    def analyze(self, text):
        return self.pipeline(text)[0]["label"]

class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.pipeline = pipeline("summarization", model=model_name)
    def summarize(self, text):
        return self.pipeline(text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]

class QAHelper:
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad"):
        self.pipeline = pipeline("question-answering", model=model_name)
    def answer(self, question, context):
        return self.pipeline(question=question, context=context)["answer"]

class TextGenerator:
    def __init__(self, model_name="gpt2"):
        self.pipeline = pipeline("text-generation", model=model_name)
    def generate(self, prompt):
        return self.pipeline(prompt, max_length=150)[0]["generated_text"]

class Classifier:
    def __init__(self, model, tokenizer):
        self.pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    def classify(self, text):
        return self.pipeline(text)[0]

class CodeHelper:
    def __init__(self, model_name="Salesforce/codegen-350M-multi"):
        self.pipeline = pipeline("text-generation", model=model_name)
    def complete(self, code_snippet):
        return self.pipeline(code_snippet, max_length=len(code_snippet.split()) + 100)[0]["generated_text"]