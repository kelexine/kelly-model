# dependency_injection.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model_components import SentimentAnalyzer, Summarizer, QAHelper, TextGenerator, Classifier, CodeHelper
from model import Kelly

def create_kelly(tokenizer_name="distilbert-base-uncased", classifier_model_name="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_model_name, num_labels=4)
    sentiment_analyzer = SentimentAnalyzer()
    summarizer = Summarizer()
    qa_helper = QAHelper()
    text_generator = TextGenerator()
    classifier = Classifier(classifier_model, tokenizer)
    code_helper = CodeHelper()
    kelly = Kelly(tokenizer=tokenizer, classifier_model=classifier_model)
    kelly.sentiment_analyzer = sentiment_analyzer
    kelly.summarizer = summarizer
    kelly.qa_helper = qa_helper
    kelly.text_generator = text_generator
    kelly.classifier = classifier
    kelly.code_helper = code_helper
    return kelly