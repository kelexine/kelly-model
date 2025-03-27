# model.py
"""
Facade for Kelly AI.
Provides a unified interface to specialized components.
Uses dependency injection for easy testing and swapping of components.
"""
from logger_config import setup_logger

logger = setup_logger("model", "./logs/model.log")

class KellyError(Exception):
    pass

class Kelly:
    def __init__(self, tokenizer=None, classifier_model=None):
        try:
            logger.info("Initializing Kelly AI facade.")
            self.version = "kv1"
            self.developer = "kelexine"
            self.contact = "t.me/kelexine2"
            self.tokenizer = tokenizer
            self.classifier_model = classifier_model
            self.sentiment_analyzer = None
            self.summarizer = None
            self.qa_helper = None
            self.text_generator = None
            self.classifier = None
            self.code_helper = None
        except Exception as e:
            logger.error(f"Error initializing Kelly: {e}")
            raise KellyError(e)

    def analyze_sentiment(self, text):
        try:
            return self.sentiment_analyzer.analyze(text)
        except Exception as e:
            logger.error(f"Error in analyze_sentiment: {e}")
            return "Error analyzing sentiment."

    def summarize_text(self, text):
        try:
            return self.summarizer.summarize(text)
        except Exception as e:
            logger.error(f"Error in summarize_text: {e}")
            return "Error summarizing text."

    def answer_question(self, question, context):
        try:
            return self.qa_helper.answer(question, context)
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return "Error answering question."

    def generate_text(self, prompt):
        try:
            return self.text_generator.generate(prompt)
        except Exception as e:
            logger.error(f"Error in generate_text: {e}")
            return "Error generating text."

    def classify_text(self, text):
        try:
            return self.classifier.classify(text)
        except Exception as e:
            logger.error(f"Error in classify_text: {e}")
            return "Error classifying text."

    def code_completion(self, code_snippet):
        try:
            return self.code_helper.complete(code_snippet)
        except Exception as e:
            logger.error(f"Error in code_completion: {e}")
            return "Error completing code."

    def debug_code(self, code_snippet):
        try:
            compile(code_snippet, "<string>", "exec")
            return "No syntax errors detected."
        except Exception as e:
            logger.error(f"Error in debug_code: {e}")
            return "Error debugging code."

    def update_model_with_correction(self, input_text, correction, feedback):
        try:
            if feedback.lower() in ["yes", "correct", "true"]:
                from database import insert_knowledge
                knowledge_content = f"Input: {input_text}\nCorrection: {correction}"
                insert_knowledge(category="correction", source="user", content=knowledge_content)
                return "Model updated with new correction. Fine-tuning scheduled."
            else:
                return "Correction rejected based on feedback."
        except Exception as e:
            logger.error(f"Error in update_model_with_correction: {e}")
            return "Error updating model with correction."

    def search_internet(self, query):
        try:
            import os, requests
            SERPAPI_KEY = os.getenv("SERPAPI_KEY")
            if not SERPAPI_KEY:
                return "Search API not available."
            search_url = "https://serpapi.com/search"
            params = {"q": query, "api_key": SERPAPI_KEY, "num": 5}
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                results = response.json().get("organic_results", [])
                search_content = ""
                for result in results:
                    title = result.get("title", "No title")
                    snippet = result.get("snippet", "No snippet available.")
                    search_content += f"{title}: {snippet}\n"
                if not search_content:
                    return "No relevant information found online."
                return self.summarize_text(search_content)
            else:
                return f"Search failed with status code {response.status_code}."
        except Exception as e:
            logger.error(f"Error during search_internet: {e}")
            return f"Error during search: {str(e)}"

    def answer_sophisticated_question(self, question):
        try:
            context = ("Kelly AI integrates multiple transformers for diverse tasks and continual learning. "
                       "It is designed to be multi-purpose and adaptive.")
            answer = self.answer_question(question, context)
            if len(answer.split()) < 5:
                answer = self.search_internet(question)
            return answer
        except Exception as e:
            logger.error(f"Error in answer_sophisticated_question: {e}")
            return "Error answering sophisticated question."