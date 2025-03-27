# config.py
class Config:
    MODEL_DIR = "./kelly_finetuned"
    DEFAULT_TOKENIZER = "distilbert-base-uncased"
    
    class Training:
        BATCH_SIZE = 16
        LEARNING_RATE = 5e-5
        EPOCHS = 3
        LOGGING_STEPS = 10
        USE_DEEPSPEED = True
        DEEPSPEED_CONFIG = "deepspeed_config.json"
        
    class API:
        PORT = 5000
        DELAY_BEFORE_API = 10  # seconds
        RATE_LIMIT = "100 per hour"
        
    class ContinuousLearning:
        INTERVAL_SECONDS = 3600
        NUM_EPOCHS = 1
        BATCH_SIZE = 8
        EWC_LAMBDA = 0.4
        
    ENV = "prod"  # Options: dev, test, prod