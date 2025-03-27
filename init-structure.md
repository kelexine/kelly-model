kelly/
├── logger_config.py              # Logger configuration module with color coded output for types information (Info, debug, warning, error, critical)
├── status.py                     # Global training status and time estimation module.
├── advanced_preprocessing.py     # The brain of the AI model. Dynamic, task-aware preprocessing module to determine what type of dataset is being passed (how many data fields, what modalities, what tasks and sub tasks, and general information about the datasetsl)
├── preprocessing.py            # should take over from advanced_preprocessing.py (handles tokenization and general procedures befor training.
├── dependency_injection.py       # DI factory for constructing the Kelly facade
├── ewc.py                        # Implementation of production ready Elastic Weight Consolidation (EWC)
├── ewc_trainer.py                # Custom Trainer that adds an EWC penalty
├── model.py                      # Kelly facade using dependency injection to incorporate specialized components, back-bone of the AI model.
├── model_components.py           # Specialized model components (sentiment,text generation, summarization, QA, Code generation and debugging, and many more AI features.)
├── advanced_offline_retrieval.py # FAISS-based offline retrieval module for accessing the offline knowledge base
├── training.py                   # Fine-tuning module using Hugging Face Trainer with DeepSpeed and MLflow integration
├── evaluation.py                 # Evaluation module for model performance
├── continuous_learning.py        # Rehearsal module with replay buffer and EWC integration for rehearsal of past training.
├── flask_api.py                  # Flask API endpoints with rate limiting (should allow loading of datasets from Hugging Face URLs for training "can only be used by admin"), should also have a prediction inference call for interactions with the AI model.
├── database.py                   # Database module with persistent memory and rehearsal flags, also offline support
├── config.py                     # Centralized configuration settings module for the AI model.
├── kelly.py                      # Main driver script, should support sequential training, continuous learning, interactive mode (should check if there's a pre trained model in the /kelly_finetuned dir for launching interactive mode, else goes into training mode directly), (should be able to handle multiple users), handles the starting of the flask api module after 20 seconds. Model Name: kelly, Base Version Code: kv1.
├── deepspeed_config.json         # DeepSpeed configuration file
├── requirements.txt              # Contains Required packages for the AI to work 
└── README.md                     # Project in dept documentation

The markdown file contains the skeletal structure of an AI model am working on, help me write the code for the AI (should be production ready), it should support loading datasets from Hugging Face URLs via api calls, and also loading datasets (csv, json, etc) manually via API calls. The model name should be kelly, and a base version of kv1, if possible add the ability for the AI to occasionally scrap the web for data and train the model using the data. There should be an interactive mode for sending prompts and receiving responses.
NOTE: I have a resource contraint (am trying to train on google colab free tier), so instead of going full throttle, it should learn, adapt and improve overtime. the model should be able to adapt in the system (if a GPU "cuda" or TPU is available it should use a more complex and efficient training algorithm, else if only CPU, should use a less complex algorithm, but should be as efficient as possible)