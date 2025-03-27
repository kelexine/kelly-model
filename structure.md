kelly/
├── logger_config.py              # Logger configuration module with color coded output for types information (Info, debug, warning, error, critical)
├── preprocessing.py              # Dynamic, task-aware preprocessing module to determine what type of dataset is being passed (how many data fields, what modalities, what tasks and sub tasks, and general information about the dataset)
├── model.py                      # Back-bone of the AI model. should use the smallest "BERT" model for base.
├── training.py                   # Fine-tuning module, should handle the training and fine-tuning
├── evaluation.py                 # Evaluation module for model performance
├── flask_api.py                  # Flask API endpoints with rate limiting (should allow loading of datasets from Hugging Face URLs for training "can only be used by admin"), should also have a prediction inference call for prediction.
├── database.py                   # Database module with persistent memory and rehearsal flags, also offline support
├── kelly.py                      # Main driver script (The brain of the AI model), should support sequential training, interactive mode (should check if there's a pre trained model in the /kelly_finetuned dir for launching interactive mode, else goes into training mode directly), handles the starting of the flask api module after 20 seconds. Model Name: kelly, Base Version Code: kv1
├── requirements.txt              # Contains Required packages for the AI to work 
└── README.md                     # Project in dept documentation

1. The markdown file contains the skeletal structure of an AI model am working on, help me write the code for the AI (should be production ready).
2. It should support loading datasets from Hugging Face URLs via api calls, and also loading datasets (csv, json, etc) manually via API calls. The model name should be kelly, and a base version of kv1.
3. AI to occasionally scrap the wikipedia website for data and train the model using the data.
4. There should be an interactive mode for sending prompts and receiving responses (inference).
NOTE: I have a resource contraint (am trying to train on google colab free tier), so instead of going full throttle, it should learn, adapt and improve overtime. The model should be able to adapt in the system (if a GPU "cuda" or TPU is available it should use a more complex training algorithm, else if only CPU, should use a less complex but efficient algorithm)