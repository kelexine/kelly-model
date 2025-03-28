# Kelly AI Model (kv1)

## Overview

The **Kelly AI Model** is a production-ready solution designed for robust, scalable, and efficient deployment in real-world environments. This implementation leverages DistilBERT as the backbone for sequence classification tasks and includes a rigorous dataset analysis pipeline that utilizes metadata when available. The model supports both fine-tuning and inference, with an interactive mode that automatically loads a fine-tuned model if available. Otherwise, it defaults to training using the base model.

## Features

- **Modular Architecture:**  
  - Separated modules for logging, preprocessing, model building, training, evaluation, API, and database support.
  
- **Base Model (Updated):**
  - Now Uses DistilBERT (`distilbert-base-uncased`) as the base model for sequence classification.
  
- **Rigorous Dataset Analysis:**  
  - The preprocessor examines the dataset structure and checks for a metadata section to determine modalities and task type. If metadata is provided, it directly utilizes that information; otherwise, heuristic checks are performed.

- **Fine-Tuning and Saving:**  
  - Fine-tunes the model using Hugging Face’s Trainer API.
  - Saves the fine-tuned model weights, configuration, and tokenizer files after training, ensuring easy reload for further training or inference.

- **Interactive Mode:**  
  - Automatically loads the fine-tuned model for inference if available. Otherwise, it enters training mode.
  - Provides an interactive command-line interface for real-time predictions.

- **REST API for Inference and Admin Tasks:**  
  - Flask API endpoints with basic rate limiting.
  - Admin endpoint for dataset uploads (secured with token-based authentication).
  - Prediction endpoint for inference requests.

- **Persistent Storage:**  
  - Uses SQLite for database operations to store datasets and rehearsal flags.

## Setup Instructions
1. Create and activate a virtual environment.
   ```bash
   python -m venv myvenv #or whatever name you like
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Clone the Repo
   ```bash
   git clone https://github.com/kelexine/kelly-model
   cd kelly-model
   ```
4. Start the AI
   ```bash
   python kelly.py
   ```

## Training the Model

1. Prepare Your Dataset:

Format your dataset as a CSV or JSON.

Include a "metadata" key (if available) to explicitly define modalities and task type. Otherwise, the system will apply heuristic checks.



2. Run the Main Driver:
```bash
python kelly.py
```

**Interactive Mode:** If a fine-tuned model exists in ./kelly_finetuned, it is automatically loaded for inference.

**Training Mode:** If no fine-tuned model is found, the system will enter training mode, fine-tune the model, save all required files, and then switch to interactive mode.




## Using the Model

**Interactive Mode**

The command-line interactive mode allows you to input text and receive predictions in real time.

The inference process tokenizes input text and outputs a predicted label along with confidence scores.


## REST API Endpoints

1. Admin Dataset Loader:

Endpoint: /admin/load_dataset

Method: GET

Parameters:

token: Admin token (default is supersecrettoken)

url: URL of the dataset (expects JSON format)


**Usage Example:**
```bash
curl "http://localhost:5000/admin/load_dataset?token=supersecrettoken&url=https://huggingface.co/dataset-url"
```


2. Prediction Endpoint:

Endpoint: /predict

Method: POST

**Payload Example:**
```json
{
  "text": "Your sample text here"
}
```
**Usage Example:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "Sample input"}' http://localhost:5000/predict
```




## Example Use Cases:

**Sentiment Analysis:** Classify text as positive or negative sentiment using the fine-tuned DistilBERT model.

**Content Moderation:** Filter harmful content by classifying text segments appropriately.

**Interactive Chatbot:** Utilize the interactive mode to build conversational agents that integrate real-time predictions with dynamic dataset updates.


## Development Process and Assumptions

Step-by-Step Overview:

1. Logger Setup: Custom logger with colored outputs facilitates debugging and production logging.


2. Dataset Preprocessing: Enhanced analysis that supports both heuristic checks and metadata-driven approaches.


3. Model Building: DistilBERT is used as the base model. The system supports loading a fine-tuned model if available.


4. Training & Fine-Tuning: Fine-tuning is performed using Hugging Face’s Trainer API. The fine-tuned model (weights, configuration, tokenizer) is saved for future use.


5. Interactive & API Modes: Post-training, the model is loaded for inference in interactive mode and exposed via a Flask API.


6. Database Integration: SQLite is used to store persistent data for datasets and rehearsal flags.



## Assumptions:

Datasets are provided in standard formats (CSV/JSON) and can be processed into Pandas DataFrames.

Metadata (if provided) will include keys such as "modalities" and "task" to guide data handling.

Basic admin authentication (token-based) is acceptable for demonstration purposes.

Rate limiting is implemented in-memory; a production system may require a distributed rate limiter.

**With these components, the Kelly AI Model is ready for further development, integration, and deployment in a real-world production environment.**

## NOTE
 **Still in Early Stage of Development**

 **Contributions are Welcomed**