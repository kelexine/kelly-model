# Kelly AI Model Project

## Overview
Kelly is a multi-purpose, adaptive AI model designed for production use. It continuously learns from diverse datasets and user corrections. Key features include:

1. **Distributed Training/Inference with DeepSpeed:**  
   The training module uses DeepSpeed to scale training and inference over multiple GPUs or nodes.

2. **Advanced Continual Learning:**  
   Incorporates a replay buffer and a full implementation of Elastic Weight Consolidation (EWC) to mitigate catastrophic forgetting.

3. **Modularized Components & Dependency Injection:**  
   Specialized model components (sentiment analysis, summarization, QA, text generation, code assistance) are implemented in separate classes and assembled via dependency injection.

4. **Enhanced Monitoring & Logging with MLflow:**  
   Training runs are tracked with MLflow for experiment tracking and structured logging.

5. **Advanced Offline Retrieval:**  
   Uses FAISS for robust vector-based retrieval of offline knowledge to supplement responses.

6. **Production-ready API:**  
   The Flask API is secured with rate limiting (via flask-limiter) and is designed for deployment with Gunicorn.

## Setup Instructions
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt