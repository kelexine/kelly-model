# advanced_preprocessing.py
"""
Advanced data preprocessing for Kelly AI.
Inspects dataset features (via Hugging Face Datasets) and dynamically configures tokenization.
Configuration dicts (with "task" and "input_fields") may be supplied; otherwise, auto-detection is performed.
"""
from datasets import load_dataset
from transformers import AutoTokenizer
from logger_config import setup_logger

logger = setup_logger("advanced_preprocessing", "./logs/advanced_preprocessing.log")
DEFAULT_TOKENIZER_MODEL = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_MODEL)

def inspect_features(dataset):
    feature_dict = {}
    for key, feature in dataset.features.items():
        feature_dict[key] = feature.__class__.__name__
    logger.info(f"Dataset features: {feature_dict}")
    return feature_dict

def combine_fields(example, fields, separator=" "):
    return separator.join(example[field] for field in fields if field in example and example[field] is not None)

def preprocess_dataset_dynamic(dataset_name, config=None, max_length=128):
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    if "train" in dataset and "test" in dataset:
        train_ds = dataset["train"]
        val_ds = dataset["test"]
    elif "train" in dataset:
        logger.info("Only 'train' split found; splitting into train/test.")
        split_ds = dataset["train"].train_test_split(test_size=0.1)
        train_ds = split_ds["train"]
        val_ds = split_ds["test"]
    else:
        raise ValueError("Dataset does not contain a 'train' split.")
    features = inspect_features(train_ds)
    if config is None:
        common_fields = ["text", "sentence", "content", "document", "review"]
        for field in common_fields:
            if field in features and features[field] == "Value":
                config = {"task": "classification", "input_fields": [field]}
                logger.info(f"Auto-detected input field: {field}")
                break
        if config is None:
            raise ValueError("Could not auto-detect an input field.")
    input_fields = config.get("input_fields", [])
    task = config.get("task", "classification")
    def preprocess_fn(examples):
        if task.lower() == "qa" and len(input_fields) >= 2:
            questions = examples.get(input_fields[0], [""] * len(next(iter(examples.values()))))
            contexts = examples.get(input_fields[1], [""] * len(next(iter(examples.values()))))
            return tokenizer(questions, contexts, truncation=True, padding="max_length", max_length=max_length)
        elif len(input_fields) == 1:
            texts = examples.get(input_fields[0], [""] * len(next(iter(examples.values()))))
            return tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        else:
            combined = [combine_fields({field: examples[field][i] for field in input_fields}, input_fields)
                        for i in range(len(next(iter(examples.values()))))]
            return tokenizer(combined, truncation=True, padding="max_length", max_length=max_length)
    remove_cols = input_fields
    logger.info(f"Tokenizing train split, removing columns: {remove_cols}")
    train_ds = train_ds.map(preprocess_fn, batched=True, remove_columns=remove_cols)
    logger.info("Tokenizing validation split...")
    val_ds = val_ds.map(preprocess_fn, batched=True, remove_columns=remove_cols)
    logger.info("Dynamic preprocessing complete.")
    return train_ds, val_ds