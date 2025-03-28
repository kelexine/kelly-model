import pandas as pd
from logger_config import logger

class Preprocessor:
    def __init__(self):
        logger.debug("Initialized Preprocessor.")

    def analyze_dataset(self, dataset):
        """
        Analyze the dataset to determine:
          - Number of data fields
          - Modalities (text, numeric, etc.)
          - Task type hints based on metadata if available
          
        The dataset is expected to be a Pandas DataFrame or a dict that might include a 'metadata' key.
        """
        try:
            # If dataset is provided as a dict and contains metadata, use it.
            metadata = None
            if isinstance(dataset, dict):
                metadata = dataset.get("metadata", None)
                if "data" in dataset:
                    dataset = dataset["data"]
                else:
                    dataset = pd.DataFrame(dataset)
            elif hasattr(dataset, "metadata"):
                metadata = getattr(dataset, "metadata", None)

            if not isinstance(dataset, pd.DataFrame):
                dataset = pd.DataFrame(dataset)

            info = {
                'num_fields': len(dataset.columns),
                'fields': list(dataset.columns),
                'num_rows': len(dataset),
                'modalities': {}
            }

            # Use metadata if available to directly set modalities and task type
            if metadata is not None:
                logger.info("Using provided metadata for dataset analysis.")
                info['metadata'] = metadata
                # Assume metadata contains a key "modalities" if provided
                if "modalities" in metadata:
                    info['modalities'] = metadata["modalities"]
            else:
                # Otherwise, perform heuristic analysis
                for col in dataset.columns:
                    sample = dataset[col].dropna().head(10)
                    if sample.empty:
                        info['modalities'][col] = 'unknown'
                    elif sample.apply(lambda x: isinstance(x, str)).all():
                        info['modalities'][col] = 'text'
                    elif sample.apply(lambda x: isinstance(x, (int, float))).all():
                        info['modalities'][col] = 'numeric'
                    else:
                        info['modalities'][col] = 'mixed'

            # Further rigorous analysis could include checking for null distributions, unique counts, etc.
            logger.info("Dataset analysis complete: %s", info)
            return info
        except Exception as e:
            logger.error("Error in analyzing dataset", exc_info=True)
            raise e

if __name__ == '__main__':
    # Quick test for standalone execution
    data = {
        "data": {'text': ["Hello", "World"], 'value': [1, 2]},
        "metadata": {"modalities": {"text": "text", "value": "numeric"}, "task": "classification"}
    }
    preprocessor = Preprocessor()
    analysis = preprocessor.analyze_dataset(data)
    print(analysis)