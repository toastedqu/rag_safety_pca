from sentence_transformers import SentenceTransformer


class ModelLoader:
    """
    Load SentenceTransformer model

    Args:
        model_path (str): path to the model

    Attributes:
        model (SentenceTransformer): SentenceTransformer model
    """

    def __init__(self, model_path: str):
        self.model = SentenceTransformer(model_path)

    def encode(self, sentences) -> list:
        return self.model.encode(sentences)
