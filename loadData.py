import pandas as pd
import re
import string


class LoadData:
    def __init__(self):
        self.file_path = "IMDB Dataset.csv"
        self.sample_size = 1000
        self.df = None

        self.stop_words = {
            "the", "and", "is", "in", "to", "of", "a", "an", "it", "this",
            "that", "was", "for", "on", "with", "as", "but", "at", "by",
            "from", "or", "be", "are", "were", "has", "have", "had", "i",
            "you", "he", "she", "they", "we", "my", "your", "his", "her",
            "their", "our", "its"
        }

    def load_data(self):
        df = pd.read_csv(self.file_path)

        # Map sentiment labels
        df["label"] = df["sentiment"].map({
            "positive": 1,
            "negative": 0
        })

        # Sample data for efficiency
        df = df.sample(self.sample_size, random_state=42).reset_index(drop=True)

        # Clean text
        df["clean_review"] = df["review"].apply(self.clean_text)

        self.df = df
        return df

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"<.*?>", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def get_data(self):
        return self.df