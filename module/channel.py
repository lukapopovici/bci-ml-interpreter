
import pandas as pd

class ChannelState:
    TRAINING = "training"
    CLASSIFYING = "classifying"

    def __init__(self, state):
        if state not in (self.TRAINING, self.CLASSIFYING):
            raise ValueError(f"Invalid state: {state}")
        self.state = state

    def __repr__(self):
        return f"ChannelState(state='{self.state}')"

class Channel:
    def __init__(self, identifier: str, model=None, state=None):
        self.identifier = identifier
        self.model = model
        self.data = []
        self.state = ChannelState(state or ChannelState.TRAINING)

    def load_data(self, csv_path: str):
        df = pd.read_csv(csv_path)
        if self.identifier in df.columns:
            self.data = df[self.identifier]
        else:
            raise ValueError(f"Column '{self.identifier}' not found in CSV.")

    def add_data_from_csv(self, csv_path: str, chunk_size: int = 1):
        """
        Continuously load data from the specified column in the CSV file in chunks.
        """
        for chunk in pd.read_csv(csv_path, usecols=[self.identifier], chunksize=chunk_size):
            self.data.extend(chunk[self.identifier].tolist())
            yield chunk[self.identifier]

    def predict_verdict(self, *args, **kwargs):
        if self.model is None or self.data is None:
            raise ValueError("Model or data not set.")
        # Example: model should have a predict method
        return self.model.predict(self.data, *args, **kwargs)
