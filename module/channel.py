class Channel:
    def __init__(self, identifier: str):
        self.identifier = identifier

    def __repr__(self):
        return f"Channel(identifier='{self.identifier}')"
