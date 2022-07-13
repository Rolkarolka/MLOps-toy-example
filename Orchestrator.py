
from clothPredictionModule import FashionNetAdapter
from inputUnderstandingModule import InputUnderstandingAdapter

class Orchestrator:
    def __init__(self):
        self.fashionNet = FashionNetAdapter()
        self.inputUnderstanding = InputUnderstandingAdapter()

    def run(self, user_input):
        if user_input is not None:
            processed_input = self.inputUnderstanding.process(user_input)
            prediction = self.fashionNet.predict(processed_input)
            return prediction
