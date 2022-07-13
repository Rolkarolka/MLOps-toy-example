
from inputUnderstandingModule.InputUnderstanding import InputUnderstanding


class InputUnderstandingAdapter:
    def __init__(self):
        self.input_understanding = InputUnderstanding()

    def process(self, user_input):
        img_num = int(user_input)
        return self.input_understanding.get(img_num)