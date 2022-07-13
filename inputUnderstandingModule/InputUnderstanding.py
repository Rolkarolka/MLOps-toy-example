
from torchvision import datasets

class InputUnderstanding:
    def __init__(self):
        self.user_images = self.load_user_images()
    
    def load_user_images(self):
        return datasets.FashionMNIST('./data', download=True, train=False)
    
    def get(self, img_num):
        return self.user_images[img_num]