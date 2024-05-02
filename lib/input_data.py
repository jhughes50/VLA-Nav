"""
    CIS 6200 -- Deep Learning Final Project
    Input data structure
    April 2024
"""

class InputData:

    def __init__(self, text, image, path):

        self.text_ = text
        self.image_ = image
        self.path_ = path

    @property
    def text(self):
        return self.text_

    @property
    def image(self):
        return self.image_

    @property
    def path(self):
        return self.path_
