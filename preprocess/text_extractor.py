"""
    CIS 6200 -- Deep Learning Final Project
    Code to extract text from the RxR dataset
    Spring 2024 
"""
    
class TextExtractor:

    def get_text(self, subguide):
        if subguide['language'] == 'en-IN' or subguide['language'] == 'en-US':
            return subguide['instruction'] 
        else:
            return None
