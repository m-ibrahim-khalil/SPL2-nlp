from pymagnitude import Magnitude

'''
    You have to download this two magnitude file from this drive link
    link: https://drive.google.com/drive/folders/1K2NvQxEIOb1cmOjmoBpQpOH0Q7v5H-St?usp=sharing
    create a new directory in your project folder
    upload the downloaded file in the created directory
    check base_dir variable and update accordingly
'''


class Vectors:
    def __init__(self):
        base_dir = "F://Pycharm Projects//SPL2-nlp//magnitude//"
        glove = Magnitude(base_dir + "glove-lemmatized.6B.100d.magnitude")
        fast_text = Magnitude(base_dir + "wiki-news-300d-1M.magnitude")
        self.vectors = Magnitude(glove, fast_text)

    def load_vectors(self):
        return self.vectors
