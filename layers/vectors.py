from pymagnitude import Magnitude


class Vectors:
    def __init__(self):
        base_dir = "F://Pycharm Projects//bidaf-keras-master//data//magnitude//"
        glove = Magnitude(base_dir + "glove-lemmatized.6B.100d.magnitude")
        fast_text = Magnitude(base_dir + "wiki-news-300d-1M.magnitude")
        self.vectors = Magnitude(glove, fast_text)

    def load_vectors(self):
        return self.vectors
