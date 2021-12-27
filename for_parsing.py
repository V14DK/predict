import re
import os
import string
import numpy as np
import nltk
import pymorphy2
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import Phrases, LsiModel, TfidfModel
from gensim.models.phrases import Phraser
from sklearn.base import BaseEstimator, TransformerMixin


class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stopwords = set(stopwords.words('russian'))
        self.norm = pymorphy2.MorphAnalyzer().normal_forms

    def __remove_email(self, text):
        email = re.compile(r'\S+@\S+\.\S+$')
        return email.sub(r' ', text)

    def __remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r' ', text)

    def __remove_html(self, text):
        html = re.compile(r'<.*?>')
        return html.sub(r' ', text)

    def __remove_mail(self, text):
        mail = re.compile(r'^([a-z0-9_\.-]+)@([a-z0-9_\.-]+)\.([a-z\.]{2,6})$')
        return mail.sub(r' ', text)

    def __remove_emoji(self, text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r' ', text)

    def __mytokenize(self, text):
        spec_chars = string.punctuation + '\n\xa0«»\t—…'
        text = self.__remove_email(text)
        text = self.__remove_html(text)
        text = self.__remove_URL(text)
        text = self.__remove_emoji(text)
        text = self.__remove_mail(text)
        text = text.lower()
        text = re.sub("[^а-яёйa-z0-9]", " ", text)
        text = re.sub("\s+", " ", text)
        tbl = text.maketrans('', '', spec_chars)
        text = text.translate(tbl)
        text = nltk.word_tokenize(text)
        text = [self.norm(word)[0] for word in text if word.isalpha() and word not in self.stopwords]
        return text

    def fit(self, X, y=None):
        return self

    def transform(self, document): #documents
        return [self.__mytokenize(document)] #for document in documents] #document['Text'] для kaggle


class GensimVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, path=None):
        self.path = path
        self.dict = None
        self.__load()

    def __load(self):
        if self.path:
            if os.path.exists(self.path):
                self.dict = Dictionary.load(self.path)

    def fit(self, documents):
        bigram = Phrases(documents, min_count=10, threshold=2)
        self.bigram_phraser = Phraser(bigram)
        bigram_token = []
        for sent in documents:
            bigram_token.append(self.bigram_phraser[sent])
        if not self.dict:
            self.dict = Dictionary(bigram_token)
            self.dict.filter_extremes(no_below=3, no_above=0.8)
            self.dict.save('new_dict')
        self.corpus = [self.dict.doc2bow(line) for line in bigram_token]
        self.tf_model = TfidfModel(self.corpus)
        return self

    def transform(self, documents):
        corpus = [self.dict.doc2bow(text) for text in documents]
        return self.tf_model[corpus]


class GensimLsi(BaseEstimator, TransformerMixin):
    def __init__(self, num_topics, path_dict=None, path_model=None):
        self.path_dict = path_dict
        self.path_model = path_model
        self.num_topics = num_topics
        self.load()

    def load(self):
        self.dict = Dictionary.load(self.path_dict)
        if self.path_model:
            if os.path.exists(self.path_model):
                self.model = LsiModel.load(self.path_model)

    def save(self):
        self.model.save(self.path)

    def make_vec(self, row_matrix, num_top):
        matrix = np.zeros((len(row_matrix), num_top))
        for i, row in enumerate(row_matrix):
            matrix[i, list(map(lambda tup: tup[0], row))] = list(map(lambda tup: tup[1], row))
        return matrix

    def fit(self, documents):
        self.model = LsiModel(documents, id2word=self.dict, num_topics=self.num_topics)
        return self

    def transform(self, documents):
        corpus = self.model[documents]
        documents = self.make_vec(corpus, self.model.num_topics)
        return documents
