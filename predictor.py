import pickle
import gradio
import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
tf.config.set_visible_devices([], 'GPU')
from keras.models import load_model

from autocorrect import Speller
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from gensim.models import Word2Vec
from nltk.corpus import brown

from keras.preprocessing.text import Tokenizer

def averageWordVectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    n_words = 0

    for word in words:
        if word in vocabulary:
            n_words += 1
            feature_vector = np.add(feature_vector, model.wv[word])

    if n_words > 0:
        feature_vector = np.divide(feature_vector, n_words)

    return feature_vector

# moja klasa za preprocesiranje teksta
class TextPreprocessor:
    def __init__(self):
        self.spell = Speller(lang='en')
        self.lemmatizer = WordNetLemmatizer()

    def formalize(self, text):
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", text)

    def removeMentionsAndHashTags(self, text):
        hashtagPattern = re.compile(r"#[A-Za-z0-9_]+")
        mentionPattern = re.compile(r"@[A-Za-z0-9_]+")
        
        #Removing hashtags and mentions
        text = hashtagPattern.sub("", text)
        text = mentionPattern.sub("", text)

        return text

    def removeUrls(self, text):
        urlPatterns = [
            re.compile(r"https?://\S+"),
            re.compile(r"www\.\S+")
        ]
        #Removing links
        for pattern in urlPatterns:
            text = pattern.sub("", text)
        return text

    def removeNumbers(self, text):
        #removing numbers
        text = re.sub("[0-9]","", text)
        return text

    # mozda maknuti
    def removePunctuation(self, text):
        #removing punctuation
        text = re.sub(r'[^\w\s]','', text)
        return text

    def removeSingleQuotes(self, text):
        #removing single quotes
        text = re.sub(r"\'","", text)
        return text

    def removeStopwords(self, text):
        #removing stopwords
        text = text.split(' ')
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
        return text
    
    def preprocess(
            self,
            text,
            HASH_TAGS_MENTIONS=False,
            URLS=False,
            NUMBERS=False,
            PUNCTUATION=False,
            SINGLE_QUOTES=False,
            STOP_WORDS=True,
            SHORT_WORDS=False
        ):
            spell = Speller(lang='en')
            lemmatizer = WordNetLemmatizer()

            if(not HASH_TAGS_MENTIONS):
                text = self.removeMentionsAndHashTags(text)
            if(not URLS):
                text = self.removeUrls(text)
            if(not NUMBERS):
                text = self.removeNumbers(text)
            if(not STOP_WORDS):
                text = self.removeStopwords(text)
            if(not PUNCTUATION):
                text = self.removePunctuation(text)  
            if(not SINGLE_QUOTES):
                text = self.removeSingleQuotes(text)

            tokens = word_tokenize(text)
            tokens = [token.lower() for token in tokens]
            tokens = [self.formalize(token) for token in tokens]
            tokens = [spell(token) for token in tokens]
            tokens = [lemmatizer.lemmatize(token) for token in tokens if len(token) > 1]

            # remove short words
            if(not SHORT_WORDS):
                tokens = [token for token in tokens if len(token) > 2]

            return " ".join(tokens)

# klasa predictor koja sadrži sve modele
class Predictor:
    def __init__(self):
        self.bernoulli = None
        self.bernoulli_no_stops = None
        self.multinomial = None
        self.gaussian = None
        self.logistic_regression = None
        self.lstm = None
        self.bert = None
        self.bert_tokenizer = None

        self.count_vectorizer = None
        self.tfidf_vectorizer = None

        self.bernoulli_features_array = None

        self.vectorSize = 100
        self.word2vec = Word2Vec(
            sentences = [[word.lower() for word in sent] for sent in brown.sents()], 
            vector_size = self.vectorSize, 
            window = 5, 
            min_count = 2, 
            sg = 0
        )

        self.text_preprocessor = TextPreprocessor()

    # ucitavanje vektorizatora i arraya za bernoulli (koji koristi custom vektorizaciju)
    def loadCountVectorizer(self):
        if(self.count_vectorizer == None):
            self.count_vectorizer = pickle.load(open('outputs/count_vectorizer.pickle', 'rb'))
    
    def loadTfidfVectorizer(self):
        if(self.tfidf_vectorizer == None):
            self.tfidf_vectorizer = pickle.load(open('outputs/tfidf_vectorizer.pickle', 'rb'))

    def loadBernoulliArray(self):
        if(self.bernoulli_features_array == None):
            self.bernoulli_features_array = pickle.load(open('outputs/bernoulli_array.pickle', 'rb'))

    # workflow jednog predicta
    # Ukoliko nisi ucitao model, ucitaj ga i spremi ga
    # Ukoliko nisi ucitao potrebni vektorizator, ucitaj ga i spremi ga
    # Pozovi funkciju koja vraca predikciju

    def bernoulliFeatures(self, text):
        wordSet = set(word_tokenize(self.text_preprocessor.preprocess(text)))
        features = {}
        for word in self.bernoulli_features_array:
            features['contains({})'.format(word)] = (word in wordSet)
        return features

    def bernoulliPredict(self, text):
        if(self.bernoulli == None):
            with open('outputs/bernoulli_naive_bayes.pickle', 'rb') as f:
                self.bernoulli = pickle.load(f)
        self.loadBernoulliArray()
        return self.bernoulli.classify(self.bernoulliFeatures(text))
    
    def bernoulliNoStopsPredict(self, text):
        if(self.bernoulli_no_stops == None):
            with open('outputs/bernoulli_naive_bayes_no_stops.pickle', 'rb') as f:
                self.bernoulli_no_stops = pickle.load(f)
        self.loadBernoulliArray()
        return self.bernoulli_no_stops.classify(self.bernoulliFeatures(text))
    
    def multinomialPredict(self, text):
        if(self.multinomial == None):
            with open('outputs/multinomial_naive_bayes.pickle', 'rb') as f:
                self.multinomial = pickle.load(f)
        self.loadTfidfVectorizer()
        text_vectorized = self.tfidf_vectorizer.transform(
            [self.text_preprocessor.preprocess(text)]
        )
        return self.multinomial.predict(text_vectorized)[0]
    
    def gaussianPredict(self, text):
        if(self.gaussian == None):
            with open('outputs/gaussian_naive_bayes.pickle', 'rb') as f:
                self.gaussian = pickle.load(f)
        self.loadCountVectorizer()
        text_vectorized = self.count_vectorizer.transform(
            [self.text_preprocessor.preprocess(text)]
        ).toarray()
        return self.gaussian.predict(text_vectorized)[0]
    
    def logisticRegressionPredict(self, text):
        if(self.logistic_regression == None):
            with open('outputs/logistic_regression.pickle', 'rb') as f:
                self.logistic_regression = pickle.load(f)
        # use word2vec
        word_vectors = averageWordVectors(
            text.split(" "), 
            self.word2vec, 
            self.word2vec.wv.index_to_key, 
            self.vectorSize
        )
        return self.logistic_regression.predict([word_vectors])[0]
    
    # lstm ne radi
    def lstmPredict(self, text):
        if(self.lstm == None):
            self.lstm = load_model(
                'outputs/lstm.h5', 
                compile=False,
            )
            # self.lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3, epsilon=1e-08, clipnorm=1.0), 
            #   loss="binary_crossentropy",
            #   metrics=['accuracy'])
        tokenizer = Tokenizer()
        return self.lstm.predict(tokenizer.texts_to_sequences([text]))[0]
    
    def bertPredict(self, text):
        if(self.bert == None):
            self.bert = TFBertForSequenceClassification.from_pretrained('outputs/bert_model')
            self.bert_tokenizer = BertTokenizer.from_pretrained('outputs/bert_tokenizer')
        encoded_input = self.bert_tokenizer(text, return_tensors='tf')
        return np.argmax(self.bert(encoded_input)[0][0].numpy())