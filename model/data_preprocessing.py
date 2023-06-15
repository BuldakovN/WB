import pandas as pd
import numpy as np

import re
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from navec import Navec

import pickle


vec_size = 300

navec = Navec.load("./navec_hudlit_v1_12B_500K_300d_100q.tar")
patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()

with open("./text_cl.pkl", 'rb') as f:
    text_classifier = pickle.load(f) 

with open("./tf_idf.pkl", 'rb') as f:
    tf_idf = pickle.load(f)

id1_data = pd.read_csv('../data/id1_data.csv')
id2_data = pd.read_csv('../data/id2_data.csv')

# классификация текстов
def text_classification(doc):
    local_falls = 0
    doc = re.sub(patterns, ' ', doc)
    vector = tf_idf.transform([doc])
    proba_0, proba_1 = text_classifier.predict_proba(vector)[0]
    for token in doc.split():
        token = re.sub("[^А-Яа-я]", '', token)
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            try:
                navec[token]
            except:
                local_falls += 1
    return pd.Series([proba_0, proba_1, local_falls], index=['text_0', 'text_1', 'falls_count'])


# извелечение фичей id
def id_features_extract(id1: str, id2: str) -> pd.Series:
    result = {'id1_pred': 0., 'id1_in_database': 0., 
              'id2_pred': 0., 'id2_in_database': 0.}
    if any(id1 == id1_data['id1']):
        result['id1_pred'] = id1_data[id1_data['id1']==id1]['id1_pred'].to_numpy()[0]
        result['id1_in_database'] = id1_data[id1_data['id1']==id1]['id1_in_database'].to_numpy()[0]

    if any(id2_data['id2'] == id2):
        result['id2_pred'] = id2_data[id2_data['id2']==id2]['id2_pred'].to_numpy()[0]
        result['id2_in_database'] = id2_data[id2_data['id2']==id2]['id2_in_database'].to_numpy()[0]
    return pd.Series(data=result)


# извлечение фичей текста
def text_features_extract(s: pd.Series) -> pd.Series:
    text = s['text']
    word_count = len(re.findall(r'[а-яА-Яa-zA-Z]+', text))
    # есть отзывы, состоящие из смайликов, пробелов или нижних подчеркиваний
    if word_count == 0:
            return pd.Series({
            'text_len': len(text),
            'words_count': 0,
            'sentence_count': 0,
            'number_percentage': 0,
            'caps_percentage': 0,
            'is_empty': 1
        })
    return pd.Series({
        'text_len': len(text),
        'words_count': word_count,
        'sentence_count': len(re.split(r'[.!?]+', text)),
        'number_percentage': len(re.findall(r'\d+', text)) / word_count,
        'caps_percentage': len(re.findall(r'[А-ЯA-Z]+', text)) / word_count,
        'is_empty': 0
    })


# дополнение данных
def get_features(series: pd.Series) -> pd.Series:
    # информация о классе текста
    text_class = text_classification(series['text'])
    # выделение фичей из id
    id_features = id_features_extract(series['id1'], series['id2'])
    # выделение фичей из текста
    text_features = text_features_extract(series)
    result = series.copy()
    result = result._append([text_features, id_features, text_class])
    return result


if __name__ == "__main__":
    data = pd.read_csv("../data/wb_school_task_2.csv.gzip", compression='gzip').head(5)
    a = data.apply(get_features, axis=1)
    print(a.columns)