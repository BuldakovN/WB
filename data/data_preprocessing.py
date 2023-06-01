import pandas as pd
import numpy as np

import re
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from navec import Navec

vec_size = 300

navec = Navec.load("./navec_hudlit_v1_12B_500K_300d_100q.tar")
patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()


# векторизация текста
# лемматизация предложений

def vectorize(doc: str) -> pd.Series:
    global falls
    doc = re.sub(patterns, ' ', doc)
    vector = np.array([0.]*vec_size)
    for token in doc.split():
        token = re.sub("[^А-Яа-я]", '', token)
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            try:
                vector += navec[token]
            except KeyError:
                pass
    return pd.Series(vector, index=('emb_'+str(i) for i in range(vec_size)))


# получение фичей id
def id_features_extract(id1: str, id2: str) -> pd.Series:
    id_data = pd.read_csv('./id1_data.csv')
    result = {'id1_0': 0., 'id1_1': 0., 
              'id2_0': 0., 'id_1':1}
    if id1 in id_data['id1']:
        result['id1_0'] = id_data[id_data['id1']==id1]['id1_0']
        result['id1_1'] = id_data[id_data['id1']==id1]['id1_1']

    id_data = pd.read_csv('./id2_data.csv')
    if id1 in id_data['id2']:
        result['id2_0'] = id_data[id_data['id2']==id2]['id2_0']
        result['id2_1'] = id_data[id_data['id2']==id2]['id2_1']
    return pd.Series(result)


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
def get_features_series(series: pd.Series) -> pd.Series:
    text_embedding = vectorize(series['text'])
    id_features = id_features_extract(series['id1'], series['id2'])
    text_features = text_features_extract(series)
    result = series.copy()
    result = result._append([text_embedding, id_features, text_features])
    return result


if __name__ == "__main__":
    data = pd.read_csv("./wb_school_task_2.csv.gzip", compression='gzip').loc[:10]
    a = data.apply(get_features_series, axis=1)
    print(a.columns)