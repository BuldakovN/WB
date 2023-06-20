from catboost import CatBoostClassifier
import pandas as pd
from data_preprocessing import get_features

class FakeReviewsClassifier:
    def __init__(self, model_path='./catboost_full.model', get_features=True):
        self.catboost_model: CatBoostClassifier = CatBoostClassifier().load_model(model_path)
        self.feature_names_ = self.catboost_model.feature_names_
        self.get_features = get_features
        
    def get_params(self):
        return self.catboost_model.get_params()

    def get_features(self, data: pd.Series):
        return get_features(data)

    def preproccess(self, data: pd.Series, get_features):
        data = data.drop(['f2', 'f4'], errors='ignore')
        if get_features:
            data = self.get_features(data)
            data = data.drop(['id1', 'id2', 'id3', 'text', 'label'])   
        else:
            data = data[self.feature_names_]
        return data
    
    # предсказание по id
    def id_predict(self, data: pd.Series, threshold):
        if not ('id1_in_database' in data.index):
            return None
        if data['id1_in_database']:
            return 1 if data['id1_pred'] >= threshold else 0
        return None
        
        
    def predict(self, data: pd.Series, 
                get_features=None, 
                id_predict = True,
                threshold = 0.5):
        if get_features is None:
            get_features = self.get_features
        data = self.preproccess(data, get_features)
        # если пользователь оставляет очень много фейков или наоборот – честных отзывов
        predict = None
        if id_predict:
            predict = self.id_predict(data, threshold=threshold)
        if predict is None:
            self.catboost_model.set_probability_threshold(threshold)
            predict = self.catboost_model.predict(data)
        return predict