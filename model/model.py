from catboost import CatBoostClassifier
import pandas as pd
from data_preprocessing import get_features

class FakeReviewsClassifier:
    def __init__(self, id_mode='with_id'):
        self.id_mode = id_mode
        assert id_mode in ['without_id', 'with_id'], KeyError()
        self.catboost_model = CatBoostClassifier().load_model(f'./catboost_{id_mode}.model')

    def get_features(self, data: pd.Series):
        return get_features(data)

    def preproccess(self, data: pd.Series, get_features):
        if get_features:
             data = self.get_features(data)
        data = data.drop(['id1', 'id2', 'id3', 'text', 'label'], errors='ignore')   
        if self.id_mode == 'without_id':
            data = data.drop(['id1_0', 'id1_1', 'id2_0', 'id2_1'], errors='ignore')  
        return data
    
    # предсказание по 
    def id_predict(self, data, id_threshold):
        if self.id_mode == 'with_id':
            id1_0, id1_1 = data['id1_0'], data['id1_1']
            if abs(id1_0 - id1_1) >= id_threshold:
                return 0 if id1_0 > id1_1 else 1
            id2_0, id2_1 = data['id2_0'], data['id2_1']
            if abs(id2_0 - id2_1) >= id_threshold:
                return 0 if id2_0 > id2_1 else 1
        else:
            return None
        
    def predict(self, data: pd.Series, 
                get_features=True, 
                id_predict=False, 
                id_threshold = 0.5):
        data = self.preproccess(data, get_features)
        # если пользователь оставляет очень много фейков или наоборот – честных отзывов
        predict = None
        if id_predict and self.id_mode == 'with_id':
            predict = self.id_predict(data, id_threshold)
        if predict is None:
            predict = self.catboost_model.predict(data)
        return predict