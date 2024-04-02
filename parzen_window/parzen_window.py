import numpy as np
from typing import Literal
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

class parzen_window(BaseEstimator,RegressorMixin):
    def __init__(self, h:float=0.5,kernel:Literal['gaus', 'epanichenkov', 'square',
                                            'triangle','rectangle']  = "gaus"):
        self.h = h 
        self.kernel=kernel

    def __epanichenkov(self,r):
        res=[]
        for i in r:
            if abs(i)<=1:
                res.append((0.75)*(1-(i**2)))
            else:
                res.append(0)
        return np.array(res)
    
    def __square(self,r):
        res=[]
        for i in r:
            if abs(i)<=1:
                res.append((15.0/16.0)*((1-(i**2))**2))
            else:
                res.append(0)
        return np.array(res)
    
    def __triangle(self,r):
        res=[]
        for i in r:
            if abs(i)<=1:
                res.append(1-abs(i))
            else:
                res.append(0)
        return np.array(res)
    
    def __rectangle(self,r):
        res=[]
        for i in r:
            if abs(i)<=1:
                res.append(0.5)
            else:
                res.append(0)
        return np.array(res)
    
    def __set_kernel(self,distance:float):
        match self.kernel:
            case 'gaus': return np.exp(-0.5 * (distance / self.h) ** 2) / (np.sqrt(2 * np.pi) * self.h)
            case 'epanichenkov': return  self.__epanichenkov(distance / self.h)
            case 'square': return self.__square(distance / self.h)
            case 'triangle': return self.__triangle(distance / self.h)
            case 'rectangle': return self.__rectangle(distance / self.h)

        
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)
        
    
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            # Вычисляем расстояние от тестового объекта до всех точек обучающей выборки
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Вычисляем ядерную оценку для каждого расстояния
            kernel_values = self.__set_kernel(distance=distances)
            # Вычисляем взвешенную сумму ядерных оценок для каждого класса
            class_scores = {c: np.sum(kernel_values[self.y_train == c]) for c in self.classes}
            # Предсказываем класс с наибольшей взвешенной суммой ядерных оценок
            prediction = max(class_scores, key=class_scores.get)
            predictions.append(prediction)
        return np.array(predictions)