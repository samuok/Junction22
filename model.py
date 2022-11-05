from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import hinge_loss
import numpy as np
import pandas as pd


class Machinelearning:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def split_database(self, dataframe, features, columns):
        # We define the labels and features of the dataset and split them accordingly
        X_columns = list(filter(lambda x: x != label, dataframe.columns))
        y_columns = [label]

        # We physically split the data
        X_df = dataframe.drop(columns=y_columns)
        Y_df = dataframe.drop(columns=X_columns)

        # We format the data so it can be used by the model
        X = X_df.copy().to_numpy()
        y = Y_df.copy().to_numpy().reshape((-1,))
        return X, y

    def data_split(self, X, y, test_size):
        # Here we split the data first into 60% training and 40% testing and valitating.
        X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Here we further split the data into a validation and training set
        X_test, X_validation, y_test, y_validation = train_test_split(X_hold, y_hold, test_size=0.5,random_state=random_state)

        return [X_train, y_train, X_validation, y_validation, X_test, y_test]

    def find_parameters(self, X_train, y_train)

        # Here we format the X and y data so we can try and find the optimal weights for our training.
        #X_grid = np.concatenate((X_train, X_test))
        #y_grid = np.concatenate((y_train, y_test))

        regularisation_terms = np.linspace(0.00001, 5, 20)
        parameters = {'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'C': regularisation_terms}
        svc = SVC(random_state=random_state, kernel='rbf')
        grid = GridSearchCV(svc, parameters, n_jobs=1, verbose=2, scoring='f1_weighted')
        grid.fit(X_train,y_train)
        print(grid.best_score, grid.best_params_)
        return grid.best_params_


    def train_model(self, X_train, y_train, C=1, gamma=0.0001, iterations=1000 , random=None):
        SVC_ml = SVC(C=C, gamma=0.0001, max_iter=iterations, random_state=random)
        training_predict_SVC = SVC_ml.predict(X_train)
        training_predict_SVC_binary = pd.get_dummies(SVC_ml.predict(X_train))
        training_error_SVC = hinge_loss(y_train, training_predict_SVC_binary)
        print(training_error_SVC)

        return SVC_ml

    def predict_model(self, model, x, y, dataName = ""):
        predict_SVC  = model.predict(x)
        predict_SVC_binary =  pd.get_dummies(model.predict(x))

        error_SVC = hinge_loss(y, predict_SVC_binary)
        accuracy_SVC = accuracy_score(y, predict_SVC)
        f1_score_SVC = f1_score(y, predict_SVC, average='weighted')
        print(dataName + " error SVC",error_SVC)
        print(dataName + " accuracy SVC", accuracy_SVC)
        print(dataName + " f1_score SVC", f1_score_SVC)









