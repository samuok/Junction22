import mariadb
import pandas as pd
import sys
import random
import numpy as n
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt

import sqlalchemy
from sqlalchemy import create_engine
from model import Machinelearning
from scipy.stats import truncnorm


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class SubWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        loadUi("config.ui", self)
        self.conn = 0
        self.cur = 0
        self.connect_to_dataBase()
        self.model = 0
        self.ml = Machinelearning()
        self.train_model.clicked.connect(self.train)
        self.split.clicked.connect(self.define_set)
        self.load_data.clicked.connect(self.load_current_model)
        self.save_data.clicked.connect(self.save_current_model)
        self.predict.clicked.connect(self.predict_accuracy)
        self.clear_data.clicked.connect(self.delete_data)
        self.data = self.get_data()
        self.add_data.clicked.connect(self.make_data)
        self.x_train = 0
        self.y_train = 0
        self.x_validation = 0
        self.y_validation = 0
        self.x_test = 0
        self.y_test = 0

    def get_data(self):
        engine = sqlalchemy.create_engine("mariadb+mariadbconnector://kayttis:salis@127.0.0.1:3306/junction")
        df = pd.read_sql("SELECT * FROM junction", engine)
        return df

    def define_set(self):
        x, y = self.ml.split_database(self.data, "terapeutti")
        self.x_train, self.y_train, self.x_validation, self.y_validation, self.x_test , self.y_test = self.ml.data_split(x, y, float(self.split_val.toPlainText()))

    def train(self):
        self.model = self.ml.train_model(self.x_train, self.y_train)

    def load_current_model(self):
        self.model = self.ml.load_model("finalized_model.sav")

    def save_current_model(self):
        self.ml.save_model(self.model)

    def predict_accuracy(self):
        val = self.ml.predict_model(self.model, self.x_validation, self.y_validation)
        test = self.ml.predict_model(self.model, self.x_validation, self.y_validation)

        self.val_error.setText("Validation error: "+ str(round(val[0] * 100, 2)) + "%")
        self.val_acc.setText("Validation accuracy: "+ str(round(val[1] * 100, 2)) + "%")
        self.val_f1 .setText("Validation f1 score: "+ str(round(val[2] * 100, 2)) + "%")
        self.test_error.setText("Test error: " + str(round(test[0] * 100, 2)) + "%")
        self.test_acc.setText("Test accuracy: " + str(round(test[1] * 100, 2)) + "%")
        self.test_f1.setText("Test f1 score: " + str(round(test[2] * 100, 2)) + "%")

    def connect_to_dataBase(self):
        try:
            self.conn = mariadb.connect(
                user="kayttis",
                password="salis",
                host="localhost",
                port=3306,
                database="junction"
            )
            print('Success')

        except mariadb.Error as e:
            print(f"Error connecting to MariaDB Platform: {e}")
            sys.exit(1)

            # Get Cursor

        self.cur = self.conn.cursor()


    def delete_data(self):
        self.cur.execute(
            "DELETE FROM junction")

        self.conn.commit()

    def make_data(self):

        therapists = [[2,7,4,1,9,5,4,2], [8,2,4,7,2,2,4,2], [4,4,1,1,7, 8, 1,8], [4,1,1,5,8,3,8,3],
                      [9,8,4,7,2,4,3,8], [4,1,9,9,9,6,2,3], [2,3,2,5,2,7,5,5], [9,2,7,4,5,2,7,4]]
        for x in range(int(self.add_val.toPlainText())):
            therapist = random.randint(0, 7)

            val = therapists[therapist]
            weighted_random_1 = int(get_truncated_normal(val[0], 1, 0, 10).rvs(1))
            weighted_random_2 = int(get_truncated_normal(val[1], 1, 0, 10).rvs(1))
            weighted_random_3 = int(get_truncated_normal(val[2], 1, 0, 10).rvs(1))
            weighted_random_4 = int(get_truncated_normal(val[3], 1, 0, 10).rvs(1))
            weighted_random_5 = int(get_truncated_normal(val[4], 1, 0, 10).rvs(1))
            weighted_random_6 = int(get_truncated_normal(val[5], 1, 0, 10).rvs(1))
            weighted_random_7 = int(get_truncated_normal(val[6], 1, 0, 10).rvs(1))
            weighted_random_8 = int(get_truncated_normal(val[7], 1, 0, 10).rvs(1))

            self.cur.execute(
                "INSERT INTO junction (arvo1, arvo2, arvo3, arvo4, arvo5, arvo6, arvo7, arvo8, terapeutti) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (weighted_random_1, weighted_random_2, weighted_random_3, weighted_random_4, weighted_random_5,
                 weighted_random_6, weighted_random_7, weighted_random_8, therapist))

        self.conn.commit()
        self.data = self.get_data()



class Ui(QMainWindow):
    def __init__(self):
        self.conn = 0
        self.cur = 0
        self.connect_to_dataBase()
        self.subWindow = 0
        QMainWindow.__init__(self)
        loadUi("mainscreen.ui",self)
        self.Answer1.valueChanged.connect(self.setValue1)
        self.Answer2.valueChanged.connect(self.setValue2)
        self.Answer3.valueChanged.connect(self.setValue3)
        self.Answer4.valueChanged.connect(self.setValue4)
        self.Answer5.valueChanged.connect(self.setValue5)
        self.Answer6.valueChanged.connect(self.setValue6)
        self.Answer7.valueChanged.connect(self.setValue7)
        self.Answer8.valueChanged.connect(self.setValue8)

        self.Submit.clicked.connect(self.predict)
        self.Submit.clicked.connect(self.add_data)
        self.config.clicked.connect(self.open_window)

    def open_window(self):
        self.subWindow = SubWindow()
        self.subWindow.show()
    def setValue1(self, item):
        self.Value1.setText(str(item))

    def setValue2(self, item):
        self.Value2.setText(str(item))

    def setValue3(self, item):
        self.Value3.setText(str(item))

    def setValue4(self, item):
        self.Value4.setText(str(item))

    def setValue5(self, item):
        self.Value5.setText(str(item))

    def setValue6(self, item):
        self.Value6.setText(str(item))

    def setValue7(self, item):
        self.Value7.setText(str(item))

    def setValue8(self, item):
        self.Value8.setText(str(item))

    def connect_to_dataBase(self):
        try:
            self.conn = mariadb.connect(
                user="kayttis",
                password="salis",
                host="localhost",
                port=3306,
                database="junction"
            )
            print('Success')

        except mariadb.Error as e:
            print(f"Error connecting to MariaDB Platform: {e}")
            sys.exit(1)

            # Get Cursor

        self.cur = self.conn.cursor()

    def predict(self):
        ml = Machinelearning()
        loaded_model = ml.load_model("finalized_model.sav")
        x_new = [[int(self.Value1.text()), int(self.Value2.text()), int(self.Value3.text()), int(self.Value4.text()), int(self.Value4.text()),
                  int(self.Value5.text()), int(self.Value6.text()), int(self.Value7.text()), int(self.Value8.text())]]
        prediction = loaded_model.predict(x_new)
        self.suggested.setText("Suggested Therapist: "+ str(prediction[0]))
    def add_data(self):
        self.cur.execute(
            "INSERT INTO junction (arvo1, arvo2, arvo3, arvo4, arvo5, arvo6, arvo7, arvo8, terapeutti) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (int(self.Value1.text()), int(self.Value2.text()), int(self.Value3.text()), int(self.Value4.text()),
             int(self.Value5.text()), int(self.Value6.text()), int(self.Value7.text()), int(self.Value8.text()), 5))
        self.conn.commit()

    def get_data(self):
        engine = sqlalchemy.create_engine("mariadb+mariadbconnector://kayttis:salis@127.0.0.1:3306/junction")
        df = pd.read_sql("SELECT * FROM junction", engine)
        return df





app = QApplication([])
window = Ui()
window.show()
app.exec_()





