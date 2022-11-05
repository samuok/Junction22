import mariadb
import pandas
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


class Ui(QMainWindow):
    def __init__(self):
        self.conn = 0
        self.cur = 0
        self.connect_to_dataBase()
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

    def predict(self, model):
        ml = Machinelearning()
        loaded_model = ml.load_model("finalized_model.sav")
        x_new = [[int(self.Value1.text()), int(self.Value2.text()), int(self.Value3.text()), int(self.Value4.text()), int(self.Value4.text())]]
        prediction = loaded_model.predict(x_new)
        print("prediction", prediction)
    def add_data(self):
        self.cur.execute(
            "INSERT INTO junction (arvo1, arvo2, arvo3, arvo4, arvo5, arvo6, arvo7, arvo8, terapeutti) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (int(self.Value1.text()), int(self.Value2.text()), int(self.Value3.text()), int(self.Value4.text()),
             int(self.Value5.text()), int(self.Value6.text()), int(self.Value7.text()), int(self.Value8.text()), 5))
        self.conn.commit()

    def get_data(self):
        engine = sqlalchemy.create_engine("mariadb+mariadbconnector://kayttis:salis@127.0.0.1:3306/junction")
        df = pandas.read_sql("SELECT * FROM junction", engine)
        return df

    def make_data(self):
        for x in range(5000):
            therapist = random.randint(0, 4)
            weighted_random_1 = 0
            weighted_random_2 = 0
            weighted_random_3 = 0
            weighted_random_4 = 0
            weighted_random_5 = 0
            weighted_random_6 = 0
            weighted_random_7 = 0
            weighted_random_8 = 0

            if therapist == 0:
                weighted_random_1 = int(get_truncated_normal(2, 1, 0, 10).rvs(1))
                weighted_random_2 = int(get_truncated_normal(7, 1, 0, 10).rvs(1))
                weighted_random_3 = int(get_truncated_normal(4, 1, 0, 10).rvs(1))
                weighted_random_4 = int(get_truncated_normal(1, 1, 0, 10).rvs(1))
                weighted_random_5 = int(get_truncated_normal(9, 1, 0, 10).rvs(1))
                weighted_random_6 = int(get_truncated_normal(5, 1, 0, 10).rvs(1))
                weighted_random_7 = int(get_truncated_normal(4, 1, 0, 10).rvs(1))
                weighted_random_8 = int(get_truncated_normal(2, 1, 0, 10).rvs(1))
            if therapist == 1:
                weighted_random_1 = int(get_truncated_normal(7, 1, 0, 10).rvs(1))
                weighted_random_2 = int(get_truncated_normal(4, 1, 0, 10).rvs(1))
                weighted_random_3 = int(get_truncated_normal(2, 1, 0, 10).rvs(1))
                weighted_random_4 = int(get_truncated_normal(4, 1, 0, 10).rvs(1))
                weighted_random_5 = int(get_truncated_normal(3, 1, 0, 10).rvs(1))
                weighted_random_6 = int(get_truncated_normal(8, 1, 0, 10).rvs(1))
                weighted_random_7 = int(get_truncated_normal(5, 1, 0, 10).rvs(1))
                weighted_random_8 = int(get_truncated_normal(4, 1, 0, 10).rvs(1))

            if therapist == 2:
                weighted_random_1 = int(get_truncated_normal(5, 1, 0, 10).rvs(1))
                weighted_random_2 = int(get_truncated_normal(9, 1, 0, 10).rvs(1))
                weighted_random_3 = int(get_truncated_normal(4, 1, 0, 10).rvs(1))
                weighted_random_4 = int(get_truncated_normal(2, 1, 0, 10).rvs(1))
                weighted_random_5 = int(get_truncated_normal(8, 1, 0, 10).rvs(1))
                weighted_random_6 = int(get_truncated_normal(1, 1, 0, 10).rvs(1))
                weighted_random_7 = int(get_truncated_normal(2, 1, 0, 10).rvs(1))
                weighted_random_8 = int(get_truncated_normal(3, 1, 0, 10).rvs(1))

            if therapist == 3:
                weighted_random_1 = int(get_truncated_normal(2, 1, 0, 10).rvs(1))
                weighted_random_2 = int(get_truncated_normal(1, 1, 0, 10).rvs(1))
                weighted_random_3 = int(get_truncated_normal(4, 1, 0, 10).rvs(1))
                weighted_random_4 = int(get_truncated_normal(7, 1, 0, 10).rvs(1))
                weighted_random_5 = int(get_truncated_normal(9, 1, 0, 10).rvs(1))
                weighted_random_6 = int(get_truncated_normal(1, 1, 0, 10).rvs(1))
                weighted_random_7 = int(get_truncated_normal(8, 1, 0, 10).rvs(1))
                weighted_random_8 = int(get_truncated_normal(5, 1, 0, 10).rvs(1))
            if therapist == 4:
                weighted_random_1 = int(get_truncated_normal(9, 1, 0, 10).rvs(1))
                weighted_random_2 = int(get_truncated_normal(2, 1, 0, 10).rvs(1))
                weighted_random_3 = int(get_truncated_normal(4, 1, 0, 10).rvs(1))
                weighted_random_4 = int(get_truncated_normal(3, 1, 0, 10).rvs(1))
                weighted_random_5 = int(get_truncated_normal(7, 1, 0, 10).rvs(1))
                weighted_random_6 = int(get_truncated_normal(2, 1, 0, 10).rvs(1))
                weighted_random_7 = int(get_truncated_normal(3, 1, 0, 10).rvs(1))
                weighted_random_8 = int(get_truncated_normal(6, 1, 0, 10).rvs(1))

            self.cur.execute(
                "INSERT INTO junction (arvo1, arvo2, arvo3, arvo4, arvo5, arvo6, arvo7, arvo8, terapeutti) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (weighted_random_1, weighted_random_2, weighted_random_3, weighted_random_4, weighted_random_5,
                 weighted_random_6, weighted_random_7, weighted_random_8, therapist))

        print(int(self.Value1.text()))
        self.conn.commit()




app = QApplication([])
window = Ui()
window.show()
app.exec_()
#window.make_data()
df = window.get_data()


ml = Machinelearning()
X, y = ml.split_database(df, "terapeutti")
data = ml.data_split(X, y, 0.40)
#ml.find_parameters(data[0], data[1])
#model = ml.train_model(data[0], data[1])
#ml.save_model(model)
new_model = ml.load_model("finalized_model.sav")

ml.predict_model(new_model, data[2], data[3], "Validation")
ml.predict_model(new_model, data[4], data[5], "Test")


