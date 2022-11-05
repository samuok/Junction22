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


def connect():
    # Connect to MariaDB Platform
    try:
        conn = mariadb.connect(
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

    cur = conn.cursor()

    # cur.execute(
    #   "INSERT INTO junction (arvo,arvo1, arvo2, arvo3, arvo4) VALUES (?, ?, ?, ?, ?)",
    #  (4, 5, 8, 1, 5))

    # cur.execute("SELECT * FROM junction")

    engine = sqlalchemy.create_engine("mariadb+mariadbconnector://kayttis:salis@127.0.0.1:3306/junction")

    df = pandas.read_sql("SELECT * FROM junction", engine)
    # conn.commit()

    print(df)
    return df

class Ui(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("mainscreen.ui",self)
        self.Answer1.valueChanged.connect(self.setValue1)
        self.Answer2.valueChanged.connect(self.setValue2)
        self.Answer3.valueChanged.connect(self.setValue3)
        self.Answer4.valueChanged.connect(self.setValue4)
        self.Submit.clicked.connect(self.loadtoDataBase)

    def setValue1(self, item):
        self.Value1.setText(str(item))

    def setValue2(self, item):
        self.Value2.setText(str(item))

    def setValue3(self, item):
        self.Value3.setText(str(item))

    def setValue4(self, item):
        self.Value4.setText(str(item))

    def loadtoDataBase(self):
        try:
            conn = mariadb.connect(
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

        cur = conn.cursor()

        #cur.execute(
         #  "INSERT INTO junction (arvo,arvo1, arvo2, arvo3, arvo4) VALUES (?, ?, ?, ?, ?)",
         # (int(self.Value1.text()), int(self.Value2.text()), int(self.Value1.text()), int(self.Value5.text()), 5))

        therapist = random.randint(0,2)
        if therapist == 0:
            weighted_random = [1] * 10 + [2] * 10 + [3] * 10 + [4] * 5 + [5] * 10 + [6] * 20 + [7] * 10 + [8] * 10 + [9] * 5 + [10] * 10
        if therapist == 1:
            weighted_random = [1] * 10 + [2] * 10 + [3] * 10 + [4] * 5 + [5] * 10 + [6] * 20 + [7] * 10 + [8] * 10 + [9] * 5 + [10] * 10
        if therapist == 2:
            weighted_random = [1] * 10 + [2] * 10 + [3] * 10 + [4] * 5 + [5] * 10 + [6] * 20 + [7] * 10 + [8] * 10 + [9] * 5 + [10] * 10


        for x in range(10000):
            cur.execute(
                "INSERT INTO junction (arvo,arvo1, arvo2, arvo3, arvo4) VALUES (?, ?, ?, ?, ?)",
                (random.choice(weighted_random), random.choice(weighted_random), random.choice(weighted_random),
                 random.choice(weighted_random), random.choice(weighted_random)))
       # cur.execute(
        #    "INSERT INTO junction (arvo,arvo1, arvo2, arvo3, arvo4) VALUES (?, ?, ?, ?, ?)",
         #   (int(self.Value1.text()), int(self.Value2.text()), int(self.Value3.text()), int(self.Value4.text()), int(self.Value4.text())))

        print(int(self.Value1.text()))
        conn.commit()




connect()
app = QApplication([])
window = Ui()
window.show()
app.exec_()

df = connect()
print(df.columns)
ml = Machinelearning()
X, y = ml.split_database(df, "arvo4")
data = ml.data_split(X, y, 0.40)
#ml.find_parameters(data[0], data[1])
model = ml.train_model(data[0], data[1])
ml.predict_model(model, data[2], data[3], "Validation")
ml.predict_model(model, data[4], data[5], "Test")


