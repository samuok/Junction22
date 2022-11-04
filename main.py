import mariadb
import pandas
import sys
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt

import sqlalchemy
from sqlalchemy import create_engine


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

    # print(q)

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

        cur.execute(
            "INSERT INTO junction (arvo,arvo1, arvo2, arvo3, arvo4) VALUES (?, ?, ?, ?, ?)",
            (int(self.Value1.text()), int(self.Value2.text()), int(self.Value3.text()), int(self.Value4.text()), int(self.Value4.text())))

        print(int(self.Value1.text()))
        conn.commit()




connect()
app = QApplication([])
window = Ui()
window.show()
app.exec_()

