import mariadb
import pandas
import sys

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

    #cur.execute("SELECT * FROM junction")

    engine = sqlalchemy.create_engine("mariadb+mariadbconnector://kayttis:salis@127.0.0.1:3306/junction")


    df = pandas.read_sql("SELECT * FROM junction", engine)
    #conn.commit()

    print(df)



    #print(q)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    connect()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
