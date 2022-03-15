import psycopg2
import sqlite3
import pandas as pd

from datetime import datetime


class MyPostgres:
    def __init__(self, host='localhost', port='5432', database='postgres'):
        # progreSQL
        self.conn = psycopg2.connect(database=database, user='Quant', password='Quant', host=host, port=port)
        self.cur = self.conn.cursor()

    def disconnect(self):
        self.conn.close()

    def save_sql(self, str_sql):
        try:
            self.cur.execute(str_sql)
        except psycopg2.IntegrityError:
            pass
        except Exception as e:
            print(e)
        self.conn.commit()

    def get_data(self, str_sql):
        try:
            self.cur.execute(str_sql)
            data = self.cur.fetchall()
            return data
        except Exception as e:
            print(e)


class MySqlite:
    def __init__(self):
        global pwd
        pwd = "E:\\git_repo\\quant_master\\src\main\\quant\\projects_py\\cryptocurrency\\crypto.db"

        self.conn = sqlite3.connect(pwd)
        self.__curr = self.conn.cursor()

    def fetchall(self, sql_str):
        self.__curr.execute(sql_str)
        return self.__curr.fetchall()

    def commit(self, sql_str):
        self.__curr.execute(sql_str)
        self.conn.commit()

    def disconnect(self):
        self.conn.close()


def get_tick(ccy: str, start: datetime, end: datetime, broker: str):
    """
    Gets tick price given the time period and the condition of order from database.\n
    :param ccy: Currency pair in XXXYYY.
    :param start: Start time period.
    :param end: End time period.
    :param broker: Broker of data source, e.g. 'HSBC'.
    :return: Tick data in database with columns [DateTime, bid, offer].
    """
    my_db = MyPostgres()
    try:
        command = f"SELECT datetime, bidprice, askprice " \
                  f"FROM {str.upper(ccy)} " \
                  f"WHERE broker ='{broker}' " \
                  f"AND datetime >= '{start.strftime('%Y-%m-%dT%H:%M:%S')}' " \
                  f"AND datetime < '{end.strftime('%Y-%m-%dT%H:%M:%S')}' " \
                  f"ORDER BY datetime;"
        rows = my_db.get_data(command)
    finally:
        my_db.disconnect()

    col = ["DateTime", "bid", "offer"]

    data = pd.DataFrame(rows, columns=col)
    data.DateTime = pd.to_datetime(data.DateTime)

    return data
