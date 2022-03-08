import psycopg2
import sqlite3


class MyPostgres:
    def __init__(self, host, port, database):
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
