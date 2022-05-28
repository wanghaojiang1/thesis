import psycopg2

conn = psycopg2.connect(
            host="localhost",
            database="thesis",
            user='root',
            password='root')

def getConnection():
    global conn
    return conn.cursor()

def commit():
    conn.commit()
