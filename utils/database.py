import psycopg2

# When service is down: start service
# Open WSL: sudo service postgresql restart
# SQL workbench address: jdbc:postgresql://localhost:5432/thesis
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

def dropTables():
    sql = """
        DO $$
            DECLARE
                row record;
            BEGIN
                FOR row IN SELECT * FROM pg_tables WHERE schemaname = 'public' 
                LOOP
                    EXECUTE 'DROP TABLE public.' || quote_ident(row.tablename) || ' CASCADE';
                END LOOP;
            END;
        $$;
    """
    cursor = getConnection()
    cursor.execute(sql)

    commit()
    

