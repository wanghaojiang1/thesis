from utils import database

def get_nodes():
    cursor = database.getConnection()
    sql = "SELECT * FROM nodes"
    cursor.execute(sql)
    records = cursor.fetchall()

    return list(map(lambda node: {'id': node[0], 'table': node[1]}, records))

def get_siblings(id):
    cursor = database.getConnection()

    # Retrieving table information
    sql = "SELECT * FROM nodes WHERE column_id=%s"
    cursor.execute(sql, (id, ))
    record = cursor.fetchone()
    table = record[1]

    # Retrieving siblings: excluding id
    sql = "SELECT * FROM nodes WHERE source=%s AND column_id!=%s"
    cursor.execute(sql, (table, id,))
    siblings = list(map(lambda sibling: {'id': sibling[0], 'table': sibling[1]}, cursor.fetchall()))
    return siblings

def insert_node(column_id, source):
    cursor = database.getConnection()
    sql = 'INSERT INTO nodes (column_id, source) VALUES (%s, %s)'
    cursor.execute(sql, (column_id, source, ))
    return True

def get_match(edge_id):
    cursor = database.getConnection()
    sql = "SELECT column_1, column_2 FROM edges WHERE edge_id=%s"
    cursor.execute(sql, (edge_id, ))
    record = cursor.fetchone()

    # Retrieving table information
    sql = "SELECT edge_id, algorithm, similarity_score FROM matches WHERE edge_id=%s"
    cursor.execute(sql, (edge_id, ))
    records = cursor.fetchall()
    records = list(map(lambda match: {match[1]: match[2]}, records))
    
    records = {k: v for d in records for k, v in d.items()}
    return {
        'id': edge_id,
        'from': record[0],
        'to': record[1],
        'scores': records
    }

def get_match_by_columns(column_from, column_to):
    cursor = database.getConnection()

    # Retrieving table information
    sql = "SELECT * FROM nodes WHERE (column_1=%s AND column_2=%s) OR (column_1=%s AND column_2=%s)"
    cursor.execute(sql, (column_from, column_to, column_to, column_from, ))
    record = cursor.fetchone()

    return record

def get_matches():
    cursor = database.getConnection()
    sql = "SELECT edge_id, column_1, column_2 FROM edges"
    cursor.execute(sql)
    edges = cursor.fetchall()

    result = []

    for edge in edges:
        # Retrieving table information
        edge_id = edge[0]
        sql = "SELECT edge_id, algorithm, similarity_score FROM matches WHERE edge_id=%s"
        cursor.execute(sql, (edge_id, ))
        records = cursor.fetchall()
        records = list(map(lambda match: {match[1]: match[2]}, records))
        
        records = {k: v for d in records for k, v in d.items()}

        result.append({
                'id': edge[0],
                'from': edge[1],
                'to': edge[2],
                'scores': records
            })

    return result

def get_unlabelled_matches():
    cursor = database.getConnection()
    sql = "SELECT edge_id, column_1, column_2 FROM edges WHERE labelled=false"
    cursor.execute(sql)
    edges = cursor.fetchall()

    result = []

    for edge in edges:
        # Retrieving table information
        edge_id = edge[0]
        sql = "SELECT edge_id, algorithm, similarity_score FROM matches WHERE edge_id=%s"
        cursor.execute(sql, (edge_id, ))
        records = cursor.fetchall()
        records = list(map(lambda match: {match[1]: match[2]}, records))
        
        records = {k: v for d in records for k, v in d.items()}

        result.append({
                'id': edge[0],
                'from': edge[1],
                'to': edge[2],
                'scores': records
            })

    return result

def insert_edge(column_from, column_to):
    cursor = database.getConnection()
    sql = 'INSERT INTO edges (edge_id, column_1, column_2) VALUES (DEFAULT, %s, %s) RETURNING edge_id;'
    cursor.execute(sql, (column_from, column_to, ))
    id = cursor.fetchone()[0]
    return id

def insert_match(edge_id, algorithm, similarity_score):
    cursor = database.getConnection()
    sql = 'INSERT INTO matches (edge_id, algorithm, similarity_score) VALUES (%s, %s, %s)'
    cursor.execute(sql, (edge_id, algorithm, similarity_score, ))
    return id

def label_edge(edge_id):
    cursor = database.getConnection()
    sql = 'UPDATE edges SET labelled=true WHERE edge_id=%s'
    cursor.execute(sql, (edge_id, ))
    database.commit()
    return True

def get_tables():
    cursor = database.getConnection()
    sql = "SELECT DISTINCT source FROM nodes"
    cursor.execute(sql)
    records = cursor.fetchall()
    return list(map(lambda table: table[0], records))


def delete_nodes():
    cursor = database.getConnection()
    sql = "DELETE FROM nodes"

    cursor.execute(sql)
    database.commit()
    return True

def delete_node(id):
    cursor = database.getConnection()
    sql = "DELETE FROM nodes WHERE column_id=%s"

    cursor.execute(sql, (id, ))
    database.commit()
    return True

def delete_matches():
    cursor = database.getConnection()
    sql = "DELETE FROM matches"

    cursor.execute(sql)
    database.commit()
    return True

def delete_edges():
    cursor = database.getConnection()
    sql = "DELETE FROM edges"

    cursor.execute(sql)
    database.commit()
    return True