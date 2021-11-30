import sqlite3
from contextlib import closing

with closing(sqlite3.connect("mbus.db")) as connection:
    with closing(connection.cursor()) as cursor:
        cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='people' ''')
        if cursor.fetchone()[0]!=1 : {
            cursor.execute("CREATE TABLE people (device_id INTEGER, time DATETIME, count INTEGER)")
        }
        # Create a test data item
        cursor.execute("INSERT INTO people VALUES (42, '2021-11-29 11:39:00.000000', 1)")

        # Grab all the rows of data from the table
        rows = cursor.execute("SELECT device_id, time, count FROM people").fetchall()
        print(rows)
