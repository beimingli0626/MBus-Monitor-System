import sqlite3
from contextlib import closing
import dweepy
import logging
import signal
import sys

def create_table():
    with closing(sqlite3.connect("mbus.db")) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='people' ''')
            if cursor.fetchone()[0]!=1 : {
                cursor.execute("CREATE TABLE people (device_id INTEGER, time DATETIME, count INTEGER)")
            }

def print_all_data():
    with closing(sqlite3.connect("mbus.db")) as connection:
        with closing(connection.cursor()) as cursor:
            # Grab all the rows of data from the table
            rows = cursor.execute("SELECT device_id, time, count FROM people").fetchall()
            print(rows)
            
def insert_data_db(dev_id, count, time):
    logging.info("Inserting data into database")
    logging.info(str(dev_id) + ' ' + time + ' ' + str(count))
    with closing(sqlite3.connect("mbus.db")) as connection:
        with closing(connection.cursor()) as cursor:
            # Create a test data item
            cursor.execute("INSERT INTO people VALUES (?, ?, ?)", (dev_id, time, count))
        connection.commit()
    logging.info("Added data successfully")
    
def parse(raw_dweet):
    if ('content' not in raw_dweet.keys()):
        logging.warning("Received dweet with invalid format (no content key)")
        return
        
    if ('id' not in raw_dweet['content'].keys()):
        logging.warning("Received dweet with invalid format (no id key)")
        return
     
    if ('time' not in raw_dweet['content'].keys()):
        logging.warning("Received dweet with invalid format (no time key)")
        return
        
    if ('count' not in raw_dweet['content'].keys()):
        logging.warning("Received dweet with invalid format (no count key)")
        return
        
    time = raw_dweet['content']['time']
    count = raw_dweet['content']['count']
    dev_id = raw_dweet['content']['id']
    gsm_format_time_split = time.split('/')
    if len(gsm_format_time_split) != 5:
        logging.warning("Received dweet with invalid time format")
        return
        
    gsm_format_time = gsm_format_time_split[0] + '-' + gsm_format_time_split[1] + '-' + gsm_format_time_split[2] + ' ' + gsm_format_time_split[3] + ':' + gsm_format_time_split[4] + ':00'
    insert_data_db(str(dev_id), str(count), gsm_format_time)
    
def wait_for_messages():
    logging.info("Listening for Dweets")
    for dweet in dweepy.listen_for_dweets_from('gsm_mod'):
        logging.info("Got Dweet")
        parse(dweet)
        
def signal_handler(sig, frame):
    print("ending...")
    sys.exit(0)
    
    
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting server")
    create_table()
    wait_for_messages()
