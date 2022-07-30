import sqlite3
from contextlib import closing
import random

with closing(sqlite3.connect("mbus.db")) as connection:
    with closing(connection.cursor()) as cursor:
        for dev_id in range(0,2):
            for hour in range(8,18):
                for minute in range(0,59):
                    time = "2021-12-6 "
                    #hour = random.randint(8,17)
                    #minute = random.randint(0,59)
                    second = 0
                    time += str(hour)
                    time += ":"
                    time += str(minute)
                    time += ":"
                    time += str(second)
                    dev_id = random.randint(0,1)
                     
                    # People coming out of classes
                    if ((minute >= 0 and minute <= 10) or (minute >= 30 and minute <= 40)) and (hour >= 9 and hour <= 17):
                        count = random.randint(0,20)
                    elif (hour < 9):
                        count = random.randint(0,5)
                    else:
                        count = random.randint(0,10)  
                        
                    cursor.execute("INSERT INTO people VALUES (?, ?, ?)", (dev_id, time, count))
    connection.commit()