import matplotlib.pyplot as plt
import matplotlib.dates as d
import numpy as np
import sqlite3
from contextlib import closing
import datetime

datetimes = []
device_dict = {}

def get_data():
    with closing(sqlite3.connect("mbus.db")) as connection:
        with closing(connection.cursor()) as cursor:
            # Grab all the rows of data from the table
            rows = cursor.execute("SELECT device_id, time, count FROM people").fetchall()
            for row in rows:
                devid = str(row[0])
                count = row[2]
                date, time = row[1].split(' ')
                year,month,day = date.split('-')
                hour,minute,second = time.split(':')
                dt = datetime.datetime(int(year), int(month),int(day),int(hour),int(minute),int(second))
                if devid in device_dict:
                    device_dict[devid].append((dt,count))
                else:
                    device_dict[devid] = []
                    device_dict[devid].append((dt,count))
   
def plot(devid, times, counts):
    plt.style.use('ggplot')
    
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    

    x = times
    energy = counts

    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, energy, color='green')
    plt.xlabel("Time")
    plt.ylabel("Count (people)")
    plt.title("Device " + devid + " " + datetime.datetime.today().strftime("%b-%d-%Y"))

    plt.xticks(x_pos, x)
    plt.setp(plt.gca().get_xticklabels(), rotation=45)


    #plt.savefig(devid+".png")
    fig.savefig(devid+".png", dpi=100)
    
    #plt.show()
    
def create_plots():
    get_data()
    for k in device_dict.keys():
        #print("Key: " + k)
        device_dict[k].sort(key = lambda x: x[0]) 
        #print(device_dict[k])
        times = [x[0] for x in device_dict[k]]
        counts = [x[1] for x in device_dict[k]]
        #plot(k, times, counts)
        
        a = datetime.datetime.today()
        b = datetime.datetime(a.year, a.month, a.day)
        
        bins = []
        count_avgs = []
        
        for hour in range(8,18):
            for minute in range(0,60,20):
                n = 0
                total_count = 0
                start = datetime.datetime(a.year, a.month, a.day, hour, minute)
                end = datetime.datetime(a.year, a.month, a.day, hour, minute+19)
                for time,count in zip(times, counts):
                    if time >= start and time < end:
                        n = n+1
                        total_count = total_count+count
                #print("count: " + str(total_count))
                #print("items: " + str(n))
     
                avg = 0
                if n != 0:
                    avg = total_count/n
                count_avgs.append(avg)
                #print("Average count in range " + start.strftime("%b-%d-%Y %H-%M-%S") + " to " + end.strftime("%b-%d-%Y %H-%M-%S") + " = " + str(avg))
                    
                average_delta = (end - start) / 2
                average_ts = start + average_delta + datetime.timedelta(minutes=1)
                bins.append(average_ts.strftime("%H:%M"))
                    
        plot(k,bins,count_avgs)
        
if __name__=="__main__":
    create_plots()
