import busio
import adafruit_pcf8523
import time
import board

def getRTC(I2C, rtc):
    myI2C = busio.I2C(board.SCL, board.SDA)
    rtc = adafruit_pcf8523.PCF8523(myI2C)
    t = rtc.datetime

    days = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
    print("The date is %s %d/%d/%d" % (days[t.tm_wday], t.tm_mday, t.tm_mon, t.tm_year))
    print("The time is %d:%02d:%02d" % (t.tm_hour, t.tm_min, t.tm_sec))
