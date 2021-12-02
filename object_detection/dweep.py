import dweepy
import json

count = '2'
timeStamp = '2021/11/30/16/44'
dweet = {'count':count, 'time':timeStamp}
ret = dweepy.dweet_for('gsm_mod', dweet)
print(ret)
r = dweepy.get_dweets_for("gsm_mod")
print(r)
