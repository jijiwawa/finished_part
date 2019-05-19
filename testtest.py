import time
from datetime import datetime
time_begin = datetime.now()
s = 0
for i in range(10000):
    for j in range(10000):
        s = 20 + i
time_end =  datetime.now()

print(time_end-time_begin)