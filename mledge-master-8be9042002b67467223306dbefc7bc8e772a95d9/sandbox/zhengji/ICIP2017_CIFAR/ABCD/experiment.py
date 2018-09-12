from threading import Thread
import time

def claim(num):
    while True:
        value = num
        time.sleep(0.1)
        if value != num:
            print "thread %d: I see value %d" %(num, value)

thread_num = 20
workers = []
for i in range(thread_num):
    worker = Thread(target=claim, args=(i,))
    worker.setDaemon(True)
    worker.start()
    workers += []
time.sleep(3)
