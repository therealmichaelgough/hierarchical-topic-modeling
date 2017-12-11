import time
import random
from multiprocessing import Process, Queue, JoinableQueue, cpu_count, Lock
import signal

""" Adjust time.sleep() values to see action of consumer """

lock = Lock()

def work(id, jobs, result):
    while True:
        with lock:
            print "hello i'm {}!".format(id)
        task = jobs.get()
        if task is None:
            with lock:
                "print None!"
            break
        time.sleep(random.randint(1,10))
        with lock:
            print "hellop i'm {}. I have object: {}".format(id, task)
        result.put("%s task r")
        

def alarm_handler(x, y):
    raise Alarm()


class Alarm(Exception):
    pass


def main():
    signal.signal(signal.SIGALRM, alarm_handler)
    jobs = Queue()
    result = JoinableQueue()
    NUMBER_OF_PROCESSES = cpu_count()

    tasks = ["1","2","3","4","5"]

    for w in tasks:
        jobs.put(w)
    
    [Process(target=work, args=(i, jobs, result)).start() for i in xrange(NUMBER_OF_PROCESSES)]

    print 'starting workers'

    while True:
        signal.alarm(6)
        try:
            r = result.get()
            with lock:
                print "I consumed {}!".format(r)
            result.task_done()
            jobs.put(None)
            signal.alarm(0)
        except Alarm:

        print "done!"
        exit()

if __name__ == '__main__':
    main()
