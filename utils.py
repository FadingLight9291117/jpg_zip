import functools
import json
import time


def number_formatter(number, n=2):
    return f'{number:.4f}'


def pretty_print(data, enable=True):
    if enable:
        data = json.dumps(data, indent=4, ensure_ascii=False, sort_keys=False, separators=(',', ': '))
    print(data)


class Number:
    def __init__(self):
        self.number = 0

    def getN(self):
        self.number += 1
        return self.number

    def __call__(self, *args, **kwargs):
        return self.getN()


class Timer:
    def __init__(self):
        self._total_time = []

    def __enter__(self):
        self._begin_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.time()
        self._total_time.append(self._end_time - self._begin_time)

    def run_num(self):
        return len(self._total_time)

    def this_time(self):
        return self._end_time - self._begin_time

    def average_time(self):
        return sum(self._total_time) / len(self._total_time)

    def total_time(self):
        return sum(self._total_time)

    def all_time(self):
        return self._total_time

    def clear(self):
        self._total_time = []

    @classmethod
    def timer(cls, func):
        @functools.wraps(func)
        def wrapper(*args, **kargs):
            begin_time = time.time()
            res = func(*args, **kargs)
            end_time = time.time()
            print(f'run time: {end_time - begin_time:.2f}s.')
            return res

        return wrapper
