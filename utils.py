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
        self._num = 0
        self._total_time = 0

    def __enter__(self):
        self._begin_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.time()
        self._total_time += self._end_time - self._begin_time

        self._num += 1

    def run_num(self):
        return self._num

    def this_time(self):
        return self._end_time - self._begin_time

    def average_time(self):
        return self._total_time / self._num

    def total_time(self):
        return self._total_time

    def clear(self):
        self._num = 0
        self._total_time = 0

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
