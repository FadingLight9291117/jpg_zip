import json


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
