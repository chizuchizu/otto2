import pandas as pd

a = 0
for i in range(100000):
    a += 1
    if i % 100 == 0:
        print(i)
