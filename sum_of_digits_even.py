def sumDigi(x):
    s = 0
    while(x != 0):
        s += x % 10
        x = x // 10
    return s

for i in range(100, 201):
    if sumDigi(i) % 2 == 0:
        print(i)
