n = int(input("enter the no. of fibonacci numbers to generate: "))
a,b = 1,2
for i in range(n):
    print(a, end=" ")
    a,b = b,a+b