x = int(input("Enter a number: "))
b = 0
print("The Reversed number is: ", end="")
def countPower(x):
    n = 0
    while(x!=0):
        x= x//10
        n+=1
    return n
def to10spower(n):
    return 10**n
k = countPower(x)
for i in range(k):
    c = x%10
    x= x//10
    b+= c*to10spower(k-i-1)
print(b)