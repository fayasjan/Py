x = input("Enter the 3 numbers: ").split()
for i in range(len(x)):
    x[i] = int(x[i])
a, b, c = x[0], x[1], x[2]
print("The greatest of them 3 is: ", end="")
if a > b and a > c:
    print(a)
elif b > a and b > c:
    print(b)
else:
    print(c)
