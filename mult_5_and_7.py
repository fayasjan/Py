x = int(input("Enter a number: "))

if (x % 5 == 0) and (x % 7 == 0):
    print("The number is divisible by both 5 and 7")
elif (x % 5 == 0):
    print("The number is divisible by 5")
elif (x % 7 == 0):
    print("The number is divisible by 7")
else:
    print("The number is not divisible by both 5 and 7")
