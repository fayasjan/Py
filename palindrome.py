x = input("Enter the number: ")
try:
    x = int(x)
    y = int(str(x)[::-1])
    if x == y:
        print("The number is a palindrome")
    else:
        print("The number is not a palindrome")
except:
    print("Invalid input")
    exit()
