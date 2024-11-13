brake = False
while not brake:
    print("Welcome to calculator, options are:")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Division")
    print("5. Exit")
    choice = int(input("Enter your choice: "))
    
    if choice == 5:
        brake = True
        break
    
    x = int(input("Enter the first number: "))
    y = int(input("Enter the second number: "))
    
    if choice == 1:
        print("The sum is: ", x + y)
    elif choice == 2:
        print("The difference is: ", x - y)
    elif choice == 3:
        print("The product is: ", x * y)
    elif choice == 4:
        if y != 0:
            print("The quotient is: ", x / y)
        else:
            print("Cannot divide by zero.")
    else:
        print("Invalid choice")
