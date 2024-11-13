import numpy as np

mA, mB, mC = [], [], []

rA = int(input("Enter the number of rows for matrix A: "))
cA = int(input("Enter the number of columns for matrix A: "))

rB = int(input("Enter the number of rows for matrix B: "))
cB = int(input("Enter the number of columns for matrix B: "))

if cA != rB:
    print("The matrices cannot be multiplied")
else:
    for i in range(rA):
        row = []
        for j in range(cA):
            row.append(int(input(f"Enter the element at position ({i+1}, {j+1}) for matrix A: ")))
        mA.append(row)

    for i in range(rB):
        row = []
        for j in range(cB):
            row.append(int(input(f"Enter the element at position ({i+1}, {j+1}) for matrix B: ")))
        mB.append(row)

    mA = np.array(mA)
    mB = np.array(mB)

    mC = np.dot(mA, mB)

    print("The result of the multiplication is:")
    print(mC)
