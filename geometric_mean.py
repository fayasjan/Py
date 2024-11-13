import math
x = int(input("Enter the number "))
def geoMean(x):
    y = 1
    for i in range(1,x+1):
        y*=i
    v = math.pow(y,1/x)
    return v
print("The GeoMean of %d is %f" %(x,geoMean(x)))