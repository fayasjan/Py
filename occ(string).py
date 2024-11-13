rst = input("Enter the string: ")
word = input("Enter the word to look for: ")
count = rst.lower().split().count(word.lower())
print("The word %s occurs %d times in the string" %(word,count))
