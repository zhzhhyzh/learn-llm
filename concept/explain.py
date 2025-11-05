#Goal:Tokenize this characters short story into individual words and special characters
#Use "re" library to split this text 

import re

text = "Hello, world. This, is a test."
result = re.split(r'(\s)',text) #When there is the whitespace then split it

print(result)

#Now split commas and period too
result = re.split(r'([,.]|\s)',text)

print(result)

#The result will contain whitespace, now need to be removed
#Reduce white space can reduce memory using,
#However keeping whitespace can be useful to train model sensitive to the exact structure of the text
#Whitespace may contain some of the meaning
#In training data, whitespace is important
result = [item for item in result if item.strip()]
print(result)

#Includes more char to be splited, and filter out the whitespace
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print (result)
