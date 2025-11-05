import importlib.metadata
import tiktoken

#Verify installation
print("tiktoken version:", importlib.metadata.version("tiktoken"))

#Same as previous project tokenizer creation
tokenizer = tiktoken.get_encoding("gpt2")

text=("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
"of someunknownPlace.")

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)

print(strings)

#Test for other words
integers = tokenizer.encode("Akwirw eir")
print(integers)

strings =  tokenizer.decode(integers)
print(strings)


