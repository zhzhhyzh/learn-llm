import re
#STEP 1----
print("STEP 1")
#Print command to print the total number of char followed by 100 characters of this illustration purpose
with open("the-verdict.txt","r",encoding="utf-8")as f:
    raw_text = f.read()

print("Total number of char:", len(raw_text))
print(raw_text[:99])

#Goal:Tokenize this characters short story into individual words and special characters
#Now that we got basic tokenizer working, let's apply it to Edith Wharton's entire short story
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])

#Each unique token is mapped to an unique interger called token ID
#The token are sorted in alphabetical order in vocabulary
print(len(preprocessed))

#STEP 2 ---
print("STEP 2")
#Create token IDs
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)

#After determining the vocab size, we create the vocabulary
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if(i>=50):
        break

#Later in this book, we need to convert the outputs of an LLM from numbers back into text, we also need a way to turn token IDs into text
#So, we can create an inverse version of the vocabulary that maps token IDs back to corresponding text tokens
#Use encode method and decode method
#Encode [Sample text > Tokenized text > token IDs]
#Decode [Token IDs > tokenized text > sample text]
class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        #To avoid error when tokenizer is not found in the vocabulary which is
        # ids = [self.str_to_int[s] for s in preprocessed if s in self.str_to_int]
        #Second method is to add |unk| and |endoftext| in vocab
        return ids
    
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        #Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
#Now, instantiate the new tokenizer object that defined above, test the class
tokenizer = SimpleTokenizerV1(vocab)

text = """It's the last he painted, you know," Mrs.Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print("Token IDs:", ids)

decoded_text = tokenizer.decode(ids)
print("Decoded text:", decoded_text)

#When word is not tokenize, the words will be missing
# text = "Hello, do you like tea?"
# ids = tokenizer.encode(text)
# print(ids)

# decoded_text = tokenizer.decode(ids)
# print("Decoded text:", decoded_text)

#We need to utilize |UNK| which is unknown and |endoftext| which used to join both different documentation source
#|endoftext| able us to work with multiple text source
#|endoftext| tokens act as markers, signaling the start of end of a particular segment
#It leads to more effective to process LLM

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))

class SimpleTokenizerV2:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        #With eliminated the problem in V1
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        #Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = "<|endoftext|>".join((text1,text2))

print(text)

tokenizer.encode(text)
tokenizer.decode(tokenizer.encode(text))

#|BOS|,Beginning of Sequence, this token marks the start of a text, which a LLM begin
#|EOS|,End of Sequence, this toke is positioned at the end of a text, useful when concatening two different sources
#|PAD|, Padding,  when training LLMs with batch size larger than, the batch might containing vary length, to ensure each text have same length in each batch, 
# The shorter texts are extended or "padded" using the [PAD] token, up to the length of the longest batch

#GPT doesn't mention any tokens but only <|endoftext|>
#GPT use a byte pair token encoding to break down words into subword unit to replace <|unk|>
