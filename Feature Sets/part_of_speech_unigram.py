import nltk
from nltk import word_tokenize

input = "And now for something completely different"

text = word_tokenize(input)
POS_Set = nltk.pos_tag(text)

unigrams = input.split()

print(POS_Set)
print(unigrams)