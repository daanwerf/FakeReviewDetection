from nltk import RegexpTagger, UnigramTagger, BigramTagger
from nltk.corpus import brown
from nltk import word_tokenize
import nltk

train_text = brown.tagged_sents()
regexp_tagger = RegexpTagger(
            [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
             (r'(The|the|A|a|An|an)$', 'AT'),   # articles
             (r'.*able$', 'JJ'),                # adjectives
             (r'.*ness$', 'NN'),                # nouns formed from adjectives
             (r'.*ly$', 'RB'),                  # adverbs
             (r'.*s$', 'NNS'),                  # plural nouns
             (r'.*ing$', 'VBG'),                # gerunds
             (r'.*ed$', 'VBD'),                 # past tense verbs
             (r'.*', 'NN')                      # nouns (default)
        ])

unigram_tagger = UnigramTagger(train_text, backoff=regexp_tagger)
bigram_tagger = BigramTagger(train_text, backoff=unigram_tagger)


text = "The quick brown fox jumps over the lazy dog"

bigrams = list(nltk.bigrams(word_tokenize(text)))
tagged_bigrams = bigram_tagger.tag(word_tokenize(text))

print(bigrams)
print(tagged_bigrams)