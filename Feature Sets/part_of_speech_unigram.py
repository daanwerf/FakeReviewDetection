import nltk
from nltk import RegexpTagger, UnigramTagger
from nltk.corpus import brown
from nltk import word_tokenize

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

text = "The quick brown fox jumps over the lazy dog"

unigrams = word_tokenize(text)
tagged_unigrams = unigram_tagger.tag(unigrams)

print(unigrams)
print(tagged_unigrams)