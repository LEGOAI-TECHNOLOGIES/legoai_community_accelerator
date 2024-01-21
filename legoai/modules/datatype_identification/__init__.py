import nltk

try:
    nltk.data.find('corpora/words')
except LookupError:    
    nltk.download('words')