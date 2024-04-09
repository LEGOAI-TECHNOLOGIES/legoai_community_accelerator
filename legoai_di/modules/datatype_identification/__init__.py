import nltk
from legoai_di.core.configuration import MODEL_CONFIG

try:
    nltk.data.find('corpora/words')
except LookupError:    
    nltk.download('words')