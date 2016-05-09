"""Analyzing robins seminar text to confirm that, yes, he talks a lot about projects..."""
import nltk
from nltk import FreqDist
from nltk.tokenize import word_tokenize

def read_words(name):
    with open(name, 'r') as rfile:
        yield from (word for word in word_tokenize(rfile.read()))


stopwords = (
    set((',', '[', ']', '.', '!', ';', ':', '*', "''", '``', '–', "'s"))  # docs-formatting and other non-words
    | set(nltk.corpus.stopwords.words('english'))  # uninteresting words like 'that', 'was', 'is', etc.
)

stopwords.remove('i')  # robin talks about himself a lot, let's include that for the sake of comedy


robin = FreqDist(
    word.lower() for word in read_words('robin.txt') if word.lower() not in stopwords
)

# would be more interesting to use lots of other prosam texts as a base, but I've already procrastinated enough...
base = FreqDist(
    word.lower() for word in nltk.corpus.brown.words() if word.lower() not in stopwords
)


# I run it like `python freq_analyse.py | sort -n | tail -n 15`
for k,v in robin.items():
    print('%.10f' % abs(robin.freq(k) - base.freq(k)),k)  # difference in word frequency


# sample result:

# 0.0074393564 well
# 0.0078216064 know
# 0.0088326042 learned
# 0.0103504380 like
# 0.0104640871 feel
# 0.0104676596 education
# 0.0107427367 learning
# 0.0112185385 good
# 0.0121670187 really
# 0.0126582278 i’ve
# 0.0143654005 also
# 0.0161659053 courses
# 0.0178241815 knowledge
# 0.0251949931 projects
# 0.0938501265 i
