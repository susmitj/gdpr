import string
import nltk
from nltk.stem.snowball import SnowballStemmer
import pickle
from collections import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI
import os
import collections
import re
from nltk.chunk import conlltags2tree, tree2conlltags
ner_tags = collections.Counter()

corpus_root = "gmb-2.2.0"   # Make sure you set the proper path to the unzipped corpus

for root, dirs, files in os.walk(corpus_root):
    for filename in files:
        if filename.endswith(".tags"):
            with open(os.path.join(root, filename), 'rb') as file_handle:
                file_content = file_handle.read().decode('utf-8').strip()
                annotated_sentences = file_content.split('\n\n')   # Split sentences
                for annotated_sentence in annotated_sentences:
                    annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]  # Split words

                    standard_form_tokens = []

                    for idx, annotated_token in enumerate(annotated_tokens):
                        annotations = annotated_token.split('\t')   # Split annotations
                        word, tag, ner = annotations[0], annotations[1], annotations[3]

                        ner_tags[ner] += 1

print ner_tags

def to_conll_iob(annotated_sentence):
    """
    `annotated_sentence` = list of triplets [(w1, t1, iob1), ...]
    Transform a pseudo-IOB notation: O, PERSON, PERSON, O, O, LOCATION, O
    to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    """
    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sentence):
        tag, word, ner = annotated_token

        if ner != 'O':
            if idx == 0:
                ner = "B-" + ner
            elif annotated_sentence[idx - 1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        proper_iob_tokens.append((tag, word, ner))
    return proper_iob_tokens


def read_gmb(corpus_root):
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]

                        standard_form_tokens = []

                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]

                            if ner != 'O':
                                ner = ner.split('-')[0]

                            if tag in ('LQU', 'RQU'):   # Make it NLTK compatible
                                tag = "``"

                            standard_form_tokens.append((word, tag, ner))

                        conll_tokens = to_conll_iob(standard_form_tokens)

                        # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                        # Because the classfier expects a tuple as input, first item input, second the class
                        yield [((w, t), iob) for w, t, iob in conll_tokens]

reader = read_gmb(corpus_root)
# def features(tokens, index, history):
#     """
#     `tokens`  = a POS-tagged sentence [(w1, t1), ...]
#     `index`   = the index of the token we want to extract features for
#     `history` = the previous predicted IOB tags
#     """
#
#     # init the stemmer
#     stemmer = SnowballStemmer('english')
#
#     # Pad the sequence with placeholders
#     tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
#     history = ['[START2]', '[START1]'] + list(history)
#
#     # shift the index with 2, to accommodate the padding
#     index += 2
#
#     word, pos = tokens[index]
#     prevword, prevpos = tokens[index - 1]
#     prevprevword, prevprevpos = tokens[index - 2]
#     nextword, nextpos = tokens[index + 1]
#     nextnextword, nextnextpos = tokens[index + 2]
#     previob = history[index - 1]
#     contains_dash = '-' in word
#     contains_dot = '.' in word
#     allascii = all([True for c in word if c in string.ascii_lowercase])
#
#     allcaps = word == word.capitalize()
#     capitalized = word[0] in string.ascii_uppercase
#
#     prevallcaps = prevword == prevword.capitalize()
#     prevcapitalized = prevword[0] in string.ascii_uppercase
#
#     nextallcaps = prevword == prevword.capitalize()
#     nextcapitalized = prevword[0] in string.ascii_uppercase
#
#     return {
#         'word': word,
#         'lemma': stemmer.stem(word),
#         'pos': pos,
#         'all-ascii': allascii,
#
#         'next-word': nextword,
#         'next-lemma': stemmer.stem(nextword),
#         'next-pos': nextpos,
#
#         'next-next-word': nextnextword,
#         'nextnextpos': nextnextpos,
#
#         'prev-word': prevword,
#         'prev-lemma': stemmer.stem(prevword),
#         'prev-pos': prevpos,
#
#         'prev-prev-word': prevprevword,
#         'prev-prev-pos': prevprevpos,
#
#         'prev-iob': previob,
#
#         'contains-dash': contains_dash,
#         'contains-dot': contains_dot,
#
#         'all-caps': allcaps,
#         'capitalized': capitalized,
#
#         'prev-all-caps': prevallcaps,
#         'prev-capitalized': prevcapitalized,
#
#         'next-all-caps': nextallcaps,
#         'next-capitalized': nextcapitalized,
#     }
def _english_wordlist():
        from nltk.corpus import words
        _en_wordlist = set(words.words('en-basic'))
        wl = _en_wordlist
        return wl

def shape(word):
    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word, re.UNICODE):
        return 'number'
    elif re.match('\W+$', word, re.UNICODE):
        return 'punct'
    elif re.match('\w+$', word, re.UNICODE):
        if word.istitle():
            return 'upcase'
        elif word.islower():
            return 'downcase'
        else:
            return 'mixedcase'
    else:
        return 'other'

def simplify_pos(s):
    if s.startswith('V'): return "V"
    else: return s.split('-')[0]

def features(tokens, index, history):
        word = tokens[index][0]
        pos = simplify_pos(tokens[index][1])
        if index == 0:
            prevword = prevprevword = None
            prevpos = prevprevpos = None
            prevshape = prevtag = prevprevtag = None
        elif index == 1:
            prevword = tokens[index-1][0].lower()
            prevprevword = None
            prevpos = simplify_pos(tokens[index-1][1])
            prevprevpos = None
            prevtag = history[index-1][0]
            prevshape = prevprevtag = None
        else:
            prevword = tokens[index-1][0].lower()
            prevprevword = tokens[index-2][0].lower()
            prevpos = simplify_pos(tokens[index-1][1])
            prevprevpos = simplify_pos(tokens[index-2][1])
            prevtag = history[index-1]
            prevprevtag = history[index-2]
            prevshape = shape(prevword)
        if index == len(tokens)-1:
            nextword = nextnextword = None
            nextpos = nextnextpos = None
        elif index == len(tokens)-2:
            nextword = tokens[index+1][0].lower()
            nextpos = tokens[index+1][1].lower()
            nextnextword = None
            nextnextpos = None
        else:
            nextword = tokens[index+1][0].lower()
            nextpos = tokens[index+1][1].lower()
            nextnextword = tokens[index+2][0].lower()
            nextnextpos = tokens[index+2][1].lower()

        # 89.6
        feature = {
            'bias': True,
            'shape': shape(word),
            'wordlen': len(word),
            'prefix3': word[:3].lower(),
            'suffix3': word[-3:].lower(),
            'pos': pos,
            'word': word,
            'en-wordlist': (word in _english_wordlist()),
            'prevtag': prevtag,
            'prevpos': prevpos,
            'nextpos': nextpos,
            'prevword': prevword,
            'nextword': nextword,
            'word+nextpos': '{0}+{1}'.format(word.lower(), nextpos),
            'pos+prevtag': '{0}+{1}'.format(pos, prevtag),
            'shape+prevtag': '{0}+{1}'.format(prevshape, prevtag),
            }

        return feature

class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)
        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)

reader = read_gmb(corpus_root)
data = list(reader)
#training_samples = data[:int(len(data) * 0.01)]
training_samples = data[:10]
print training_samples[0]
training_samples.append([(('151-32-2558', 'CD'), u'B-id'), ((',', ','), u'O'), (('1952/11/19', 'CD'), u'B-tim'), ((',', ','), u'O'), (('Stockdale', 'NNP'), u'B-PERSON'),
((',', ','), u'O'), (('Zwick', 'NNP'), u'B-PERSON'), ((',', ','), u'O'), (('Rebecca,784', 'NNP'), u'O'), (('Beechwood', 'NNP'), u'B-PERSON'), (('Avenue', 'NNP'), u'I-PERSON'),
 ((',', ','), u'O'), (('Piscataway', 'NNP'), u'B-GPE'), ((',', ','), u'O'), (('NJ,8854,908-814-6733', 'NNP'), u'O'), ((',', ','), u'O'), (('rzwick', 'NN'), u'O'),
  (('@', 'NNP'), u'O'), (('domain.com', 'NN'), u'O'), ((',', ','), u'O'), (('5252', 'CD'), u'B-credit'),
(('5971', 'CD'), u'I-credit'), (('4219', 'CD'), u'I-Credit'), (('4116', 'CD'), 'I-Credit'), (('id', u'IN'), u'O'), (('gender', u'NN'), u'O'), (('birthdate', u'NN'), u'O'),
(('maiden_name', u'NN'), u'O'), (('lname', u'NN'), u'O'), (('fname', u'NN'), u'B-tim'), (('address', u'NN'), u'O'), (('city', u'NN'), u'O'), (('state', u'NN'), u'O'),
(('zip', u'NN'), u'O'), (('phone', u'NN'), u'B-tim'), (('email', u'NN'), u'B-tim'), (('cc_type', u'NN'), u'O'), (('cc_number', u'NN'), u'O'), (('cc_cvc', u'NN'), u'O'),
(('cc_expiredate', u'NN'), u'O'), (('172-32-1176', u'CD'), u'B-id'), (('m', u'NN'), u'B-Gender'), (('1958/04/21', u'CD'), u'I-tim'), (('Smith', u'NNP'), u'B-per'),
(('White', u'NNP'), u'I-per'), (('Johnson', u'NNP'), u'I-per'), (('10932', u'CD'), u'B-LOC'), (('Bigge', u'NNP'), u'B-LOC'), (('Rd', u'NNP'), u'I-LOC'),
(('Menlo', u'NNP'), u'I-LOC'), (('Park', u'NNP'), u'I-LOC'), (('CA', u'NNP'), u'I-LOC'), (('94025', u'CD'), u'B-tim'), (('408', u'CD'), u'B-Phone'),
(('496-7223', u'CD'), u'I-Phone'), (('jwhite@domain.com', u'EMAIL'), u'O'), (('m', u'NN'), u'B-Gender'), (('5270', u'CD'), u'B-Credit'), (('4267', u'CD'), u'I-Credit'),
(('6450', u'CD'), u'I-Credit'), (('5516', u'CD'), u'I-Credit'), (('123', u'CD'), u'B-cvv'), (('2010/06/25', u'CD'), u'B-tim'), (('514-14-8905', u'CD'), u'B-id'),
(('f', u'NN'), u'B-Gender'), (('1944/12/22', u'CD'), u'B-tim')])

test_samples = data[int(len(data) * 0.09):]

print "#training samples = %s" % len(training_samples)    # training samples = 55809
print "#test samples = %s" % len(test_samples)

chunker = NamedEntityChunker(training_samples[:2000])
#from nltk import  word_tokenize, wordpunct_tokenize
from train import pos_tag
entities_list = []
import re
HANG_RE = re.compile(r',')
def tokenize(text):
    return HANG_RE.sub(r' ', text).split()
print chunker.parse(pos_tag(tokenize("I'm going to Germany this Monday on 10/12/2018.")))
#print nltk.TreebankWordTokenizer().tokenize("151-32-2558,f,1952/11/19,Stockdale,Zwick,Rebecca,784 Beechwood Avenue,Piscataway,NJ,8854,908-814-6733,rzwick@domain.com,v,5252597142194116,173,2011/02/01")
print chunker.parse(pos_tag(tokenize("@steve,151-32-2558,f,1952/11/19,Stockdale,Zwick,Rebecca,784 Beechwood Avenue,Piscataway,NJ,8854,908-814-6733,rzwick@domain.com,v,5252597142194116,173,2011/02/01")))
print chunker.parse(pos_tag(tokenize("044-34-6954,m,1967/05/28,Simpson,Lowe,Tim,1620 Maxwell Street,East Hartford,CT,6108,860-755-0293,tlowe@domain.com,m,5144 8691 2776 1108,616,2011/10/01")))
#print chunker.parse(pos_tag(tokenize('''My name is Indigo Montoya. I am from the Congo. Oh my gosh I accidentally included my credit card number 14320099 and passport P123411.
#RT best tweet evarrrr @123fake @justinbieber @BarackObama. Hello my email is fakeperson@example.com and I am here. Today is June, 2008-06-29 ''')))

with open('sample-data.csv', 'r') as myfile:
    text=myfile.read().replace('\n', '')

#print text
print "-------\n"
print "Taggged Data"
entities_list = []
for sent in nltk.sent_tokenize(text):
    for chunk in chunker.parse(pos_tag(nltk.TweetTokenizer().tokenize(sent))):
        entities_list.append(chunk)

#print entities_list
