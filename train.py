import nltk
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import json
import pickle
import re
#tagged_sentences = list(nltk.corpus.treebank.tagged_sents())
#tagged_sentences = []
#tagged_sentences.append()
#with open('social.txt', 'r') as myfile:
#    tagged_sentences=json.dumps(myfile.read().replace('\n', ''))
tagged_sentences = pickle.load(open('social.pickle','rb'))
#tagged_sentences.insert(0,[(u'johnsmith65@email.com', u'EMAIL'), (u'john_smith@email.com', u'EMAIL'), (u'smith.j@email.com', u'EMAIL'), (u'johnpsmith@email.com', u'EMAIL'), (u'jp_smith65@email.com', u'EMAIL')])
#print tagged_sentences
print "Tagged words:", len(nltk.corpus.treebank.tagged_words())


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:],
        'is_first_symbol': sentence[index][0] == '@',
        #'is_email': re.findall(r'[\w.-]+@[\w.-]+',sentence[index][0]) == None
    }

import pprint
pprint.pprint(features(['This', 'is', 'a', 'abc@sentence.com'], 3))

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

cutoff = int(.85 * len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:]

print len(training_sentences)
print len(test_sentences)

def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])

    return X, y
X, y = transform_to_dataset(training_sentences)

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])

clf.fit(X[:1000], y[:1000])   # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)

print 'Training completed'

X_test, y_test = transform_to_dataset(test_sentences)

print "Accuracy:", clf.score(X_test, y_test)

def pos_tag(sentence):
    tagged_sentence = []
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return zip(sentence, tags)
HANG_RE = re.compile(r',')
def tokenize(text):
    return HANG_RE.sub(r' ', text).split()
#print pos_tag(tokenize('''My name is Indigo Montoya. I am from the Congo. Oh my gosh I accidentally included my credit card number 14320099 and passport P123411.
#RT best tweet evarrrr @123fake @justinbieber @BarackObama. Hello my email is fakeperson@example.com and I am here. Today is June, 2008-06-29 '''))
#print nltk.ne_chunk(pos_tag(tokenize('''My name is Indigo Montoya. I am from the Congo. Oh my gosh I accidentally included my credit card number 14320099 and passport P123411.
#RT best tweet evarrrr @123fake @justinbieber @BarackObama. Hello my email is fakeperson@example.com and I am here. Today is June, 2008-06-29 ''')))
with open('sample-data.csv', 'r') as myfile:
    text=myfile.read().replace('\n', '')

#print text
print "-------\n"
print "Taggged Data"
entities_list = []
from chunk import chunker
for sent in nltk.sent_tokenize(text):
    #print pos_tag(tokenize(sent))
    for chunk in chunker.parse(pos_tag(tokenize(sent))):
#        print chunk
        entities_list.append(chunk)

with open('out.txt', 'w') as file:
    file.write(str(entities_list))
#print entities_list
import unicodecsv as csv
from commonregex import CommonRegex
from nltk.tag.stanford import StanfordNERTagger
parser = CommonRegex()
standford_ner = StanfordNERTagger('classifiers/english.conll.4class.distsim.crf.ser.gz','stanford-ner.jar')
people = []
organizations = []
locations = []
emails = []
phone_numbers = []
street_addresses = []
credit_cards = []
ips = []
data = []

with open('sample-data.csv', 'r') as filedata:
    reader = csv.reader(filedata)
    for row in reader:
        data.extend(row)
    for text in row:
                emails.extend(parser.emails(text))
                phone_numbers.extend(parser.phones("".join(text.split())))
                street_addresses.extend(parser.street_addresses(text))
                credit_cards.extend(parser.credit_cards(text))
                ips.extend(parser.ips(text))

for title, tag in standford_ner.tag(set(data)):
            if tag == 'PERSON':
                people.append(title)
            if tag == 'LOCATION':
                locations.append(title)
            if tag == 'ORGANIZATION':
                organizations.append(title)

sa = {'people': people, 'locations': locations, 'organizations': organizations,
                'emails': emails, 'phone_numbers': phone_numbers, 'street_addresses': street_addresses,
                'credit_cards': credit_cards, 'ips': ips
                }
print sa
