import nltk
import re

with open('data.txt', 'r') as myfile:
    text=myfile.read().replace('\n', '')

print text
print "-------\n"
print "Taggged Data"
entities_list = []
for sent in nltk.sent_tokenize(text):
    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
        print chunk
        entities_list.append(chunk)

print entities_list
ids_list = []
for sent in nltk.sent_tokenize(text):
    for word in nltk.word_tokenize(sent):
        if len(word) >= 4 and any(char.isdigit() for char in word):
            ids_list.append(("ID", word))

print "\nFound IDs"
print  ids_list

""" Returns e-mail addresses [tag: EMAIL] """
emails_regex = "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}"
emails_re = re.compile(emails_regex)
emails_list = [("EMAIL", email) for email in emails_re.findall(text)]
#print "Found Emails /n"
emails_list

""""Returns Twitter usernames of form @handle
        (alphanumerical and "_", max length: 15) [tag: USERNAME]
        """
# twitter_regex = r'\[A-Za-z0-9_]{1,15}'
twitter_regex = r'^|[^@\w](@\w{1,15})\b'
twitter_re = re.compile(twitter_regex)
twitter_list = [( twitter, "TWITTER") for twitter in twitter_re.findall(
            text) if twitter != ""]
print "\nFound Social handles"
print twitter_list


patterns = [(r'^|[^@\w](@\w{1,15})\b', 'TWITTER'),(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)",'EMAIL') ]
#regexp_tagger = nltk.RegexpTagger(patterns)
default_tagger = nltk.data.load("taggers/maxent_treebank_pos_tagger/english.pickle")
# Build a tagger that add reg_tagger in an existing tagger (MaxEnt)
#tagger = nltk.DefaultTagger("TWITTER")
#print [entities_list + twitter_list]
#t1 = nltk.BigramTagger(entities_list + twitter_list)
print "-------\n"
print "Taggged Data"
st = nltk.tag.StanfordNERTagger('/home/susmit/nltk_data/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz',
        '/home/susmit/nltk_data/stanford-ner-2018-02-27/stanford-ner.jar')
entities_list1 = []
for sent in nltk.sent_tokenize(text):
    print st.tag(nltk.word_tokenize(sent))
    for chunk in nltk.ne_chunk(st.tag(nltk.word_tokenize(sent))):
        #print chunk
        entities_list1.append(chunk)
print entities_list1
