import nltk
import numpy as np
import csv
import collections
import random

sent_scores = collections.defaultdict(list)

with open('data/SentiWordNet_3.0.0_20130122.txt', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t',quotechar='"')
    for line in reader:
        if line[0].startswith('#') or len(line) == 1:
            continue
        POS, ID, PosScore, NegScore, SynsetTerms, Gloss = line
        if len(POS)==0 or len(ID)==0:
            continue
        for term in SynsetTerms.split(" "):
            term = term.split('#')[0].replace("-", " ").replace("_", " ")
            key = "%s/%s"%(POS,term.split("#")[0])
            sent_scores[key].append((float(PosScore),float(NegScore)))

for key, value in sent_scores.iteritems():
    sent_scores[key] = np.mean(value,axis=0)

#-------------------------------
# POS tagging
#-------------------------------
def score(word, pos):
    if pos[0:2] == 'NN':
        pos_type = 'n'
    elif pos[0:2] == 'JJ':
        pos_type = 'a'
    elif pos[0:2] == 'VB':
        pos_type = 'v'
    elif pos[0:2] == 'RB':
        pos_type = 'r'
    else:
        pos_type = 0
    if pos_type != 0:
        key = pos_type + '/' + word
        score = sent_scores[key]
        if len(score) == 2:
            return score
        else:
            return np.array([0.0,0.0])
    else:
        return np.array([0.0,0.0])

#--------------------------------
# Score words and sentences 
# according to POS tag
#--------------------------------
def score_word(word):
    pos = nltk.pos_tag([word])[0][1]
    return score(word, pos)

def score_sentence(sentence):
    pos = nltk.pos_tag(sentence)
    mean_score = np.array([0.0,0.0])
    for j in range(len(pos)):
        mean_score += score(pos[j][0], pos[j][1])
    return mean_score

def pos_vector(sentence):
    posvector = nltk.pos_tag(sentence)
    vector = np.zeros(4)
    for j in range(len(sentence)):
        pos = posvector[j][1]
        if pos[0:2] == 'NN':
            vector[0]+=1
        elif pos[0:2] == 'JJ':
            vector[1]+=1
        elif pos[0:2] == 'VB':
            vector[2]+=1
        elif pos[0:2] == 'RB':
            vector[3]+=1
    return vector

#-----------------------------------------------------------
# Unigrams,Bigrams, Capitals, Sentiment Feature Extraction
#-----------------------------------------------------------

def feature_extract(sentence):
	feature_array = {}
    
    # tokenizing
	tokens = nltk.word_tokenize(sentence)
	tokens = [(t.lower()) for t in tokens]
    
    # lemmatization
	#porter = nltk.PorterStemmer()
	#unigrams = [porter.stem(t) for t in tokens]
	lemmatizer = nltk.WordNetLemmatizer()
	unigrams= [lemmatizer.lemmatize(u) for u in tokens]
	bigrams = nltk.bigrams(unigrams)
	bigrams = [tup[0]+' ' +tup[1] for tup in bigrams]
	grams = unigrams + bigrams
	for t in grams:
		feature_array['contains(%s)' % t] = 1.0
    
    # sentiment analysis
	mean_sentiment = score_sentence(tokens)
	feature_array['positive sentiment'] = mean_sentiment[0]
	feature_array['negative sentiment'] = mean_sentiment[1]
	feature_array['sentiment'] = mean_sentiment[0]-mean_sentiment[1]
    
    # parts of speech counter
	posvector = pos_vector(tokens)
	for j in range(len(posvector)):
		feature_array['POS' + str(j+1)] = posvector[j]
    
    # capitalization counter
	counter = 0
	threshold = 4
	for j in range(len(sentence)):
		counter += int(sentence[j].isupper())
	feature_array['capitalization'] = int(counter>=threshold)
	
	# textblob 
	try:
		blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip())
		feature_array['Blob sentiment'] = blob.sentiment.polarity
		feature_array['Blob subjectivity'] = blob.sentiment.subjectivity
	except:
		feature_array['Blob sentiment'] = 0.0
		feature_array['Blob subjectivity'] = 0.0
    
    #Split in 2
	if len(tokens)==1:
		tokens+=['.']
	first_half = tokens[0:len(tokens)/2]
	second_half = tokens[len(tokens)/2:]
    
    
	mean_sentiment_first = score_sentence(first_half)
	feature_array['Positive sentiment 1/2'] = mean_sentiment_first[0]
	feature_array['Negative sentiment 1/2'] = mean_sentiment_first[1]
	feature_array['Sentiment 1/2'] = mean_sentiment_first[0]-mean_sentiment_first[1]
    
	mean_sentiment_second = score_sentence(second_half)
	feature_array['Positive sentiment 2/2'] = mean_sentiment_second[0]
	feature_array['Negative sentiment 2/2'] = mean_sentiment_second[1]
	feature_array['Sentiment 2/2'] = mean_sentiment_second[0]-mean_sentiment_second[1]
    
	feature_array['Sentiment contrast 2'] = np.abs(feature_array['Sentiment 1/2']-feature_array['Sentiment 2/2'])

     #TextBlob sentiment analysis
	try:
		blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in first_half]).strip())
		feature_array['Blob sentiment 1/2'] = blob.sentiment.polarity
		feature_array['Blob subjectivity 1/2'] = blob.sentiment.subjectivity
	except:
		feature_array['Blob sentiment 1/2'] = 0.0
		feature_array['Blob subjectivity 1/2'] = 0.0
	try:
		blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in second_half]).strip())
		feature_array['Blob sentiment 2/2'] = blob.sentiment.polarity
		feature_array['Blob subjectivity 2/2'] = blob.sentiment.subjectivity
	except:
		feature_array['Blob sentiment 2/2'] = 0.0
		feature_array['Blob subjectivity 2/2'] = 0.0
        
	feature_array['Blob Sentiment contrast 2'] = np.abs(feature_array['Blob sentiment 1/2']-feature_array['Blob sentiment 2/2'])

    #Split in 3
	if len(tokens)==2:
		tokens+=['.']
	first_half = tokens[0:len(tokens)/3]
	second_half = tokens[len(tokens)/3:2*len(tokens)/3]
	third_half = tokens[2*len(tokens)/3:]
    
	mean_sentiment_first = score_sentence(first_half)
	feature_array['Positive sentiment 1/3'] = mean_sentiment_first[0]
	feature_array['Negative sentiment 1/3'] = mean_sentiment_first[1]
	feature_array['Sentiment 1/3'] = mean_sentiment_first[0]-mean_sentiment_first[1]
    
	mean_sentiment_second =score_sentence(second_half)
	feature_array['Positive sentiment 2/3'] = mean_sentiment_second[0]
	feature_array['Negative sentiment 2/3'] = mean_sentiment_second[1]
	feature_array['Sentiment 2/3'] = mean_sentiment_second[0]-mean_sentiment_second[1]
    
	mean_sentiment_third = score_sentence(third_half)
	feature_array['Positive sentiment 3/3'] = mean_sentiment_third[0]
	feature_array['Negative sentiment 3/3'] = mean_sentiment_third[1]
	feature_array['Sentiment 3/3'] = mean_sentiment_third[0]-mean_sentiment_third[1]

	feature_array['Sentiment contrast 3'] = np.abs(feature_array['Sentiment 1/3']-feature_array['Sentiment 3/3'])

	#TextBlob sentiment analysis
	try:
		blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in first_half]).strip())
		feature_array['Blob sentiment 1/3'] = blob.sentiment.polarity
		feature_array['Blob subjectivity 1/3'] = blob.sentiment.subjectivity
	except:
		feature_array['Blob sentiment 1/3'] = 0.0
		feature_array['Blob subjectivity 1/3'] = 0.0
	try:
		blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in second_half]).strip())
		feature_array['Blob sentiment 2/3'] = blob.sentiment.polarity
		feature_array['Blob subjectivity 2/3'] = blob.sentiment.subjectivity
	except:
		feature_array['Blob sentiment 2/3'] = 0.0
		feature_array['Blob subjectivity 2/3'] = 0.0
	try:
		blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in third_half]).strip())
		features['Blob sentiment 3/3'] = blob.sentiment.polarity
		features['Blob subjectivity 3/3'] = blob.sentiment.subjectivity
	except:
		feature_array['Blob sentiment 3/3'] = 0.0
		feature_array['Blob subjectivity 3/3'] = 0.0

		feature_array['Blob Sentiment contrast 3'] = np.abs(feature_array['Blob sentiment 1/3']-feature_array['Blob sentiment 3/3'])

	return feature_array


########## Read Train and Test Data ##########

with open('data/train-balanced.csv', 'r') as f1:
    train_data = f1.read().rstrip().split('\n')
train_size = 20000
train_data = random.sample(train_data, train_size)
train_sents = []
train_labels = []
for line in train_data:
	if line.split('\t')[0] == '1':
		y = 1
	else:
		y = -1
	train_labels.append(y)
	train_sents.append(line.split('\t')[1])
train_features = []
for line in train_sents:
    train_features.append(feature_extract(line))

with open('data/test-balanced.csv', 'r') as f2:
    test_data = f2.read().rstrip().split('\n')
test_size = 4000
test_data = random.sample(test_data, test_size)
test_sents = []
test_labels = []
for line in test_data:
	if line.split('\t')[0] == '1':
		y = 1
	else:
		y = -1
	test_labels.append(y)
	test_sents.append(line.split('\t')[1])
test_features = []
for line in test_sents:
    test_features.append(feature_extract(line))
 

########## Generate a List of all Unique Features ##########

unique_features = []
for item in train_features:
	for key, value in item.items():
		if key not in unique_features:
			unique_features.append(key)
for item in test_features:
	for key, value in item.items():
		if key not in unique_features:
			unique_features.append(key)
print (len(unique_features)) #prints the number of unique features


########## Generate LIBSVM Files for Train and Test Data ##########

new_train_data = []
for i in range(len(train_features)):
	row = [train_labels[i]]
	item = train_features[i]
	for key, value in item.items():
		feature = str(unique_features.index(key)) + ':' + str(value)
		row.append(feature)
	new_train_data.append(row)

with open('data/extracted_features_train.csv', 'w') as f1:
    wr = csv.writer(f1, delimiter=' ')
    wr.writerows(new_train_data)

new_test_data = []
for i in range(len(test_features)):
	row = [test_labels[i]]
	item = test_features[i]
	for key, value in item.items():
		feature = str(unique_features.index(key)) + ':' + str(value)
		row.append(feature)
	new_test_data.append(row)
with open('data/extracted_features_test.csv', 'w') as f2:
    wr = csv.writer(f2, delimiter=' ')
    wr.writerows(new_test_data)