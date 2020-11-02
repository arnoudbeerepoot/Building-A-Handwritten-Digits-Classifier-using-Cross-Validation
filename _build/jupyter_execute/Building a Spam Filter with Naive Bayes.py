#!/usr/bin/env python
# coding: utf-8

# # Building a Spam Filter with Naive Bayes

# In this project a spam filter is built based on the Naive Bayes algorithm. It will be used to classify SMS messages as spam or non-spam. The spam-filter is being trained with a dataset of 5572 SMS messages that are already classified by humans (The UCI Machine Learning Repository) (https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
# See wikipedia for background information (https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

# ## Import dataset and analyze the data

# In[1]:


import pandas as pd

data = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['Label', 'SMS']) 
data.info()


# In[2]:


# show the first five rows
data.head(5)


# In[3]:


# how many spam vs non-spam messages are in the dataset?
data['Label'].value_counts()


# In[4]:


percentage_spam = round(747/(4825+747)*100,1)
"In the dataset {} % of the messages is spam".format(percentage_spam)


# ## Create a train and test set

# In[5]:


# Randomize the entire dataset
random = data.sample(frac=1, random_state=1)


# In[6]:


# Split the randomized dataset in a train and test set
# 80% is training that is 4458 records
train = random[:4458].reset_index(drop=True)
test = random[4458:].reset_index(drop=True)
train.info()


# In[7]:


# number of ham and spam messages in the train set
train['Label'].value_counts()


# In[8]:


percentage_spam_train = round(600/(3858+600)*100,1)
"In the training dataset {} % of the messages is spam".format(percentage_spam_train)


# In[9]:


# number of ham and spam messages in the test set
test['Label'].value_counts()


# In[10]:


percentage_spam_test = round(147/(967+147)*100,1)
"In the test dataset {} % of the messages is spam".format(percentage_spam_test)


# Percentages of spam messages in train en test set are in the same range as the original dataset.

# In[11]:


train.head(10)


# ## Clean the dataset

# In[12]:


# remove punctuation
train['SMS'] = train['SMS'].str.replace('\W', ' ')
train.head(10)


# In[13]:


# Transform every letter in every word to lowercase
train['SMS'] = train['SMS'].str.lower()
train.head(5)


# In[14]:


# Create a vocabulary for the messages in the training set.
# The vocabulary should be a Python list containing all the unique words across all messages, 
# where each word is represented as a string.

# Create a list of words
train['SMS'] = train['SMS'].str.split()
vocabulary = []

for item in train['SMS']:
    for word in item:
        vocabulary.append(word)
vocabulary = list(set(vocabulary))
vocabulary


# In[15]:


# Create dictionary with number of words in every SMS
word_counts_per_sms = {unique_word: [0] * len(train['SMS']) for unique_word in vocabulary}

for index, sms in enumerate(train['SMS']):
    for word in sms:
        word_counts_per_sms[word][index] += 1


# In[16]:


word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()


# In[17]:


training_set_clean = pd.concat([train, word_counts], axis=1)
training_set_clean.head()


# In[18]:


# Calculate the probabilities P(Spam) and P(Ham)

# P(Spam) = Label(Spam)/Total 

p_spam = len(training_set_clean[training_set_clean['Label']=='spam'])/len(training_set_clean)
p_spam


# In[19]:


# P(Ham) = Label(Ham)/Total

p_ham = len(training_set_clean[training_set_clean['Label']=='ham'])/len(training_set_clean)
p_ham


# In[20]:


# Calculate N_Spam; total number of words in all spam messages

n_spam = 0

for item in training_set_clean[training_set_clean['Label']=='spam']['SMS']:
    n_spam += len(item)
n_spam


# In[21]:


# Calculate N_Ham; total number of words in all ham messages

n_ham = 0

for item in training_set_clean[training_set_clean['Label']=='ham']['SMS']:
    n_ham += len(item)
n_ham


# In[22]:


# Calculate N_Vocabulary; Total number of unique words in all messages

n_vocabulary = len(vocabulary)
n_vocabulary


# In[23]:


# In the calculations Laplace smoothing with a value of 1 is used

alpha = 1


# In[24]:


# Calculate the parameters P(wi|Spam) and P(wi|Ham)

p_wi_spam = {}
p_wi_ham = {}

for item in vocabulary:
    p_wi_spam[item] = 0
    p_wi_ham[item] = 0
    
spam = training_set_clean[training_set_clean['Label']=='spam']
ham = training_set_clean[training_set_clean['Label']=='ham']

for word in vocabulary:
    n_word_given_spam = spam[word].sum()
    p_word_given_spam = (n_word_given_spam + alpha)/(n_spam + alpha * n_vocabulary)
    p_wi_spam[word] = p_word_given_spam
    
    n_word_given_ham = ham[word].sum()
    p_word_given_ham = (n_word_given_ham + alpha)/(n_ham + alpha * n_vocabulary)
    p_wi_ham[word] = p_word_given_ham
    
p_wi_spam    


# In[25]:


# Function to be used as spam filter

import re

def classify(message):

    message = re.sub('\W', ' ', message)
    message = message.lower()
    message = message.split()

    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    
    for word in message:
        if word in p_wi_spam:
            p_spam_given_message *= p_wi_spam[word]
        if word in p_wi_ham:
            p_ham_given_message *= p_wi_ham[word]

    print('P(Spam|message):', p_spam_given_message)
    print('P(Ham|message):', p_ham_given_message)

    if p_ham_given_message > p_spam_given_message:
        print('Label: Ham')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal proabilities, have a human classify this!')


# In[26]:


classify('WINNER!! This is the secret code to unlock the money: C3421.')


# In[27]:


classify('Sounds good, Tom, then see u there')


# In[28]:


# Test the accuracy of the algorithme

def classify_test_set(message):

    message = re.sub('\W', ' ', message)
    message = message.lower()
    message = message.split()

    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in message:
        if word in p_wi_spam:
            p_spam_given_message *= p_wi_spam[word]

        if word in p_wi_ham:
            p_ham_given_message *= p_wi_ham[word]

    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_spam_given_message > p_ham_given_message:
        return 'spam'
    else:
        return 'needs human classification'


# In[29]:


test['predicted'] = test['SMS'].apply(classify_test_set)
test.head()


# In[30]:


# Accuracy of the spam filter
correct = 0
total = len(test)

for row in test.iterrows():
    row = row[1]
    if row['Label'] == row['predicted']:
        correct += 1
        
print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)


# 
# The accuracy is close to 98.74%, which is really good. Our spam filter looked at 1,114 messages that it hasn't seen in training, and classified 1,100 correctly.
# 
# Next Steps
# In this project, we managed to build a spam filter for SMS messages using the multinomial Naive Bayes algorithm. The filter had an accuracy of 98.74% on the test set we used, which is a pretty good result. Our initial goal was an accuracy of over 80%, and we managed to do way better than that.
# 
# Next steps include:
# 
# - Analyze the 14 messages that were classified incorrectly and try to figure out why the algorithm classified them incorrectly
# - Make the filtering process more complex by making the algorithm sensitive to letter case
# - Get the project portfolio-ready by using a few tips from our style guide for data science projects.

# In[ ]:




