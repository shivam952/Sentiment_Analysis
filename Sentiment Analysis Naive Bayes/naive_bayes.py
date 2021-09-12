from utils2 import process_tweet,lookup
import nltk
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd

# get the sets of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# avoid assumptions about the length of all_positive_tweets
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

def count_tweets(result, tweets, ys): # dictionary : tuple:freq,,, tuple:word,label
    for y,tweet in zip(ys,tweets):
        for word in process_tweet(tweet):
            pair = (word,y)
            
            if pair in result:
                result[pair]+=1
            else:
                result[pair]=1
    return result  


freqs = count_tweets({}, train_x, train_y) #freq dictionary

#train my model
def train_naivebayes(freq,train_x,train_y):
    loglikelihood = {}
    logprior=0
    
    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    
    # calculate N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = V_pos = V_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:
            # increment the count of unique positive words by 1
            V_pos += 1

            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]

        # else, the label is negative
        else:
            # increment the count of unique negative words by 1
            V_neg += 1

            # increment the number of negative words by the count for this (word,label) pair
            N_neg += freqs[pair]
    D_pos = (len(list(filter(lambda x: x > 0, train_y))))

    # Calculate D_neg, the number of negative documents
    D_neg = (len(list(filter(lambda x: x <= 0, train_y))))
    logprior = np.log(D_pos) - np.log(D_neg)
    
    for word in vocab:
        freq_pos= lookup(freqs, word, 1)
        freq_neg = lookup(freqs,word,0)
        
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)
        
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)
    return logprior,loglikelihood

logprior, loglikelihood = train_naivebayes(freqs, train_x, train_y)
print(logprior)
print(len(loglikelihood))

def predict_naiveBayes(tweet,logprior,loglikelihood):
    word_l = process_tweet(tweet)

    # initialize probability to zero
    p = 0

    # add the logprior
    p += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]
    return p


def test_naiveBayes(test_x,test_y,logprior,loglikelihood):
    y_hat=[]
    for tweet in test_x:
        if predict_naiveBayes(tweet,logprior,loglikelihood) >0:
            y_hat.append(1)
        else:
            y_hat.append(0)
    error = np.mean(np.absolute(y_hat-test_y))    
    accuracy = 1-error

    

    return accuracy


print("Naive Bayes accuracy = %0.4f" %
      (test_naiveBayes(test_x, test_y, logprior, loglikelihood)))

my_tweet = 'you are bad :('
predict_naiveBayes(my_tweet, logprior, loglikelihood)

    
        
    
    
    
        
    
    
    
    
    

         
