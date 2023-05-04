import numpy as np
import pandas as pd
import re
import nltk
import csv
import codecs
import datetime
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# load stop words
stop_words = set(stopwords.words('english'))

# back to word
lemmatizer = WordNetLemmatizer()


# preprocess the text
def preprocess_text(text):
    # remove URL
    text = re.sub(r"http\S+", "", text)

    # remove coma and special text
    text = re.sub(r'[^\w\s]', '', text)

    # transfer text into lower capital
    text = text.lower()

    # spilt the word
    words = nltk.word_tokenize(text)

    # remove stop word
    words = [word for word in words if word not in stop_words]

    # reshade the word
    words = [lemmatizer.lemmatize(word) for word in words]

    # return text after processing
    return ' '.join(words)


# function for pair the word into lexicon
def word_sentiment(word):
    sentiment = np.array([0, 0, 0, 0, 0, 0])
    sentiment_s = np.array([0, 0, 0, 0, 0, 0])
    with open("dict.csv", encoding="GBK") as lexicon:
        # ignore the first line
        lexicon.readline()
        # return the content(in lines) as list
        lexicon_obj = lexicon.readlines()

        # through the element in the list
        for line in lexicon_obj:
            lists = line.rstrip("\n").split(",")

            # research the word
            if word == lists[0]:
                # 7-13 stands for 6 dimensions
                sentiment = lists[7:13]
                # transfer list into int
                sentiment_s = [int(i) for i in sentiment]
    # return the sentiment vector
    return sentiment_s


# re-encoding

# original CSV encode with Windows-1254
with codecs.open('elonmusk.csv', 'r', encoding='GB2312',  errors='ignore') as f_in:
    reader = csv.reader(f_in)

    # new csv encode with gbk(create the new file first)
    with codecs.open('elonmusk twitter.csv', 'w', encoding='gbk', errors='ignore') as f_out:
        writer = csv.writer(f_out)

        # read csv
        for row in reader:
            # gain line 4(text)
            text = row[4]

            # process the text within function
            processed_text = preprocess_text(text)

            # return to line 4
            row[4] = processed_text

            # return to the file
            writer.writerow(row)

# start to mark the tweets
with open("elonmusk twitter.csv", encoding="GBK") as data:
    # ignore the first line
    data.readline()
    # return the content(in lines) as list
    data_obj = data.readlines()
    # news for storing all sentiments scores
    news = []
    # each line stands for a tweet
    for line in data_obj:
        # spilt the text
        lists = line.rstrip("\n").split(",")
        # line 4 is the text
        text1 = lists[4]
        # extract the date, save as index, remove time
        date_str = lists[3]
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        date_str = date_obj.strftime("%Y-%m-%d")

        # new array for storing the score
        score = np.array([0, 0, 0, 0, 0, 0])

        # loop for a single tweet
        for i in text1.upper().split():
            # sum the aggregate score for a tweet
            score = np.add(score, word_sentiment(i))
        # store in temp
        temp = [date_str, score]

        #add into news
        news.append(temp)

# as it is a massive dataset, we would like to store the variable outside the python
df = pd.DataFrame(news, columns=['date', 'sentiments'])
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = (df.resample('d').sum())

# save the accumulated sentiments
df['accumulated'] = df.apply(lambda x: x.cumsum().tolist(), axis=0)

# save the df into a pkl file
f = open('df.pkl', 'wb')
pickle.dump(df, f)
f.close()
