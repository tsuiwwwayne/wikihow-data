'''
Reference: https://github.com/mahnazkoupaee/WikiHow-Dataset/blob/master/process.py
This code is used to create article and summary files from the csv file.
The output of the file will be a directory of text files representing separate articles and their summaries.
Every file starts with the article followed by its summary, separated by the "@summary" tag.
'''
import pandas as pd
import os
import hashlib

# Reference: https://github.com/abisee/cnn-dailymail/blob/master/make_datafiles.py#L98
def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()

# read data from the csv file (from the location it is stored)
df = pd.read_csv(r'wikihowAll_cleaned.csv')
df = df.astype(str)
rows, columns = df.shape

# create a file to record the file names. This can be later used to divide the dataset in train/dev/test sets
title_file = open('titles.txt', 'wb')

# The path where the articles are to be saved
path = "articles"
if not os.path.exists(path): os.makedirs(path)

# go over the all the articles in the data file
for row in range(rows):
    summary = df.ix[row,'headline'] # headline is the column representing the summary
    article = df.ix[row,'text']     # text is the column representing the article

    # A threshold is used to remove short articles with long summaries as well as articles with no summary
    if len(summary) < (0.75 * len(article)) and len(summary) != 0:
        summary = summary.encode('utf-8')
        article = article.encode('utf-8')

        # Store WikiHow titles into text file
        title = df.ix[row,'title']
        title = title.encode('utf-8')
        title_file.write(title + b'\n')

        # A file name is created from the hash of a title
        title_hexdigest = hashhex(title)
        filename = title_hexdigest + '.txt'

        with open(path + '/' + filename,'wb') as f:
            f.write(article)
            f.write(b'\n')
            # Separate article and summary with tag
            f.write(b'@summary' + b'\n')
            f.write(summary)
            f.write(b'\n')

title_file.close()
