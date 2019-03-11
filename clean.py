import pandas as pd

def decouple_sentences(string):
    l = []
    i = 0
    end_punc = {'.', '?', '!'}
    while i < len(string):
        c = string[i]
        prev_c = None if i == 0 else string[i-1]
        if c.isalpha() and c.isupper() and prev_c in end_punc:
            l.append(' ')
        l.append(c)
        i += 1
    return ''.join(l)

def reformat(string):
    string = ' '.join(string.split())
    string = string.strip(' ;')
    string = decouple_sentences(string)
    return string if any(c.isalpha() for c in string) else ''

# Clean the WikiHowSep.csv dataset and output a cleaned copy
df = pd.read_csv(r'wikihowSep.csv')
df.fillna('', inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
df = df.astype(str)
df = df.applymap(reformat)
df = df[df['title'] != '']
df.to_csv(r'wikihowSep_cleaned.csv', index=False, encoding='utf-8')

def concat(df):
    return pd.Series({
        'headline': '\n'.join(filter(lambda s: s != '', df['headline'])),
        'text': '\n'.join(filter(lambda s: s != '', df['text']))})

# Use the cleaned df and create a cleaned WikiHowAll.csv
df.drop(columns=['overview', 'sectionLabel'], inplace=True)
df = df.groupby('title').apply(concat).reset_index()
df.to_csv(r'wikihowAll_cleaned.csv', index=False, encoding='utf-8')
