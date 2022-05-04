import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

DIALOGS_MERGED_DATA_PATH = "D:/pythonchik/progapy/proj/merged_data/dialogs_data_all.csv"
DIALOGS_META_MERGED_DATA_PATH = "D:/pythonchik/progapy/proj/merged_data/dialogs_users_all.csv"

df = pd.read_csv(DIALOGS_MERGED_DATA_PATH, low_memory=False)
df_meta = pd.read_csv(DIALOGS_META_MERGED_DATA_PATH, low_memory=False)

my_id = '463034736'

# most popular word

my_df_copy = df.copy()
my_df_copy = my_df_copy[my_df_copy['from_id'] == 'PeerUser(user_id=' + str(my_id) + ')']
my_df = pd.DataFrame()

my_df['message'] = my_df_copy['message'].str.lower().str.split('\W+')

my_list = []
for index, row in my_df.iterrows():
    if isinstance(row['message'], list):
        my_list += row['message']
    else:
        my_list.append(row['message'])
my_list = [x for x in my_list if len(str(x)) > 3]

keys = Counter(my_list).keys()
values = Counter(my_list).values()
quantities = {'word': keys, 'value': values}
quan = pd.DataFrame(quantities)
print(quan.sort_values(by='value', ascending=False).head(10))

plt = quan.sort_values(by='value', ascending=False).head(10).plot.bar(
    x='word', rot=0, figsize=(8, 6), color='orange', title='My most popular words:')
plt.set_xlabel("words")
plt.set_ylabel("occurrence in my messages")
plt.show()

# wordcloud

my_df_copy = df.copy()
my_df_copy = my_df_copy[my_df_copy['from_id'] == 'PeerUser(user_id=' + str(my_id) + ')']
list_of_messages = list(my_df_copy['message'])
list_of_messages = [str(x) for x in list_of_messages]

cleanedList = [x for x in list_of_messages if str(x) != 'nan']
text = " ".join(cleanedList).lower()

stop_words = frozenset(['да', 'и', 'в', 'на', 'у', 'а', 'я', 'с', 'не'])

wordcloud = WordCloud(width=800, height=400, stopwords=stop_words, background_color="white").generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

plt.figure(figsize=(20, 10))

plt.show()

# wordcloud of longer words

plt.clf()

text = re.sub(r'\b\w{1,6}\b', '', text)
# text = " ".join(cleanedList).lower()

wordcloud2 = WordCloud(width=800, height=400, stopwords=stop_words, background_color="white").generate(text)

plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")

plt.figure(figsize=(20, 10))

plt.show()

# sentiment analysis

tokenizer = RegexTokenizer()
model = FastTextSocialNetworkModel(tokenizer=tokenizer)

my_df_copy = df.copy()
my_df_copy = my_df_copy[my_df_copy['from_id'] == 'PeerUser(user_id=' + str(my_id) + ')']

list_of_messages = [str(x) for x in list_of_messages]

results = model.predict(list_of_messages, k=2)

for message, sentiment in zip(list_of_messages[3418:3425], results[3418:3425]):
    print(message, '-&gt;', sentiment)

df_with_results = pd.DataFrame(list(zip(list_of_messages, results)), columns=['message', 'result'])
# df_with_results['date'] = my_df_copy['date'][3410:3430]
print(df_with_results[3410:3430])

# df with positive negative etc

sentiment_df = pd.DataFrame(columns=['message', 'positive', 'neutral', 'negative', 'skip', 'speech'])
sentiment_df = sentiment_df.append(results)
sentiment_df['message'] = list_of_messages
sentiment_df = sentiment_df.fillna(0.0)
print(sentiment_df[3410:3430])

positive = sentiment_df['positive'].sum()
negative = sentiment_df['negative'].sum()
neutral = sentiment_df['neutral'].sum()
skip = sentiment_df['skip'].sum()
speech = sentiment_df['speech'].sum()
print('Positive:', positive)
print('Negative:', negative)
print('Neutral:', neutral)
print('Skip:', skip)
print('Speech:', speech)

# pie chart of my msgs

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('equal')
labels = ['positive', 'neutral', 'negative', 'skip', 'speech']
values = [positive, neutral, negative, skip, speech]
ax.pie(values, labels=labels, autopct='%1.2f%%', radius=2,
       colors=['#a1c9fd', '#fec34b', '#fe8c46', '#d5465c', '#fe6053'])
plt.show()

# highlighting areas on plot with naum

naum_dialog_id = 428330492
naum = df.copy()
naum = naum[naum['dialog_id'] == naum_dialog_id]

min_date = '2021-12-01 00:00:00+00:00'
max_date = '2022-02-23 00:00:00+00:00'
naum = naum[(naum['date'] >= min_date) & (naum['date'] <= max_date)]
naum['date'] = pd.to_datetime(naum["date"]).dt.date

# weekdays

s = pd.date_range('2016-12-31', '2017-01-08', freq='D').to_series()
# s.dt.dayofweek
# weekdays = naum.copy()
# weekdays = weekdays['date']
# weekdays.head()

# week = {'date' : naum['date']}
# weekdays = pd.DataFrame(week)

# weekdays['date'] = pd.date_range('2021-12-01', '2022-02-23', freq='D').to_series()
# weekdays['day'] = weekdays['date'].dt.dayofweek
# weekdays.head(10)

# df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

d = '2015-01-08 22:44:09'
date = pd.to_datetime(d).date()

# monday = 0, sunday = 6

ax = naum.groupby('date').size().plot.bar(color='orange', figsize=(14, 8), title='How many messages we wrote a day:')
ax.set_ylabel("number of messages")
# p = plt.axvspan('2021-12-21', '2021-12-25', facecolor='g', alpha=1,zorder=3)

# merging dataframes

df_copy = df.copy()
df_meta_copy = df_meta.copy()

df_meta_copy['clear_id'] = ''
ind = 0
clear_list = []

for index, row in df_meta_copy.iterrows():
    needed_id = ''
    dig = False
    for i in str(row['users']):
        if i.isdigit():
            dig = True
            needed_id += str(i)
        if (not i.isdigit()) & dig == True:
            break
    clear_list.append(needed_id)
    ind += 1

df_meta_copy['clear_id'] = clear_list

df_copy['clear_id'] = ''
ind = 0
clear_list_for_big_df = []
for index, row in df_copy.head(30).iterrows():
    needed_str = str(row['from_id'])
    if needed_str != '':
        for i in clear_list:
            if needed_str.find(str(i)) != -1:
                clear_list_for_big_df.append(i)
    else:
        clear_list_for_big_df.append('')

merged_df = df.merge(df_meta, on="dialog_id")
merged_df.head(10)
