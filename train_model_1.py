import pandas as pd
import psycopg2
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from category_encoders.one_hot import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

### Loading data (users)

user_info = pd.read_sql(
    """SELECT * FROM public.user_data""",
    
    con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
)

### Loading data (posts)

posts_info = pd.read_sql(
    """SELECT * FROM public.post_text_df""",
    
    con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
)

### Loading data (information about users' actions)

feed_data = pd.read_sql(
    """SELECT * FROM public.feed_data LIMIT 1000000""",
    
    con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
)

feed_data = feed_data[feed_data.action=='view']

wnl = WordNetLemmatizer()
nltk.download('omw-1.4')

# preprocessing of text data

def preprocessing(line, token=wnl):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    line = line.replace('\n\n', ' ').replace('\n', ' ')
    line = ' '.join([token.lemmatize(x) for x in line.split(' ')])
    return line

# implementation of a TF-IDF algorithm to text in posts

tfidf = TfidfVectorizer(
    stop_words='english',
    preprocessor=preprocessing
)

tfidf_data = (
    tfidf
    .fit_transform(posts_info['text'])
    .toarray()
)

tfidf_data = pd.DataFrame(
    tfidf_data,
    index=posts_info.post_id,
    columns=tfidf.get_feature_names_out()
)

# Extract parameters from numerical features of posts_info['text']

posts_info['TotalTfIdf'] = tfidf_data.sum(axis=1).reset_index()[0]
posts_info['MaxTfIdf'] = tfidf_data.max(axis=1).reset_index()[0]
posts_info['MeanTfIdf'] = tfidf_data.mean(axis=1).reset_index()[0]

df = pd.merge(feed_data,
              posts_info,
              on='post_id',
              how='left')

df = pd.merge(df,
              user_info,
              on='user_id',
              how='left')

df = df.drop([
#    'timestamp',  ### timestamp пока оставим
    'action',
    'text',
],
    axis=1)

df = df.set_index(['user_id', 'post_id'])

# converting data format

df['hour'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.hour)
df['month'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.month)
df['day'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.day)

# splitting data for validation of a future model

df_train = df[df.timestamp < '2021-12-15']
df_test = df[df.timestamp >= '2021-12-15']

df_train = df_train.drop('timestamp', axis=1)
df_test = df_test.drop('timestamp', axis=1)

X_train = df_train.drop('target', axis=1)
X_test = df_test.drop('target', axis=1)

y_train = df_train['target']
y_test = df_test['target']


object_cols = [
    'topic', 'gender', 'country',
    'city', 'exp_group',
    'os', 'source'
]

# implementing of OneHotEncoder and MeanTargetEncoder for categorial features

cols_for_ohe = [x for x in object_cols if X_train[x].nunique() < 5]
cols_for_mte = [x for x in object_cols if X_train[x].nunique() >= 5]


cols_for_ohe_idx = [list(X_train.columns).index(col) for col in cols_for_ohe]
cols_for_mte_idx = [list(X_train.columns).index(col) for col in cols_for_mte]

t = [
    ('OneHotEncoder', OneHotEncoder(), cols_for_ohe_idx),
    ('MeanTargetEncoder', TargetEncoder(), cols_for_mte_idx)
]

# for distinguishing categorial and numerical features

col_transform = ColumnTransformer(transformers=t)

pipe_rf = Pipeline([("column_transformer",
                     col_transform),
                     
                    ("random_forest", 
                     RandomForestClassifier(max_depth=2, random_state=0))])

pipe_rf.fit(X_train, y_train)

print(f"Quality of train data: {roc_auc_score(y_train, pipe_rf.predict_proba(X_train)[:, 1])}")
print(f"Quality of test data: {roc_auc_score(y_test, pipe_rf.predict_proba(X_test)[:, 1])}")
