import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BertModel  # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel
from transformers import RobertaModel  # https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaModel
from transformers import DistilBertModel  # https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.DistilBertModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import psycopg2
import re
import string
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
    
    con="...."
)

### Loading data (posts)

posts_info = pd.read_sql(
    """SELECT * FROM public.post_text_df""",
    
    con="...."
)

### Loading data (information about users' actions)

feed_data = pd.read_sql(
    """SELECT * FROM public.feed_data LIMIT 1000000""",
    
    con="...."
)

feed_data = feed_data[feed_data.action=='view']

# preprocessing of text data using pretraining ML-models

def get_model(model_name):
    assert model_name in ['bert', 'roberta', 'distilbert']

    checkpoint_names = {
        'bert': 'bert-base-cased',  # https://huggingface.co/bert-base-cased
        'roberta': 'roberta-base',  # https://huggingface.co/roberta-base
        'distilbert': 'distilbert-base-cased'  # https://huggingface.co/distilbert-base-cased
    }

    model_classes = {
        'bert': BertModel,
        'roberta': RobertaModel,
        'distilbert': DistilBertModel
    }

    return AutoTokenizer.from_pretrained(checkpoint_names[model_name]), model_classes[model_name].from_pretrained(checkpoint_names[model_name])

tokenizer, model = get_model('distilbert')

# Creating numerical datasets for posts
# For implementing model "distilbert" it is necessary to modify data in a sertan format PostDataset
# And after that to assign the batch size

class PostDataset(Dataset):
    def __init__(self, texts, tokenizer):
        super().__init__()

        self.texts = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        return {'input_ids': self.texts['input_ids'][idx], 'attention_mask': self.texts['attention_mask'][idx]}

    def __len__(self):
        return len(self.texts['input_ids'])
    
    
dataset = PostDataset(posts_info['text'].values.tolist(), tokenizer)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

loader = DataLoader(dataset, batch_size=32, collate_fn=data_collator, pin_memory=True, shuffle=False)

b = next(iter(loader))

# Get embeddings

@torch.inference_mode()
def get_embeddings_labels(model, loader):
    model.eval()
    
    total_embeddings = []
    
    for batch in tqdm(loader):
        batch = {key: batch[key].to(device) for key in ['attention_mask', 'input_ids']}

        embeddings = model(**batch)['last_hidden_state'][:, 0, :]

        total_embeddings.append(embeddings.cpu())

    return torch.cat(total_embeddings, dim=0)

device = torch.device('cpu')

model = model.to(device)

embeddings = get_embeddings_labels(model, loader).numpy()

# Clissify embeddings using KMeans

centered = embeddings - embeddings.mean()

pca = PCA(n_components=20)
pca_decomp = pca.fit_transform(centered)


kmeans = KMeans(n_clusters=15, random_state=0).fit(pca_decomp)

posts_info['TextCluster'] = kmeans.labels_

dists_columns = ['DistanceTo1thCluster',
                 'DistanceTo2thCluster',
                 'DistanceTo3thCluster',
                 'DistanceTo4thCluster',
                 'DistanceTo5thCluster',
                 'DistanceTo6thCluster',
                 'DistanceTo7thCluster',
                 'DistanceTo8thCluster',
                 'DistanceTo9thCluster',
                 'DistanceTo10thCluster',
                 'DistanceTo11thCluster',
                 'DistanceTo12thCluster',
                 'DistanceTo13thCluster',
                 'DistanceTo14thCluster',
                 'DistanceTo15thCluster']

dists_df = pd.DataFrame(
    data=kmeans.transform(pca_decomp),
    columns=dists_columns
)

#Prepearing all features for model training

posts_info = pd.concat((posts_info, dists_df), axis=1)

df = pd.merge(feed_data,
              posts_info,
              on='post_id',
              how='left')

df = pd.merge(df,
              user_info,
              on='user_id',
              how='left')

df = df.drop([
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

pipe_rf=Pipeline([("column_transformer",
                     col_transform),
                     
                    ("random_forest", 
                     RandomForestClassifier(max_depth=2, random_state=0))])

pipe_rf.fit(X_train, y_train)

print(f"Качество на трейне: {roc_auc_score(y_train, pipe_rf.predict_proba(X_train)[:, 1])}")
print(f"Качество на тесте: {roc_auc_score(y_test, pipe_rf.predict_proba(X_test)[:, 1])}")