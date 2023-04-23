import os
from datetime import datetime
import hashlib
from typing import List, Tuple

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI
from loguru import logger
from schema import PostGet, Response
from sqlalchemy import create_engine


app=FastAPI()

def batch_load_sql(query: str):
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn=engine.connect().execution_options(
        stream_results=True)
    chunks=[]
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
        logger.info(f"Got chunk: {len(chunk_dataframe)}")
    conn.close()
    return pd.concat(chunks, ignore_index=True)
def get_model_path(path: str) -> str:
    MODEL_PATH=path
    return MODEL_PATH

def load_models(model_version: str):
    model_path = get_model_path(model_version)
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model


def load_features_test():

    #logger.info("loading liked posts")
    print("loading liked posts")
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        where action='like'"""
    liked_posts=batch_load_sql(liked_posts_query)

    posts_features=pd.read_sql(
        """SELECT * FROM public.posts_info_features_t""",
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )

    user_features=pd.read_sql(
        """SELECT * FROM public.user_data""",
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml"
    )
    return [liked_posts, posts_features, user_features]

def load_features_control():

    #logger.info("loading liked posts")
    print("loading liked posts")
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        where action='like'"""
    liked_posts=batch_load_sql(liked_posts_query)

    posts_features=pd.read_sql(
        """SELECT * FROM public.posts_info_features_t""",
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )

    user_features=pd.read_sql(
        """SELECT * FROM public.user_data""",
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml"
    )
    return [liked_posts, posts_features, user_features]

model_control = load_models("control")
model_test = load_models("test")

features_control = load_features_control()
features_test = load_features_test()


SALT = "my_salt"


def get_user_group(id: int) -> str:
    value_str = str(id) + SALT
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    percent = value_num % 100
    if percent < 50:
        return "control"
    elif percent < 100:
        return "test"
    return "unknown"

def calculate_features_control(
    id: int, time: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Loading users' features for control model

    user_features = features_control[2].loc[features_control[2].user_id == id]
    user_features = user_features.drop('user_id', axis=1)

    posts_features = features_control[1].drop(['index', 'text'], axis=1)
    content = features_control[1][['post_id', 'text', 'topic']]

    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month
    user_posts_features['day'] = time.day

    return user_features, user_posts_features


def calculate_features_test(
        id: int, time: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Loading users' features for test model

    user_features = features_test[2].loc[features_test[2].user_id == id]
    user_features = user_features.drop('user_id', axis=1)

    posts_features = features_test[1].drop(['index', 'text'], axis=1)
    content = features_test[1][['post_id', 'text', 'topic']]

    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month
    user_posts_features['day'] = time.day

    return user_features, user_posts_features


def get_recommended_feed(id: int, time: datetime, limit: int) -> Response:
    
    # Choose the model for A/B-testing
    user_group = get_user_group(id=id)

    if user_group == "control":
        model = model_control
        user_features, user_posts_features = calculate_features_control(
            id=id, time=time
        )
        liked_posts = features_control[0]
        content = features_control[1][['post_id', 'text', 'topic']]
    elif user_group == "test":
        model = model_test
        user_features, user_posts_features = calculate_features_test(
            id=id, time=time
        )
        liked_posts = features_test[0]
        content = features_test[1][['post_id', 'text', 'topic']]
    else:
        raise ValueError("unknown group")


    #Predicting
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features["predicts"] = predicts

    #deleting liked posts

    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    
    recommended_posts=filtered_.sort_values('predicts')[-limit:].index

    return Response(
        recommendations=[
            PostGet(id=i,
                    text=content[content.post_id == i].text.values[0],
                    topic=content[content.post_id == i].topic.values[0])
            for i in recommended_posts
        ],
        exp_group=user_group,
    )




@app.get("/post/recommendations/",response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int=10) -> Response:
    return get_recommended_feed(id, time, limit)
