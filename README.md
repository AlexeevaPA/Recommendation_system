# Recommendation_system

![Happy user](https://user-images.githubusercontent.com/104028421/235664571-bafcc747-b495-4fc3-bb70-dddcfb4013b4.png)

**Overview**

The task was to build two recommendation systems for users, one of them is based on machine learning and the other one includes deep learning algorithms. 
The quality of models was estimated based on hitrate@5 metric.

**Data**

The SQL database from Karpov courses. It contains information about signed-in users and the posts which they estimated (liked, didn't like).
There are three tables: users, posts and feed. The table "users" contains information about users' id, gender, age, country, city, operation system of their gadget.
The table "posts" describes a text of a certain post and its topic. And the last table named "feed" contains information about time of an action
and user's preferences (like or not) about posts.

**Processing data**

Data didn't contain null falues or correlation features, so the main idea was to process the text and categorial features to get the lowest hitrate@5 metric.

**Data processing without deep learning methods**

There are a few methods to modify a text for using it in models. In this case, I used TF-IDF algorithm. In this algorithm each word is represented 
as a frequency of its appearing in the text and document. 
After implementing of the algorithm I extracted max min and medium meanings of calculated numerical features to decrease the amount of data.
For other categorial features, two methods were used: OneHot Encoding and MeanTargetEncoder. 
The first one was used if the number of unique features was less than 5, otherwise, the second method.

![TF-IDF method](https://user-images.githubusercontent.com/104028421/235689167-c47c1c53-1836-40fd-968a-24adec10eccf.png)

**Data processing with deep learning algorithms**

In this case, I implemented a pretrained model from the huggingface database named "DistilBertModel". The DistilBERT model was proposed in the blog post smaller, 
faster, cheaper, lighter: Introducing DistilBERT, 
a distilled version of BERT, and the paper DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. 
DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base.
It has 40% less parameters than bert-base-uncased, 
runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark. 
After getting word's embeddings, it is necessary to decrease the amount of data.
I decided to divide data into 15 classes using the K-means algorithm, and after that just 15 features were used, which described text data.
For other categorial features, two methods were used: OneHot Encoding and MeanTargetEncoder. 
The first one was used if the number of unique features was less than 5, otherwise, the second method.

**Models**

Both the first and the second models were trained based on a random forest algorithm. The final ROC AUC score for the first model was quite close to the second one:

| Model |Train | Test |
| ------ |----- | ------ | 
| model_1 | 0.62  | 0.61|
| model_2 | 0.61  | 0.60|

**Application**

This project was built as a backend of a web application. Input data are time, user's id and amount of posts, the required method is get. The output of the application 
is a JSON file with a list of recommendation posts. Moreover, in this project, there is a simulation of an A/B-test, whole users were divided into two groups, and together
with the recommendation returned a group of an A/B-test.

       
