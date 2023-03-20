#!/usr/bin/env python
# coding: utf-8



import os
os.listdir("../input/nlp-dataset")

# 파일의 디렉토리를 확인한다. (압축해제 후 다시 업로드한 데이터셋)




import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc

sns.set_style('whitegrid')

import warnings 
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')




plt.rcParams["axes.unicode_minus"] = False
fontpath = "../input/koreanfont/a3.ttf"
fontprop = font_manager.FontProperties(fname = fontpath)




df_train = pd.read_csv("../input/nlp-dataset/labeledTrainData.tsv",
                       header = 0, delimiter = "\t", quoting = 3)

df_test = pd.read_csv("../input/nlp-dataset/testData.tsv",
                      header = 0, delimiter = "\t", quoting = 3)

df_train.shape

# header = 0 : 파일의 첫 번째 줄에 열 이름이 있음을 나타낸다.
# delimiter = "\t" : \t는 필드가 tab으로 구분되는 것을 의미한다.
# quoting = 3 : 3은 텍스트의 쌍따옴표를 무시하도록 한다.




df_train.head()




df_test.shape




df_test.head()




print("Train Columns: ")  
print(df_train.columns.values)
print("----------------------------")
print("Test Columns: ")  
print(df_test.columns.values)

# Test 데이터 셋에 없는 Sentiment를 예측한다.




df_train.info()

# null value는 없다.




df_train.describe()

# sentiment의 통계값 확인




df_train["sentiment"].value_counts()

# sentiment의 클래스가 딱 절반으로 되어있음을 알 수 있다. (부정, 긍정)




df_train["review"][0][:700]

# review 컬럼을 700자 까지만 확인해본다.




get_ipython().system('pip install BeautifulSoup4')




from bs4 import BeautifulSoup

exam1 = BeautifulSoup(df_train["review"][0], "html5lib")
print(df_train["review"][0][:700])
exam1.get_text()[:700]

# BeautifulSoup을 불러와서 review를 확인한다. 
# 그냥 print 한것과 exam으로 불러온 텍스트를 비교해보면 
# <br \>과 같은 html 태그들이 사라진 것을 볼 수있다.




import re

letters_only = re.sub("[^a-zA-Z]", " ", exam1.get_text())
letters_only[:700]

# re를 불러와서 정규표현식으로 특수문자를 제거한다.
# 소문자와 대문자가 아닌 것은 공백으로 대체한다 (re.sub("바꿔야할것", "바꾸고싶은것"))
# output을 보면 특수문자들이 전부 공백으로 대체된 것을 볼 수 있다.




lower_case = letters_only.lower()

words = lower_case.split()
print(len(words))
words[:10]

# letters_only를 전부 소문자로 대체해준다.
# split을 사용하여 단어단위로 나눈다. (토큰화)
# 437개의 토큰으로 이루어져 있다.




import nltk
from nltk.corpus import stopwords
stopwords.words("english")[:10]

# NLTK를 불러오고, stopwords까지 불러와서 확인해본다. 




words = [w for w in words if not w in stopwords.words("english")]
print(len(words))
words[:10]

# words에 담겨져 있던 단어에 Stopwords가 있다면 제거한다.
# 제거한 토큰들을 확인한다.
# 토큰이 437개에서 219개로 줄어들었음을 알 수 있다.




# 포터 스태머의 사용 예시

stemmer = nltk.stem.PorterStemmer()
print(stemmer.stem("maximum"))
print("The stemmed form of running is : {}".format(stemmer.stem("running")))
print("The stemmed form of runs is: {}".format(stemmer.stem("runs")))
print("Tje stemmed form of run is: {}".format(stemmer.stem("run")))

# maximum이 그대로 출력된다. 
# run의 변형어들은 run으로 어간이 추출된다.




# 랭커스터 스태머의 사용 예시

from nltk.stem.lancaster import LancasterStemmer

lanc_stemmer = LancasterStemmer()
print(lanc_stemmer.stem("maximum"))
print("The stemmed form of running is : {}".format(lanc_stemmer.stem("running")))
print("The stemmed form of runs is: {}".format(lanc_stemmer.stem("runs")))
print("Tje stemmed form of run is: {}".format(lanc_stemmer.stem("run")))

# maximum의 어간이 maxim으로 추출된다.
# run의 변형어들은 마찬가지로 run으로 어간이 추출된다.




words[:10]

# 처리하기 전 단어들을 확인해본다.
# going, started 등 변형된 단어가 있음을 알 수 있다.




from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
words = [stemmer.stem(w) for w in words]

words[:10]

# 튜토리얼에서는 스노우볼 스태머를 사용해서 words의 어간을 추출해본다.
# going, started등 어간이 잘 추출된 것을 확인할 수 있다. 




from nltk.stem import WordNetLemmatizer
wordnet_lem = WordNetLemmatizer()

print(wordnet_lem.lemmatize("fly"))
print(wordnet_lem.lemmatize("flies"))

words = [wordnet_lem.lemmatize(w) for w in words]

words[:10]

# lemmatizer를 사용하여 fly, flies를 처리하면 둘 다 fly로 바뀐다.
# words를 lemmatization 처리한 후 결과를 확인해본다.
# stemming한 결과와 마찬가지로 출력된다.




# 위에서 배운 내용을 바탕을 전체적으로 수행할 수 있도록 함수를 만들어 준다.

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    return(" ".join(stemming_words))

# 0. def로 함수 선언 
# 1. HTML 제거
# 2. 영문자가 아닌 문자는 공백으로 변환
# 3. 소문자로 전체 변환
# 4. 파이썬에서는 리스트보다 세트로 찾는게 훨씬 빠르다. stopwods를 세트로 변환
# 5. Stopwords 불용어 제거
# 6. Stemming으로 어간추출
# 7. 공백으로 구분된 문자열로 결합하여 결과 반환




clean_review = review_to_words(df_train["review"][0])
clean_review

# review 데이터의 첫번째 데이터를 함수에 넣고 실행해본다.
# review 문장들이 토큰화되어 깔끔하게 처리되었음을 확인한다.




num_reviews = df_train["review"].size
num_reviews




# 적용시간이 오래걸리는 문제로 인해 multiprocessing을 사용하여 함수를 적용시켜준다.
# multiprocessing을 사용하면 복잡하고 오래걸리는 작업을 별도의 프로세스를 생성 후
# 병렬처리해서 보다 빠른 응답처리 속도를 기대할 수 있는 장점이 있다.
# 출처: https://gist.github.com/yong27/7869662

from multiprocessing import Pool
import numpy as np

def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    # 키워드 항목 중 workers 파라미터를 꺼냄
    workers = kwargs.pop("workers")
    # 위에서 가져온 workers 수로 프로세스 풀을 정의
    pool = Pool(processes = workers)
    # 실행할 함수와 데이터프레임을 워커의 수 만큼 나눠서 작업
    result = pool.map(_apply_df, [(d, func, kwargs)
                                 for d in np.array_split(df, workers)])
    pool.close()
    #작업 결과를 합쳐서 반환
    return pd.concat(list(result))




get_ipython().run_line_magic('time', 'clean_train_reviews = apply_by_multiprocessing(df_train["review"],review_to_words, workers = 4)')




get_ipython().run_line_magic('time', 'clean_test_reviews = apply_by_multiprocessing(df_test["review"],review_to_words, workers = 4)')




from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def displayWordCloud(data = None, backgroundcolor = "black", width = 800, height = 600):
    wordcloud = WordCloud(stopwords = STOPWORDS,
                         background_color = backgroundcolor,
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()




displayWordCloud(" ".join(clean_train_reviews))




df_train["num_words"] = clean_train_reviews.apply(lambda x : len(str(x).split()))
df_train["num_uniq_words"] = clean_train_reviews.apply(lambda x: len(set(str(x).split())))

# 단어 개수 컬럼 생성
# 중복을 제거한 unique 단어 개수 컬럼 생성




x = clean_train_reviews[0]
x = str(x).split()
print(len(x))
x[:10]

# 첫 번째 리뷰의 단어를 세어보면 219개이다. 




import seaborn as sns

fig, ax = plt.subplots(ncols = 2, figsize = (18, 6))
print("리뷰별 단어 평균 값: ", df_train["num_words"].mean())
print("리뷰별 단어 중간 값: ", df_train["num_words"].median())
sns.distplot(df_train["num_words"], bins = 100, ax = ax[0])
ax[0].axvline(df_train["num_words"].median(), linestyle = "dashed")
ax[0].set_title("리뷰별 단어 수 분포", fontproperties = fontprop)

print("리뷰별 고유 단어 평균 값: ", df_train["num_uniq_words"].mean())
print("리뷰별 고유 단어 중간 값: ", df_train["num_uniq_words"].median())
sns.distplot(df_train["num_uniq_words"], bins = 100, color = "g", ax = ax[1])
ax[1].axvline(df_train["num_uniq_words"].median(), linestyle = "dashed")
ax[1].set_title("리뷰별 고유 단어 수 분포", fontproperties = fontprop)




from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# 튜토리얼과 다르게 파라미터 값을 수정
# 파라미터 값만 수정해도 리더보드 스코어 차이가 큼 
vectorizer = CountVectorizer(analyzer = "word", 
                            tokenizer = None,
                            preprocessor = None,
                            stop_words = None,
                            min_df = 2, # 토큰이 나타날 최소 문서 개수
                            ngram_range = (1, 3), # 유니그램, 바이그램 등 
                            max_features = 20000 # 최대 피쳐의 개수
                            )
vectorizer




# 속도 개선을 위해 파이프라인을 사용하도록 개선

pipeline = Pipeline([
    ("vect", vectorizer),
])




get_ipython().run_line_magic('time', 'train_data_features = pipeline.fit_transform(clean_train_reviews)')

train_data_features




train_data_features.shape

# 25000의 관측치와 위에서 지정해주었던 20000개의 feature로 이루어져 있음.




vocab = vectorizer.get_feature_names()
print(len(vocab))
vocab[:10]

# feature의 이름 (단어)를 확인




dist = np.sum(train_data_features, axis = 0)

for tag, count in zip(vocab, dist):
    print(count, tag)
    
pd.DataFrame(dist, columns = vocab)

# 단어를 count해줘서 한번에 확인 




pd.DataFrame(train_data_features[:10].toarray(), columns = vocab).head()

# 각각의 row가 어떤 단어를 포함하고 있는지 확인하기 위함 




from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs =1)
model




get_ipython().run_line_magic('time', 'model = model.fit(train_data_features, df_train["sentiment"])')




from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('time', 'score = np.mean(cross_val_score(                                     model, train_data_features,                                     df_train["sentiment"], cv = 10,                                     scoring = "roc_auc"))')




clean_test_reviews[0]




get_ipython().run_line_magic('time', 'test_data_features = pipeline.transform(clean_test_reviews)')
test_data_features = test_data_features.toarray()

# test 데이터도 똑같이 파이프라인을 사용하여 벡터화 시켜준다.




test_data_features




# 벡터화하며 만든 사전에서 해당 단어가 무엇인지 찾아볼 수 있다.
# vocab = vectorizer.get_feature_names()
vocab[8], vocab[2558], vocab[2559], vocab[2560]




y_pred = model.predict(test_data_features)
y_pred[:10]




sub = pd.DataFrame(data = {"id":df_test["id"], "sentiment" : y_pred})
sub.head()




sub.to_csv("./tutorial_1_LB{:.5f}.csv".format(score), index = False, quoting = 3)




sub_sent = sub["sentiment"].value_counts()
print(sub_sent[0] - sub_sent[1])
sub_sent

# submission의 부정과 긍정의 차이를 살펴본다.






