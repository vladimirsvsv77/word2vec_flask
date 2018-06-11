__author__ = 'VladimirSveshnikov'
import json
import math

import nltk

import gensim
from scipy.spatial.distance import euclidean
from nltk.corpus import stopwords

from flask import Flask, request, jsonify
import indicoio


app = Flask(__name__)


indicoio.config.api_key = 'your_api_key'


print('start app')
# model link: https://fasttext.cc/docs/en/english-vectors.html
model_path = 'your_path'
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')
print('model loaded')
stop = set(stopwords.words('russian'))


MAX_DISTANCE = 1.2


def get_similarity_euql(model, first_sentence, second_sentence):
    # remove stop words
    first_sentence = [i for i in nltk.word_tokenize(first_sentence.lower()) if i not in stop]
    second_sentence = [i for i in nltk.word_tokenize(second_sentence.lower()) if i not in stop]

    # convert to vector
    first_vectors = [model[i] for i in first_sentence if i in model]
    second_vectors = [model[i] for i in second_sentence if i in model]

    if len(first_vectors) == 0 or len(second_vectors) == 0:
        return MAX_DISTANCE

    # calculate distance
    similarity = 0
    for first_vector in first_vectors:
        sim_i = 0
        for second_vector in second_vectors:
            sim_i += euclidean(first_vector, second_vector)
        similarity += sim_i / len(second_vectors)
    return similarity / len(first_vectors)


def get_category_w2v(question, questions, w2v):
    dist = {}
    for category in questions:
        distance = 0
        for j in questions[category]['questions']:
            distance += get_similarity_euql(w2v, question, j)
        dist[category] = distance / len(questions[category]['questions'])
    dist = sorted(dist.items(), key=lambda value: value[1])
    return [dist[0][0], dist[0][1]]


def get_category(question, questions, w2v):

    cats_info = get_category_w2v(question, questions, w2v)
    answer = cats_info[0]
    distance = str(cats_info[1])

    if math.isnan(float(distance)):
        answer = 'no'
    return answer


@app.route("/get_dist", methods=["POST"])
def get_dist():
    text = request.form['text']
    dialog = json.loads(request.form['dialog'])
    responce = get_category(str(text), dialog, model)

    return jsonify(responce)


@app.route("/get_sentiment", methods=["GET"])
def get_sentiment():
    text = request.args.get('text')
    responce = indicoio.sentiment(text, language='ru')
    return str(responce)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=False)
