import json
import csv
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH


class DeepMojiWrapper(object):
    def __init__(self):
        self.maxlen = 30

        self.load_mappings()

        print('Loading model from {}.'.format(PRETRAINED_PATH))
        self.model = deepmoji_emojis(self.maxlen, PRETRAINED_PATH)
        self.model.summary()

    def load_mappings(self):
        self.mapping = {}
        with open('/mnt/drive1/data/eco/models/unicode_mapping.txt') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=';')
            for row in readCSV:
                deepmoji_index = int(row[0])
                unicode_emoji = row[1]
                self.mapping[deepmoji_index] = '&#x' + unicode_emoji

    def top_elements(self, array, k):
        ind = np.argpartition(array, -k)[-k:]
        return ind[np.argsort(array[ind])][::-1]

    def predict(self, sentence):
        sentence_to_analyze = [sentence]

        print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)
        st = SentenceTokenizer(vocabulary, self.maxlen)
        tokenized, _, _ = st.tokenize_sentences(sentence_to_analyze)

        print('Running predictions.')
        prob = self.model.predict(tokenized)

        # Find top emojis for each sentence. Emoji ids (0-63)
        # correspond to the mapping in emoji_overview.png
        # at the root of the DeepMoji repo.
        scores = []
        for i, t in enumerate(sentence_to_analyze):
            t_tokens = tokenized[i]
            t_score = [t]
            t_prob = prob[i]
            ind_top = self.top_elements(t_prob, 5)
            ind_top_unicode = []
            for index in ind_top:
                unicode = self.mapping[index]
                ind_top_unicode.append(unicode)
            print(ind_top)
            t_score.append(sum(t_prob[ind_top]))
            t_score.extend(ind_top)
            t_score.extend([t_prob[ind] for ind in ind_top])
            scores.append(t_score)
            print(t_score)
            return ind_top_unicode

if __name__ == "__main__":
    wrappper = DeepMojiWrapper()
    unicode1 = wrappper.predict('this is pretty fucking nice')
    print(unicode1)
    unicode2 = wrappper.predict('this is pretty fucking shit')
    print(unicode2)