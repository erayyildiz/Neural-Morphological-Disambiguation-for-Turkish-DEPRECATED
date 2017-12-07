# coding=utf-8
import random
import re
from keras.layers import Embedding, Add, Activation, Input, Lambda, Reshape, TimeDistributed, Bidirectional, LSTM, K
from keras.models import Model, load_model
from collections import namedtuple
from pprint import pprint
import numpy as np
import time
import cPickle as pickle
from datetime import datetime
import sys
import tensorflow as tf
from keras.utils import to_categorical

from maskedreshape import MaskedReshape


class MorphologicalDisambiguator(object):
    SENTENCE_BEGIN_TAG = "<s>"
    SENTENCE_END_TAG = "</s>"

    WordStruct = namedtuple("WordStruct", ["surface_word", "roots", "tags"])
    analysis_regex = re.compile(r"^([^\\+]*)\+(.+)$", re.UNICODE)
    tag_seperator_regex = re.compile(r"[\\+\\^]", re.UNICODE)

    max_sentence_length = 40
    max_n_analyses = 12
    max_word_root_length = 20
    max_analysis_length = 30
    max_surface_form_length = 50
    char_embedding_dim = 100
    tag_embedding_dim = 100
    char_lstm_dim = 100
    tag_lstm_dim = 100
    sentence_level_lstm_dim = 200
    sentence_level_lstm_dim

    def _build_model(self, sess=None):
        if sess:
            K.set_session(sess)
        sentences_word_root_input = Input(
            shape=(self.max_sentence_length, self.max_n_analyses, self.max_word_root_length),
            dtype='int32',
            name='sentences_word_root_input')

        sentences_analysis_input = Input(
            shape=(self.max_sentence_length, self.max_n_analyses, self.max_analysis_length,),
            dtype='int32',
            name='sentences_analysis_input')

        char_embedding_layer = Embedding(len(self.char2id_root) + 2,
                                         self.char_embedding_dim,
                                         name='char_embedding_layer',
                                         mask_zero=True)

        tag_embedding_layer = Embedding(len(self.tag2id)+ 2,
                                        self.tag_embedding_dim,
                                        name='tag_embedding_layer',
                                        mask_zero=True)

        surface_form_input = Input(shape=(self.max_sentence_length, self.max_surface_form_length,),
                                   dtype='int32',
                                   name='surface_form_input')

        print sentences_word_root_input

        input_char_embeddings, char_lstm_layer_output \
            = MorphologicalDisambiguator.create_two_level_bi_lstm(sentences_word_root_input,
                                           char_embedding_layer, self.max_sentence_length,
                                           self.max_n_analyses, self.max_word_root_length,
                                           self.char_lstm_dim, self.char_embedding_dim, sess=sess)

        input_tag_embeddings, tag_lstm_layer_output = \
            MorphologicalDisambiguator.create_two_level_bi_lstm(sentences_analysis_input,
                                         tag_embedding_layer,
                                         self.max_sentence_length, self.max_n_analyses,
                                         self.max_analysis_length, self.tag_lstm_dim, self.tag_embedding_dim, sess=sess)

        print "char_lstm_layer_output", char_lstm_layer_output

        added_root_and_analysis_embeddings = Add()([char_lstm_layer_output, tag_lstm_layer_output])
        R_matrix = Activation('tanh')(added_root_and_analysis_embeddings)
        # (None, max_sentence_length, max_n_analyses, 2*char_lstm_dim)

        print "R_matrix", R_matrix

        h = MorphologicalDisambiguator\
            .create_context_bi_lstm(surface_form_input, char_embedding_layer,
                               self.max_sentence_length, self.max_surface_form_length,
                               self.char_lstm_dim, self.char_embedding_dim, self.sentence_level_lstm_dim, sess=sess)

        print "h", h

        ll = Lambda(self.dot_product_over_specific_axis,
                    output_shape=self.fabricate_calc_output_shape(self.max_sentence_length, self.max_n_analyses))

        # compute h
        p = Activation('softmax', name="p")(ll([R_matrix, h]))

        print "p", p

        predicted_tags = Lambda(lambda x: K.max(x, axis=2), output_shape=lambda s: s)(p)

        model = Model(inputs=[sentences_word_root_input, sentences_analysis_input, surface_form_input],
                      outputs=[p])

        return model

    @staticmethod
    def dot_product_over_specific_axis(inputs, sess=None):
        if sess:
            K.set_session(sess)
        print "INPUTS TO LAMBDA:", inputs
        x = inputs[0]
        # x [?, 5, 10, 14]
        y = inputs[1]
        # x = tf.transpose(x, [0,1,3,2])
        # x.T [?, 5, 14, 10]
        # y [?, 5, 14]
        y = tf.reshape(y, tf.concat([tf.shape(y), [1]], axis=0))
        # y [?, 5, 14, 1]
        result = tf.matmul(x, y)
        result = tf.squeeze(result, axis=[3])
        return result

    @staticmethod
    def fabricate_calc_output_shape(max_sentence_length, max_n_analyses, sess=None):
        if sess:
            K.set_session(sess)
        def calc_output_shape(input_shape, sess=None):
            if sess:
                K.set_session(sess)
            return tuple([None, max_sentence_length, max_n_analyses])

        return calc_output_shape

    @staticmethod
    def create_two_level_bi_lstm(input_4d, embedding_layer,
                                 max_sentence_length, max_n_analyses, max_word_root_length,
                                 lstm_dim, embedding_dim,
                                 masked=True,
                                 silent=False, sess=None):
        if sess:
            K.set_session(sess)
        r = Reshape((max_sentence_length * max_n_analyses * max_word_root_length,))
        # input_4d = Lambda(lambda x: x, output_shape=lambda s: s)(input_4d)
        rr = r(input_4d)
        input_embeddings = embedding_layer(rr)
        if not silent:
            print input_embeddings
        if masked:
            r = MaskedReshape((max_sentence_length * max_n_analyses, max_word_root_length, embedding_dim),
                              (max_sentence_length * max_n_analyses, max_word_root_length))
        else:
            r = Reshape(
                (max_sentence_length * max_n_analyses, max_word_root_length, embedding_dim))
        # input_embeddings = Lambda(lambda x: x, output_shape=lambda s: s)(input_embeddings)
        rr = r(input_embeddings)
        lstm_layer = Bidirectional(LSTM(lstm_dim,
                                        input_shape=(max_word_root_length, embedding_dim)))
        td_lstm_layer = TimeDistributed(lstm_layer,
                                        input_shape=(max_word_root_length, embedding_dim))

        lstm_layer_output = td_lstm_layer(rr)
        lstm_layer_output_relu = Activation('relu')(lstm_layer_output)
        if not silent:
            print "lstm_layer_output_relu", lstm_layer_output_relu
        r = Reshape((max_sentence_length, max_n_analyses, 2 * lstm_dim))
        lstm_layer_output_relu = Lambda(lambda x: x, output_shape=lambda s: s)(lstm_layer_output_relu)
        lstm_layer_output_relu_reshaped = r(lstm_layer_output_relu)
        if not silent:
            print "lstm_layer_output_relu_reshaped", lstm_layer_output_relu_reshaped
        return input_embeddings, lstm_layer_output_relu_reshaped

    @staticmethod
    def create_context_bi_lstm(input_3d, embedding_layer,
                               max_sentence_length, max_surface_form_length,
                               lstm_dim, embedding_dim,
                               sentence_level_lstm_dim, sess=None):
        if sess:
            K.set_session(sess)
        r = Reshape((max_sentence_length * max_surface_form_length,))
        rr = r(input_3d)
        input_embeddings = embedding_layer(rr)
        print input_embeddings
        # input_embeddings = Lambda(lambda x: x, output_shape=lambda s: s)(input_embeddings)
        r = MaskedReshape((max_sentence_length, max_surface_form_length, embedding_dim),
                          (max_sentence_length, max_surface_form_length))
        rr = r(input_embeddings)
        lstm_layer = Bidirectional(LSTM(lstm_dim,
                                        input_shape=(max_surface_form_length, embedding_dim)))
        td_lstm_layer = TimeDistributed(lstm_layer,
                                        input_shape=(max_surface_form_length, embedding_dim))

        char_bi_lstm_outputs = td_lstm_layer(rr)
        print "char_bi_lstm_outputs", char_bi_lstm_outputs

        sentence_level_lstm_layer = Bidirectional(LSTM(sentence_level_lstm_dim,
                                                       input_shape=(max_sentence_length, 2 * lstm_dim),
                                                       return_sequences=True),
                                                  merge_mode='sum',
                                                  input_shape=(max_sentence_length, sentence_level_lstm_dim))
        # sentence_level_td_lstm_layer = TimeDistributed(sentence_level_lstm_layer,
        #                                      input_shape=(max_sentence_length, 2 * lstm_dim))
        # sentence_level_bi_lstm_outputs = sentence_level_td_lstm_layer(char_bi_lstm_outputs)
        char_bi_lstm_outputs = Lambda(lambda x: x, output_shape=lambda s: s)(char_bi_lstm_outputs)
        sentence_level_bi_lstm_outputs = sentence_level_lstm_layer(char_bi_lstm_outputs)
        sentence_level_bi_lstm_outputs_tanh = Activation('tanh')(sentence_level_bi_lstm_outputs)

        print "sentence_level_bi_lstm_outputs", sentence_level_bi_lstm_outputs
        print "sentence_level_bi_lstm_outputs_tanh", sentence_level_bi_lstm_outputs_tanh

        return sentence_level_bi_lstm_outputs_tanh

    @classmethod
    def _encode(cls, tokens, vocab):
        return [vocab[token] for token in tokens]

    @classmethod
    def _embed(cls, token, char_embedding_table):
        return [char_embedding_table[ch] for ch in token]

    @classmethod
    def _print_namedtuple(cls, nt):
        pprint(dict(nt._asdict()))

    def __init__(self, train_from_scratch=True, char_representation_len=100, word_lstm_rep_len=200,
                 train_data_path="data/data.train.txt", dev_data_path="data/data.dev.txt",
                 test_data_paths=["data/data.test.txt", "data/test.merge", "data/Morph.Dis.Test.Hand.Labeled-20K.txt"], model_file_name=None, char2id=None, tag2id=None):
        assert word_lstm_rep_len % 2 == 0
        if train_from_scratch:
            assert train_data_path
            assert len(test_data_paths) > 0
            self.test_data_paths = test_data_paths
            self.tests = [self.load_data(test_data_path) for test_data_path in test_data_paths]
            if dev_data_path:
                self.dev = self.load_data(dev_data_path)
            else:
                self.dev = None

            print "Loading Vocabulary..."
            self.char2id_surface = char2id
            self.char2id_root = char2id
            self.tag2id = tag2id
            self.surface_word2id = None
            self.root2id = None

            print "Creating keras model"
            self.model = self._build_model()
            self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

            self.iterative_training(train_data_path)
        else:
            print "Loading Pre-Trained Model"
            assert model_file_name
            self.load_model(model_file_name, char_representation_len, word_lstm_rep_len)
            self.iterative_training(train_data_path)

    def load_data(self, file_path, max_sentence=sys.maxint):
        sentence = []
        sentences = []
        with open(file_path, 'r') as f:
            for line in f:
                trimmed_line = line.decode("utf-8").strip(" \r\n\t")
                if trimmed_line.startswith("<S>") or trimmed_line.startswith("<s>"):
                    sentence = []
                elif trimmed_line.startswith("</S>") or trimmed_line.startswith("</s>"):
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        if len(sentences) > max_sentence:
                            return sentences
                elif len(trimmed_line) == 0 or "<DOC>" in trimmed_line or trimmed_line.startswith("</DOC>") or trimmed_line.startswith(
                        "<TITLE>") or trimmed_line.startswith("</TITLE>"):
                    pass
                else:
                    parses = re.split(r"[\t ]", trimmed_line)
                    surface = parses[0]
                    analyzes = parses[1:]
                    roots = [self._get_root_from_analysis(analysis) for analysis in analyzes]
                    tags = [self._get_tags_from_analysis(analysis) for analysis in analyzes]
                    current_word = self.WordStruct(surface, roots, tags)
                    sentence.append(current_word)
        return sentences

    def _get_tags_from_analysis(self, analysis):
        if analysis.startswith("+"):
            return self.tag_seperator_regex.split(analysis[2:])
        else:
            return self.tag_seperator_regex.split(self.analysis_regex.sub(r"\2", analysis))

    def _get_root_from_analysis(self, analysis):
        if analysis.startswith("+"):
            return "+"
        else:
            return self.analysis_regex.sub(r"\1", analysis)

    def sentence_from_str(self, str):
        sentence = []
        lines = str.split("\n")
        for line in lines:
            parses = line.split(" ")
            surface = parses[0]
            analyzes = parses[1:]
            roots = [self._get_root_from_analysis(analysis) for analysis in analyzes]
            tags = [self._get_tags_from_analysis(analysis) for analysis in analyzes]
            current_word = self.WordStruct(surface, roots, tags)
            sentence.append(current_word)
        return sentence

    def predict_indices(self, sentence):
        selected_indices = []
        encoded_surface_words, encoded_roots, encoded_tags, ground_truth = self.encode_batch([sentence])
        predictions = self.model.predict([encoded_roots, encoded_tags, encoded_surface_words])
        for prediction in predictions[0]:
            selected_indices.append(np.argmax(prediction))
            # print "prediction index:{}, score:{}".format(np.argmax(prediction), np.max(prediction))
        return selected_indices

    def predict(self, sentence):
        res = []
        selected_indices = self.predict_indices(sentence)
        for w, i in zip(sentence, selected_indices):
            res.append(w.roots[i] + "+" + "+".join(w.tags[i]))
        return res

    def calculate_acc(self, sentences, labels=None, print_result=False):
        corrects = 0
        non_ambigious_count = 0
        total = 0
        if not labels:
            labels = [[0 for w in sentence] for sentence in sentences]
        encoded_surface_words, encoded_roots, encoded_tags, ground_truth = self.encode_batch(sentences)
        predictions = self.model.predict([encoded_roots, encoded_tags, encoded_surface_words])
        for sentence in sentences:
            non_ambigious_count += [1 for w in sentence if len(w.roots) == 1].count(1)
            total += len(sentence)

        for sentence_label, sentence_prediction in zip(labels, predictions):
            for word_label, word_prediction in zip(sentence_label, sentence_prediction):
                predicted_index = np.argmax(word_prediction)
                if predicted_index == word_label:
                    corrects += 1
        acc = (corrects * 1.0) / total
        amb_acc = (corrects - non_ambigious_count) * 1.0 / (total - non_ambigious_count)
        if print_result:
            print "Loss={}   Accuracy={}   Ambiguous Accuracy={}".format(loss, acc, ambiguous_acc)
        return 0.0, acc, amb_acc

    def encode_batch(self, sentences):
        n = len(sentences)
        encoded_surface_words = np.zeros((n, self.max_sentence_length, self.max_surface_form_length))
        encoded_roots = np.zeros((n, self.max_sentence_length, self.max_n_analyses, self.max_word_root_length))
        encoded_tags = np.zeros((n, self.max_sentence_length, self.max_n_analyses, self.max_analysis_length))
        ground_truth = np.zeros((n, self.max_sentence_length, self.max_n_analyses))

        for i, s in enumerate(sentences):
            if len(s) > self.max_sentence_length:
                continue
            for j, word in enumerate(s):
                assert len(word.roots) == len(word.tags)
                modified_word = word
                if len(modified_word.tags) > self.max_n_analyses:
                    modified_word = self.WordStruct(modified_word.surface_word,
                                                     modified_word.roots[:self.max_n_analyses], modified_word.tags[:self.max_n_analyses])
                if len(modified_word.surface_word) > self.max_surface_form_length:
                    modified_word = self.WordStruct(modified_word.surface_word[:self.max_surface_form_length],
                                                    modified_word.roots, modified_word.tags)
                encoded_surface_word = self._encode(modified_word.surface_word, self.char2id_surface)
                for k, c in enumerate(encoded_surface_word):
                    encoded_surface_words[i, j, k] = c
                for k, root in enumerate(modified_word.roots):
                    if len(root) > self.max_word_root_length:
                        root = root[:self.max_word_root_length]
                    encoded_root = self._encode(root, self.char2id_root)
                    for l, c in enumerate(encoded_root):
                        encoded_roots[i, j, k, l] = c
                for k, tag in enumerate(modified_word.tags):
                    if len(tag) > self.max_analysis_length:
                        tag = tag[-self.max_analysis_length:]
                    encoded_tag = self._encode(tag, self.tag2id)
                    for l, c in enumerate(encoded_tag):
                        encoded_tags[i, j, k, l] = c
                ground_truth[i, j, 0] = 1.0
        return encoded_surface_words, encoded_roots, encoded_tags, ground_truth

    def iterative_training(self, train_data_path, batch_size=5000, notify_size=50000, model_name="model", early_stop=False,
                           num_epoch=5):
        self.model.summary()
        max_acc = 0.0
        epoch_loss = 0
        for epoch in xrange(num_epoch):
            t1 = datetime.now()
            count = 1
            with open(train_data_path, "r") as f:
                sentences = []
                sentence = []
                for line in f:
                    trimmed_line = line.decode("utf-8").strip(" \r\n\t")
                    if trimmed_line.startswith("<S>") or trimmed_line.startswith("<s>"):
                        pass
                    elif trimmed_line.startswith("</S>") or trimmed_line.startswith("</s>"):
                        if count % batch_size == 0 and len(sentences) > 0:
                            encoded_surface_words, encoded_roots, encoded_tags, ground_truth = self.encode_batch(sentences)
                            self.model.fit(
                                [
                                    encoded_roots,
                                    encoded_tags,
                                    encoded_surface_words
                                ],
                                ground_truth,
                                batch_size=1)
                            sentences = []
                        if count % notify_size == 0:
                            t4 = datetime.now()
                            delta = t4 - t1
                            print "Processed {} sentences in {} minutes".format(count, delta.seconds / 60.0)
                            if self.dev:
                                print "Calculating Accuracy on dev set"
                                _, acc, amb_acc = self.calculate_acc(self.dev)
                                print "Dev set --> Accuracy:{}   AmbiguousAccuracy={}" \
                                    .format(acc, amb_acc)
                            print "Calculating Accuracy on test sets"
                            accs = []
                            amb_accs = []
                            for q, test_set in enumerate(self.tests):
                                _, acc, amb_acc = self.calculate_acc(test_set)
                                accs.append(acc)
                                amb_accs.append(amb_acc)
                            for q in range(0, len(self.test_data_paths)):
                                print "Test set: {} --> Accuracy:{}   AmbiguousAccuracy={}"\
                                    .format(self.test_data_paths[q], accs[q], amb_accs[q])
                        if len(sentence) > 0:
                            sentences.append(sentence)
                            sentence = []
                            count += 1
                    elif len(trimmed_line) == 0 or "<DOC>" in trimmed_line or trimmed_line.startswith(
                            "</DOC>") or trimmed_line.startswith(
                        "<TITLE>") or trimmed_line.startswith("</TITLE>"):
                        pass
                    else:
                        parses = re.split(r"[\t ]", trimmed_line)
                        surface = parses[0]
                        analyzes = parses[1:]
                        roots = [self._get_root_from_analysis(analysis) for analysis in analyzes]
                        tags = [self._get_tags_from_analysis(analysis) for analysis in analyzes]
                        current_word = self.WordStruct(surface, roots, tags)
                        sentence.append(current_word)
            t4 = datetime.now()
            delta = t4 - t1
            print "epoch {} finished in {} minutes. loss = {}".format(epoch, delta.seconds / 60.0,
                                                                      epoch_loss / count * 1.0)
            epoch_loss = 0
            _, acc, amb_acc = self.calculate_acc(self.dev)
            print " accuracy on dev set: {} ambiguous accuracy on dev: ".format(acc, amb_acc)
            if acc > max_acc:
                max_acc = acc
                print "Max accuracy increased = {}, saving model...".format(str(max_acc))
                self.save_model(model_name)
            elif early_stop and max_acc - acc > 0.05:
                print "Max accuracy did not incrase, early stopping!"
                break
            print "Calculating Accuracy on test sets"
            accs = []
            amb_accs = []
            for q, test_set in enumerate(self.tests):
                _, acc, amb_acc = self.calculate_acc(test_set)
                accs.append(acc)
                amb_accs.append(amb_acc)
            for q in range(0, len(self.test_data_paths)):
                print "Test set: {} --> Accuracy:{}   AmbiguousAccuracy={}" \
                    .format(self.test_data_paths[q], accs[q], amb_accs[q])

    def save_model(self, model_name):
        self.model.save("models/"+model_name+"-"+time.strftime("%d.%m.%Y")+".model.h5")
        with open("models/" + model_name + "-" + time.strftime("%d.%m.%Y") + ".char2id_root", "w") as f:
            pickle.dump(self.char2id_root, f)
        with open("models/" + model_name + "-" + time.strftime("%d.%m.%Y") + ".char2id_surface", "w") as f:
            pickle.dump(self.char2id_surface, f)
        with open("models/" + model_name + "-" + time.strftime("%d.%m.%Y") + ".tag2id", "w") as f:
            pickle.dump(self.tag2id, f)
        with open("models/" + model_name + "-" + time.strftime("%d.%m.%Y") + ".root2id", "w") as f:
            pickle.dump(self.root2id, f)
        with open("models/" + model_name + "-" + time.strftime("%d.%m.%Y") + ".surface_word2id", "w") as f:
            pickle.dump(self.surface_word2id, f)

    def load_model(self, model_name):
        with open("models/" + model_name + ".char2id_root", "r") as f:
            self.char2id_root = pickle.load(f)
        with open("models/" + model_name + ".char2id_surface", "r") as f:
            self.char2id_surface = pickle.load(f)
        with open("models/" + model_name + ".tag2id", "r") as f:
            self.tag2id = pickle.load(f)
        with open("models/" + model_name + ".root2id", "r") as f:
            self.root2id = pickle.load(f)
        with open("models/" + model_name + ".surface_word2id", "r") as f:
            self.surface_word2id = pickle.load(f)

        self.model = load_model("models/" + model_name + ".model.h5")

    @classmethod
    def create_from_existed_model(cls, model_path):
        return MorphologicalDisambiguator(train_from_scratch=False, model_file_name=model_path)


if __name__ == "__main__":
    char2id = None
    tag2id = None
    with open("defaultdic/char2id", "r") as f:
        char2id = pickle.load(f)
    with open("defaultdic/tag2id", "r") as f:
        tag2id = pickle.load(f)

    disambiguator = MorphologicalDisambiguator(train_from_scratch=True, char_representation_len=100,
                                               word_lstm_rep_len=200,
                                               train_data_path="data/data.train.txt", model_file_name=None,
                                               char2id=char2id, tag2id=tag2id)

    # disambiguator = MorphologicalDisambiguator.create_from_existed_model("model-22.10.2017")
    # print "Loading test data"
    test_sentences = disambiguator.load_data("data/Morph.Dis.Test.Hand.Labeled-20K.txt")
    print "Calculating Accuracy on Morph.Dis.Test.Hand.Labeled-20K.txt"
    print disambiguator.calculate_acc(test_sentences)

    test_sentences = disambiguator.load_data("data/test.merge")
    print "Calculating Accuracy on test.merge"
    print disambiguator.calculate_acc(test_sentences)

    test_sentences = disambiguator.load_data("data/data.test.txt")
    print "Calculating Accuracy on data.test.txt"
    print disambiguator.calculate_acc(test_sentences)
