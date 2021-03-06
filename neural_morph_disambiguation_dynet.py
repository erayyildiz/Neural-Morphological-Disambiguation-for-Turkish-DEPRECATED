# coding=utf-8
import random
import re
import math
import sys
from collections import defaultdict
from collections import namedtuple
from pprint import pprint

import dynet as dy
import numpy as np
import time
import cPickle as pickle
from datetime import datetime

class MorphologicalDisambiguator(object):
    SENTENCE_BEGIN_TAG = "<s>"
    SENTENCE_END_TAG = "</s>"

    WordStruct = namedtuple("WordStruct", ["surface_word", "roots", "tags"])
    analysis_regex = re.compile(r"^([^\+]*)\+(.+)$", re.UNICODE)
    tag_seperator_regex = re.compile(r"[\+\^]", re.UNICODE)

    @classmethod
    def _create_vocab_words(cls, sentences):
        surface_word2id = defaultdict(int)
        surface_word2id[cls.SENTENCE_BEGIN_TAG] = len(surface_word2id) + 1
        surface_word2id[cls.SENTENCE_END_TAG] = len(surface_word2id) + 1
        root2id = defaultdict(int)
        tag2id = defaultdict(int)
        for sentence in sentences:
            for word in sentence:
                if word.surface_word not in surface_word2id:
                    surface_word2id[word.surface_word] = len(surface_word2id) + 1
                for root in word.roots:
                    if root not in root2id:
                        root2id[root] += len(root2id) + 1
                for tags in word.tags:
                    for tag in tags:
                        if tag not in tag2id:
                            tag2id[tag] += len(tag2id) + 1
        return surface_word2id, root2id, tag2id

    @classmethod
    def _create_vocab_chars(cls, sentences):
        char2id_surface = defaultdict(int)
        char2id_surface["<"] = len(char2id_surface) + 1
        char2id_surface["/"] = len(char2id_surface) + 1
        char2id_surface[">"] = len(char2id_surface) + 1
        char2id_root = defaultdict(int)
        for sentence in sentences:
            for word in sentence:
                for ch in word.surface_word:
                    if ch not in char2id_surface:
                        char2id_surface[ch] = len(char2id_surface) + 1
                for root in word.roots:
                    for ch in root:
                        if ch not in char2id_root:
                            char2id_root[ch] = len(char2id_root) + 1
        return char2id_surface, char2id_root

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
               test_data_paths=["data/data.test.txt"], model_file_name=None, char2id=None, tag2id=None, train_itearative=False):
        assert word_lstm_rep_len % 2 == 0
        if train_from_scratch:
            assert train_data_path
            assert len(test_data_paths) > 0
            print "Loading data..."
            if not train_itearative:
                self.train = self.load_data(train_data_path)
            if dev_data_path:
                self.dev = self.load_data(dev_data_path)
            else:
                self.dev = None
            self.test_paths = test_data_paths
            self.tests = []
            for test_path in self.test_paths:
                self.tests.append(self.load_data(test_path))
            print "Creating or Loading Vocabulary..."
            if char2id:
                self.char2id_surface = char2id
                self.char2id_root = char2id
            else:
                self.char2id_surface, self.char2id_root = self._create_vocab_chars(self.train)
            if tag2id:
                self.tag2id = tag2id
                self.surface_word2id = None
                self.root2id = None
            else:
                self.surface_word2id, self.root2id, self.tag2id = self._create_vocab_words(self.train)

            if not self.dev and not train_itearative:
                train_size = int(math.floor(0.99 * len(self.train)))
                self.dev = self.train[train_size:]
                self.train = self.train[:train_size]
            if train_itearative and not self.dev:
                self.dev = None

            self.model = dy.Model()
            self.trainer = dy.AdamTrainer(self.model)
            self.SURFACE_CHARS_LOOKUP = self.model.add_lookup_parameters(
                (len(self.char2id_surface) + 2, char_representation_len))
            self.ROOT_CHARS_LOOKUP = self.model.add_lookup_parameters(
                (len(self.char2id_root) + 2, char_representation_len))
            self.TAGS_LOOKUP = self.model.add_lookup_parameters((len(self.tag2id) + 2, char_representation_len))
            self.fwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.bwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.fwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.bwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.fwdRNN_tag = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.bwdRNN_tag = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.fwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
            self.bwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
            if train_itearative:
                self.iterative_training(train_data_path)
            else:
                self.train_model()
        else:
            print "Loading Pre-Trained Model"
            assert model_file_name
            if char2id:
                self.char2id_surface = char2id
                self.char2id_root = char2id
            if tag2id:
                self.tag2id = tag2id
            self.load_model(model_file_name, char_representation_len, word_lstm_rep_len)

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


    def propogate(self, sentence):
        dy.renew_cg()
        fwdRNN_surface_init = self.fwdRNN_surface.initial_state()
        bwdRNN_surface_init = self.bwdRNN_surface.initial_state()
        fwdRNN_root_init = self.fwdRNN_root.initial_state()
        bwdRNN_root_init = self.bwdRNN_root.initial_state()
        fwdRNN_tag_init = self.fwdRNN_tag.initial_state()
        bwdRNN_tag_init = self.bwdRNN_tag.initial_state()
        fwdRNN_context_init = self.fwdRNN_context.initial_state()
        bwdRNN_context_init = self.bwdRNN_context.initial_state()

        # CONTEXT REPRESENTATIONS
        surface_words_rep = []

        # SENTENCE BEGIN TAG REPRESENTATION
        for index, word in enumerate(sentence):
            encoded_surface_word = self._encode(word.surface_word, self.char2id_surface)
            surface_word_char_embeddings = self._embed(encoded_surface_word, self.SURFACE_CHARS_LOOKUP)
            fw_exps_surface_word = fwdRNN_surface_init.transduce(surface_word_char_embeddings)
            bw_exps_surface_word = bwdRNN_surface_init.transduce(reversed(surface_word_char_embeddings))
            surface_word_rep = dy.concatenate([fw_exps_surface_word[-1], bw_exps_surface_word[-1]])
            surface_words_rep.append(surface_word_rep)

        # SENTENCE END TAG REPRESENTATION
        fw_exps_context = fwdRNN_context_init.transduce(surface_words_rep)
        bw_exps_context = bwdRNN_context_init.transduce(reversed(surface_words_rep))
        scores = []
        # MORPH ANALYSIS REPRESENTATIONS
        for index, word in enumerate(sentence):
            encoded_roots = [self._encode(root, self.char2id_root) for root in word.roots]
            encoded_tags = [self._encode(tag, self.tag2id) for tag in word.tags]
            roots_embeddings = [self._embed(root, self.ROOT_CHARS_LOOKUP) for root in encoded_roots]
            tags_embeddings = [self._embed(tag, self.TAGS_LOOKUP) for tag in encoded_tags]
            analysis_representations = []
            for root_embedding, tag_embedding in zip(roots_embeddings, tags_embeddings):
                fw_exps_root = fwdRNN_root_init.transduce(root_embedding)
                bw_exps_root = bwdRNN_root_init.transduce(reversed(root_embedding))
                root_representation = dy.rectify(dy.concatenate([fw_exps_root[-1], bw_exps_root[-1]]))
                fw_exps_tag = fwdRNN_tag_init.transduce(tag_embedding)
                bw_exps_tag = bwdRNN_tag_init.transduce(reversed(tag_embedding))
                tag_representation = dy.rectify(dy.concatenate([fw_exps_tag[-1], bw_exps_tag[-1]]))
                analysis_representations.append(dy.rectify(dy.esum([root_representation, tag_representation])))

            left_context_rep = fw_exps_context[index]
            right_context_rep = bw_exps_context[len(sentence) - index - 1]
            context_rep = dy.tanh(dy.esum([left_context_rep, right_context_rep]))
            scores.append((dy.reshape(context_rep, (1, context_rep.dim()[0][0])) * dy.concatenate(analysis_representations, 1))[0])

        return scores

    def get_loss(self, sentence):
        scores = self.propogate(sentence)
        errs = []
        for score in scores:
            err = dy.pickneglogsoftmax(score, 0)
            errs.append(err)
        return dy.esum(errs)

    def predict_indices(self, sentence):
        selected_indices = []
        scores = self.propogate(sentence)
        for score in scores:
            probs = dy.softmax(score)
            selected_indices.append(np.argmax(probs.npvalue()))
        return selected_indices

    def predict(self, sentence):
        res = []
        selected_indices = self.predict_indices(sentence)
        for w, i in zip(sentence, selected_indices):
            res.append(w.roots[i] + "+" + "+".join(w.tags[i]))
        return res

    def calculate_acc(self, sentences, labels=None):
        corrects = 0
        non_ambigious_count = 0
        total = 0
        if not labels:
            labels = [[0 for w in sentence] for sentence in sentences]
        for sentence, sentence_labels in zip(sentences, labels):
            predicted_labels = self.predict_indices(sentence)
            corrects += [1 for l1, l2 in zip(sentence_labels, predicted_labels) if l1 == l2].count(1)
            non_ambigious_count += [1 for w in sentence if len(w.roots) == 1].count(1)
            total += len(sentence)
        return (corrects * 1.0 / total), ((corrects - non_ambigious_count) * 1.0 / (total - non_ambigious_count))

    def iterative_training(self, train_data_path, batch_size=22850, notify_size=22850, model_name="model", early_stop=False, num_epoch=4, train_dev_ratio=20):
        max_acc = 0.0
        epoch_loss = 0
        for epoch in xrange(num_epoch):
            t1 = datetime.now()
            count = 1
            with open(train_data_path, "r") as f:
                sentences = []
                for line in f:
                    trimmed_line = line.decode("utf-8").strip(" \r\n\t")
                    if trimmed_line.startswith("<S>") or trimmed_line.startswith("<s>"):
                        sentence = []
                    elif trimmed_line.startswith("</S>") or trimmed_line.startswith("</s>"):
                        if count % batch_size == 0 and len(sentences) > 0:
                            random.shuffle(sentences)
                            if count / batch_size == train_dev_ratio and not self.dev:
                                print "Calculating Accuracy on dev set"
                                print self.calculate_acc(sentences)
                            else:
                                for i, s in enumerate(sentences):
                                    loss_exp = self.get_loss(s)
                                    cur_loss = loss_exp.scalar_value()
                                    epoch_loss += cur_loss
                                    loss_exp.backward()
                                    self.trainer.update()
                                    if i > 0 and i % 100 == 0:  # print status
                                        t2 = datetime.now()
                                        delta = t2 - t1
                                        print("loss = {}  /  {} instances finished in  {} seconds".
                                              format(epoch_loss / (count - (batch_size - i) * 1.0),
                                                     count - (batch_size - i), delta.seconds))
                            sentences = []
                        if count % notify_size == 0:
                            if self.dev:
                                print "Calculating Accuracy on dev set"
                                acc, amb_acc = self.calculate_acc(self.dev)
                                print " accuracy: {}    ambiguous accuracy: {}".format(acc, amb_acc)
                                if acc > max_acc:
                                    max_acc = acc
                                    print "Max accuracy increased = {}, saving model...".format(str(max_acc))
                                    self.save_model(model_name)
                            print "Calculating Accuracy on test sets"
                            for q in range(len(self.test_paths)):
                                print "Calculating Accuracy on test set: {}".format(self.test_paths[q])
                                acc, amb_acc = self.calculate_acc(self.tests[q])
                                print " accuracy: {}    ambiguous accuracy: {}".format(acc, amb_acc)
                        if len(sentence) > 0:
                            sentences.append(sentence)
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
            print "epoch {} finished in {} minutes. loss = {}".format(epoch, delta.seconds / 60.0, epoch_loss / count * 1.0)
            epoch_loss = 0
            acc, amb_acc = self.calculate_acc(self.dev)
            print " accuracy on dev set: ", acc, " ambiguous accuracy on dev: ", amb_acc
            if acc > max_acc:
                max_acc = acc
                print "Max accuracy increased = {}, saving model...".format(str(max_acc))
                self.save_model(model_name)
            elif early_stop and max_acc - acc > 0.05:
                print "Max accuracy did not increase, early stopping!"
                break
            print "Calculating Accuracy on test sets"
            for q in range(len(self.test_paths)):
                print "Calculating Accuracy on test set: {}".format(self.test_paths[q])
                acc, amb_acc = self.calculate_acc(self.tests[q])
                print " accuracy: {}    ambiguous accuracy: {}".format(acc, amb_acc)


    def train_model(self, model_name="model", early_stop=False, num_epoch=20):
        max_acc = 0.0
        epoch_loss = 0
        for epoch in xrange(num_epoch):
            random.shuffle(self.train)
            t1 = datetime.now()
            count = 0
            for i, sentence in enumerate(self.train, 1):
                loss_exp = self.get_loss(sentence)
                cur_loss = loss_exp.scalar_value()
                epoch_loss += cur_loss
                loss_exp.backward()
                self.trainer.update()
                if i > 0 and i % 100 == 0:  # print status
                    t2 = datetime.now()
                    delta = t2 - t1
                    print("loss = {}  /  {} instances finished in  {} seconds".format(epoch_loss / (i * 1.0), i, delta.seconds))
                count = i
            t2 = datetime.now()
            delta = t2 - t1
            print "epoch {} finished in {} minutes. loss = {}".format(epoch, delta.seconds / 60.0, epoch_loss / count * 1.0)
            epoch_loss = 0
            acc, amb_acc = self.calculate_acc(self.dev)
            print " accuracy on dev set: ", acc, " ambiguous accuracy on dev: ", amb_acc
            if acc > max_acc:
                max_acc = acc
                print "Max accuracy increased = {}, saving model...".format(str(max_acc))
                self.save_model(model_name)
            elif early_stop and max_acc - acc > 0.05:
                print "Max accuracy did not incrase, early stopping!"
                break

            print "Calculating Accuracy on test sets"
            for q in range(len(self.test_paths)):
                print "Calculating Accuracy on test set: {}".format(self.test_paths[q])
                acc, amb_acc = self.calculate_acc(self.tests[q])
                print " accuracy: {}    ambiguous accuracy: {}".format(acc, amb_acc)

    def save_model(self, model_name):
        self.model.save("models/"+model_name+"-"+time.strftime("%d.%m.%Y")+".model")
        with open("models/"+model_name+"-"+time.strftime("%d.%m.%Y")+".char2id_root", "w") as f:
            pickle.dump(self.char2id_root, f)
        with open("models/"+model_name+"-"+time.strftime("%d.%m.%Y")+".char2id_surface", "w") as f:
            pickle.dump(self.char2id_surface, f)
        with open("models/"+model_name+"-"+time.strftime("%d.%m.%Y")+".tag2id", "w") as f:
            pickle.dump(self.tag2id, f)
        #with open("models/"+model_name+"-"+time.strftime("%d.%m.%Y")+".root2id", "w") as f:
        #    pickle.dump(self.root2id, f)
        #with open("models/"+model_name+"-"+time.strftime("%d.%m.%Y")+".surface_word2id", "w") as f:
        #    pickle.dump(self.surface_word2id, f)

    def load_model(self, model_name, char_representation_len, word_lstm_rep_len):
        with open("models/"+model_name+".char2id_root", "r") as f:
            self.char2id_root = pickle.load(f)
        with open("models/" + model_name + ".char2id_surface", "r") as f:
            self.char2id_surface = pickle.load(f)
        with open("models/"+model_name+".tag2id", "r") as f:
            self.tag2id = pickle.load(f)
        #with open("models/"+model_name+".root2id", "r") as f:
        #    self.root2id = pickle.load(f)
        #with open("models/"+model_name+".surface_word2id", "r") as f:
        #    self.surface_word2id = pickle.load(f)
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        self.SURFACE_CHARS_LOOKUP = self.model.add_lookup_parameters(
            (len(self.char2id_surface) + 2, char_representation_len))
        self.ROOT_CHARS_LOOKUP = self.model.add_lookup_parameters((len(self.char2id_root) + 2, char_representation_len))
        self.TAGS_LOOKUP = self.model.add_lookup_parameters((len(self.tag2id) + 2, char_representation_len))
        self.fwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.bwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.fwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.bwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.fwdRNN_tag = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.bwdRNN_tag = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.fwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
        self.bwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
        self.model.populate("models/" + model_name + ".model")

    @classmethod
    def create_from_existed_model(cls, model_path, char2id=None, tag2id=None):
        return MorphologicalDisambiguator(train_from_scratch=False, model_file_name=model_path,
                                          char2id=char2id, tag2id=tag2id)


if __name__ == "__main__":
    char2id = None
    tag2id = None
    #with open("defaultdic/char2id", "r") as f:
    #    char2id = pickle.load(f)
    #with open("defaultdic/tag2id", "r") as f:
    #    tag2id = pickle.load(f)
    #disambiguator = MorphologicalDisambiguator(train_from_scratch=True, char_representation_len=100, word_lstm_rep_len=200,
    #           train_data_path="data/gt8.txt",
    #           test_data_paths=["data/data.test.txt","data/test.merge","data/Morph.Dis.Test.Hand.Labeled-20K.txt"], model_file_name=None, char2id=char2id, tag2id=tag2id, train_itearative=True)

    #22.10.2017 -> best model
    #22.11.2017 -> trained on unambiguous 200K fragments greater than 8
    disambiguator = MorphologicalDisambiguator.create_from_existed_model("model-22.10.2017")
    print "Loading data"
    disambiguator.dev = disambiguator.load_data("data/data.dev.txt")
    disambiguator.test_paths = ["data/data.test.txt","data/test.merge","data/Morph.Dis.Test.Hand.Labeled-20K.txt"]
    disambiguator.tests = []
    for test_path in disambiguator.test_paths:
        disambiguator.tests.append(disambiguator.load_data(test_path))
    #disambiguator.train = disambiguator.load_data("data/data.train.txt")
    #print "Starting training..."
    #disambiguator.train_model()
    print "Loading test data"
    test_sentences = disambiguator.load_data("data/Morph.Dis.Test.Hand.Labeled-20K.txt")
    print "Calculating Accuracy on Morph.Dis.Test.Hand.Labeled-20K.txt"
    print disambiguator.calculate_acc(test_sentences)

    test_sentences = disambiguator.load_data("data/test.merge")
    print "Calculating Accuracy on test.merge"
    print disambiguator.calculate_acc(test_sentences)

    test_sentences = disambiguator.load_data("data/data.test.txt")
    print "Calculating Accuracy on data.test.txt"
    print disambiguator.calculate_acc(test_sentences)