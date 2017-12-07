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


class JointStemPosModel(object):
    SENTENCE_BEGIN_TAG = "<s>"
    SENTENCE_END_TAG = "</s>"

    WordStruct = namedtuple("WordStruct", ["surface_word", "roots", "suffixes", "postags"])
    analysis_regex = re.compile(r"^([^\\+]*)\+(.+)$", re.UNICODE)
    tag_seperator_regex = re.compile(r"[\\+\\^]", re.UNICODE)
    split_root_tags_regex = re.compile(r"^([^\\+]+)\+(.+)$", re.IGNORECASE)

    @classmethod
    def _create_vocab_chars(cls, sentences):
        char2id = defaultdict(int)
        char2id["<"] = len(char2id) + 1
        char2id["/"] = len(char2id) + 1
        char2id[">"] = len(char2id) + 1
        for sentence in sentences:
            for word in sentence:
                for ch in word.surface_word:
                    if ch not in char2id:
                        char2id[ch] = len(char2id) + 1
                for root in word.roots:
                    for ch in root:
                        if ch not in char2id:
                            char2id[ch] = len(char2id) + 1
        return char2id

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
               test_data_paths=["data/data.test.txt"], model_file_name=None, char2id=None, postag2id=None, unknown_tag="***UNKNOWN"):
        assert word_lstm_rep_len % 2 == 0
        if train_from_scratch:
            assert train_data_path
            assert len(test_data_paths) > 0
            print "Loading data..."
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
                self.char2id = char2id
            else:
                self.char2id = self._create_vocab_chars(self.train)
            if postag2id:
                self.postag2id = tag2id
            else:
                self.postag2id = {unknown_tag: 0, "Noun": 1, "Verb": 2, "Adj": 3, "Adv": 4, "Pron": 5, "Conj": 6,
                                "Interj": 7, "Punc": 8, "Num": 9, "Det": 10, "Postp": 11, "Adverb": 12, "Ques": 13}

            if not self.dev:
                train_size = int(math.floor(0.99 * len(self.train)))
                self.dev = self.train[train_size:]
                self.train = self.train[:train_size]

            self.model = dy.Model()
            self.trainer = dy.AdamTrainer(self.model)
            self.SURFACE_CHARS_LOOKUP = self.model.add_lookup_parameters(
                (len(self.char2id) + 2, char_representation_len))
            self.ROOT_CHARS_LOOKUP = self.model.add_lookup_parameters(
                (len(self.char2id) + 2, char_representation_len))
            self.fwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.bwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.fwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.bwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.fwdRNN_suffix = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.bwdRNN_suffix = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.pW = self.model.add_parameters((word_lstm_rep_len, len(self.postag2id)))
            self.pb = self.model.add_parameters(len(self.postag2id))
            self.fwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
            self.bwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
            self.train_model(model_name=model_file_name)
        else:
            print "Loading Pre-Trained Model"
            assert model_file_name
            if char2id:
                self.char2id = char2id
            if postag2id:
                self.postag2id = postag2id
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

    def _get_pos_from_analysis(self, analysis):
        tags = self._get_tagsstr_from_analysis(analysis)
        if "^" in tags:
            tags = tags[tags.rfind("^") + 4:]
        return tags.split("+")[0]

    def _get_tagsstr_from_analysis(self, analysis):
        if analysis.startswith("+"):
            return analysis[2:]
        else:
            return self.split_root_tags_regex.sub(r"\2", analysis)

    def generate_candidate_roots_and_suffixes(self, surface_word):
        candidate_roots = []
        candidate_suffixes = []
        for i in range(1,len(surface_word)):
            candidate_roots.append(surface_word[:i])
            candidate_suffixes.append(surface_word[i:])
        candidate_roots.append(surface_word)
        candidate_suffixes.append("")
        return candidate_roots, candidate_suffixes

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
                    surface = parses[0].lower()
                    candidate_roots, candidate_suffixes = self.generate_candidate_roots_and_suffixes(surface)
                    assert len(candidate_roots) == len(candidate_suffixes)
                    analyzes = parses[1:]
                    gold_root = self._get_root_from_analysis(analyzes[0]).lower()
                    roots = []
                    suffixes = []
                    roots.append(gold_root)
                    gold_suffix = surface[len(gold_root):]
                    suffixes.append(gold_suffix)
                    for candidate_root, candidate_suffix in zip(candidate_roots, candidate_suffixes):
                        if candidate_root != gold_root and candidate_suffix != gold_suffix:
                            roots.append(candidate_root)
                            suffixes.append(candidate_suffix)
                    postags = []
                    for analysis in analyzes:
                        cur_postag = self._get_pos_from_analysis(analysis)
                        if cur_postag not in postags:
                            postags.append(cur_postag)
                    current_word = self.WordStruct(surface, roots, suffixes, postags)
                    sentence.append(current_word)
        return sentences

    def propogate(self, sentence):
        dy.renew_cg()
        fwdRNN_surface_init = self.fwdRNN_surface.initial_state()
        bwdRNN_surface_init = self.bwdRNN_surface.initial_state()
        fwdRNN_root_init = self.fwdRNN_root.initial_state()
        bwdRNN_root_init = self.bwdRNN_root.initial_state()
        fwdRNN_suffix_init = self.fwdRNN_suffix.initial_state()
        bwdRNN_suffix_init = self.bwdRNN_suffix.initial_state()
        fwdRNN_context_init = self.fwdRNN_context.initial_state()
        bwdRNN_context_init = self.bwdRNN_context.initial_state()
        W = dy.parameter(self.pW)
        b = dy.parameter(self.pb)

        # CONTEXT REPRESENTATIONS
        surface_words_rep = []
        for index, word in enumerate(sentence):
            encoded_surface_word = self._encode(word.surface_word, self.char2id)
            surface_word_char_embeddings = self._embed(encoded_surface_word, self.SURFACE_CHARS_LOOKUP)
            fw_exps_surface_word = fwdRNN_surface_init.transduce(surface_word_char_embeddings)
            bw_exps_surface_word = bwdRNN_surface_init.transduce(reversed(surface_word_char_embeddings))
            surface_word_rep = dy.concatenate([fw_exps_surface_word[-1], bw_exps_surface_word[-1]])
            surface_words_rep.append(surface_word_rep)
        fw_exps_context = fwdRNN_context_init.transduce(surface_words_rep)
        bw_exps_context = bwdRNN_context_init.transduce(reversed(surface_words_rep))
        root_scores = []
        postag_scores = []
        # Stem and POS REPRESENTATIONS
        for index, word in enumerate(sentence):
            encoded_roots = [self._encode(root, self.char2id) for root in word.roots]
            encoded_suffixes = [self._encode(suffix, self.char2id) for suffix in word.suffixes]
            roots_embeddings = [self._embed(root, self.ROOT_CHARS_LOOKUP) for root in encoded_roots]
            suffix_embeddings = [self._embed(suffix, self.ROOT_CHARS_LOOKUP) for suffix in encoded_suffixes]
            root_stem_representations = []
            for root_embedding, suffix_embedding in zip(roots_embeddings, suffix_embeddings):
                fw_exps_root = fwdRNN_root_init.transduce(root_embedding)
                bw_exps_root = bwdRNN_root_init.transduce(reversed(root_embedding))
                root_representation = dy.rectify(dy.concatenate([fw_exps_root[-1], bw_exps_root[-1]]))
                if len(suffix_embedding) != 0:
                    fw_exps_suffix = fwdRNN_suffix_init.transduce(suffix_embedding)
                    bw_exps_suffix = bwdRNN_suffix_init.transduce(reversed(suffix_embedding))
                    suffix_representation = dy.rectify(dy.concatenate([fw_exps_suffix[-1], bw_exps_suffix[-1]]))
                    root_stem_representations.append(dy.rectify(dy.esum([root_representation, suffix_representation])))
                else:
                    root_stem_representations.append(root_representation)

            left_context_rep = fw_exps_context[index]
            right_context_rep = bw_exps_context[len(sentence) - index - 1]
            context_rep = dy.tanh(dy.esum([left_context_rep, right_context_rep]))
            root_scores.append(
                (dy.reshape(context_rep, (1, context_rep.dim()[0][0])) * dy.concatenate(root_stem_representations, 1))[0])
            postag_scores.append(
                (dy.reshape(context_rep, (1, context_rep.dim()[0][0])) * W + b)[0]
            )

        return root_scores, postag_scores

    def get_loss(self, sentence):
        root_scores, postag_scores = self.propogate(sentence)
        errs = []
        for i, (root_score, postag_score) in enumerate(zip(root_scores, postag_scores)):
            root_err = dy.pickneglogsoftmax(root_score, 0)
            errs.append(root_err)
            gold_postag = sentence[i].postags[0]
            if gold_postag in self.postag2id:
                pos_err = dy.pickneglogsoftmax(postag_score, self.postag2id[sentence[i].postags[0]])
                errs.append(pos_err)
        return dy.esum(errs)

    def predict_indices(self, sentence):
        selected_root_indices = []
        selected_postag_indices = []
        root_scores, postag_scores = self.propogate(sentence)
        for i, (root_score, postag_score) in enumerate(zip(root_scores, postag_scores)):
            root_probs = dy.softmax(root_score)
            selected_root_indices.append(np.argmax(root_probs.npvalue()))
            postag_probs = dy.softmax(postag_score)
            selected_postag_indices.append(np.argmax(postag_probs.npvalue()))
        return selected_root_indices, selected_postag_indices

    def calculate_acc(self, sentences, labels=None):
        reverse_postag_index = {v: k for k, v in self.postag2id.iteritems()}
        root_corrects = 0
        postag_corrects = 0
        total = 0
        if not labels:
            labels = [[0 for w in sentence] for sentence in sentences]
        for sentence, sentence_labels in zip(sentences, labels):
            selected_root_indices, selected_postag_indices = self.predict_indices(sentence)
            root_corrects += [1 for l1, l2 in zip(sentence_labels, selected_root_indices) if l1 == l2].count(1)
            postag_corrects += [1 for w, p_index in zip(sentence, selected_postag_indices) if
                                w.postags[0] == reverse_postag_index[p_index]].count(1)
            total += len(sentence)
        return (root_corrects * 1.0) / total, (postag_corrects * 1.0) / total

    def train_model(self, model_name="model", early_stop=False, num_epoch=20):
        max_root_acc = 0.0
        max_pos_acc = 0.0
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
            root_acc, postag_acc = self.calculate_acc(self.dev)
            print "Calculating Accuracy on dev set"
            print "Root accuracy on dev set:{}\nPostag accuracy on dev set:{} ".format(root_acc, postag_acc)
            if root_acc > max_root_acc and postag_acc > max_pos_acc:
                max_root_acc = root_acc
                max_pos_acc = postag_acc
                print "Max accuracy increased, saving model..."
                self.save_model(model_name)
            elif early_stop and (max_root_acc - root_acc) + (max_pos_acc - postag_acc) > 0.1:
                print "Max accuracy did not incrase, early stopping!"
                break

            print "Calculating Accuracy on test sets"
            for q in range(len(self.test_paths)):
                print "Calculating Accuracy on test set: {}".format(self.test_paths[q])
                root_acc, postag_acc = self.calculate_acc(self.tests[q])
                print "Root accuracy on test set:{}\nPostag accuracy on dev set:{} ".format(root_acc, postag_acc)

    def save_model(self, model_name):
        self.model.save("models/"+model_name+"-"+time.strftime("%d.%m.%Y")+".model")
        with open("models/"+model_name+"-"+time.strftime("%d.%m.%Y")+".char2id", "w") as f:
            pickle.dump(self.char2id, f)
        with open("models/"+model_name+"-"+time.strftime("%d.%m.%Y")+".tag2id", "w") as f:
            pickle.dump(self.postag2id, f)

    def load_model(self, model_name, char_representation_len, word_lstm_rep_len):
        with open("models/"+model_name+".char2id", "r") as f:
            self.char2id = pickle.load(f)
        with open("models/"+model_name+".tag2id", "r") as f:
            self.postag2id = pickle.load(f)

        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        self.SURFACE_CHARS_LOOKUP = self.model.add_lookup_parameters(
            (len(self.char2id) + 2, char_representation_len))
        self.ROOT_CHARS_LOOKUP = self.model.add_lookup_parameters(
            (len(self.char2id) + 2, char_representation_len))
        self.fwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.bwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.fwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.bwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.fwdRNN_suffix = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.bwdRNN_suffix = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.pW = self.model.add_parameters((word_lstm_rep_len, len(self.postag2id)))
        self.pb = self.model.add_parameters(len(self.postag2id))
        self.fwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
        self.bwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
        self.model.populate("models/" + model_name + ".model")

    @classmethod
    def create_from_existed_model(cls, model_name, char2id=None, tag2id=None):
        return JointStemPosModel(train_from_scratch=False, model_file_name=model_name,
                                          char2id=char2id, postag2id=tag2id)

    def predict(self, tokens):
        reverse_postag_index = {v: k for k, v in self.postag2id.iteritems()}
        sentence = []
        for token in tokens:
            token = token.lower().decode("utf-8")
            candidate_roots, candidate_suffixes = self.generate_candidate_roots_and_suffixes(token)
            assert len(candidate_roots) == len(candidate_suffixes)
            current_word = self.WordStruct(token, candidate_roots, candidate_suffixes, None)
            sentence.append(current_word)

        selected_root_indices, selected_postag_indices = self.predict_indices(sentence)
        res = []
        for w, root_i, postag_i in zip(sentence, selected_root_indices, selected_postag_indices):
            res.append(w.roots[root_i].encode("utf-8") + "+" + w.suffixes[root_i].encode("utf-8")
                       + "[" +reverse_postag_index[postag_i].encode("utf-8") + "]")
        return res

if __name__ == "__main__":
    #char2id = None
    #tag2id = None
    #JointStemPosModel(train_data_path="data/data.train.txt", dev_data_path="data/data.dev.txt",
    #                  test_data_paths=["data/test.merge", "data/data.test.txt", "data/Morph.Dis.Test.Hand.Labeled-20K.txt"],
    #                  model_file_name="postagger_stemmer_shen")

    stemmer  = JointStemPosModel.create_from_existed_model(model_name="postagger_stemmer_shen")
    print stemmer.predict(["bugünlerde", "canım","çok", "sıkkın", "."])
    print stemmer.predict(["elmasını", "yedim", "."])
    print stemmer.predict(["elmasini", "yedim", "."])
    print stemmer.predict(["arabalandım", "."])