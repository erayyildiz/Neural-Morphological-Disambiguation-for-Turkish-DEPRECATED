import random
import re
import math
import sys
from collections import defaultdict
from collections import namedtuple
from pprint import pprint

import dynet as dy
import numpy as np

SENTENCE_BEGIN_TAG = "<s>"
SENTENCE_END_TAG = "</s>"

WordStruct = namedtuple("WordStruct", ["surface_word", "roots", "tags"])
analysis_regex = re.compile(r"^([^\+]*)\+(.+)$", re.UNICODE)
tag_seperator_regex = re.compile(r"[\+\^]", re.UNICODE)


def encode(tokens, vocab):
    return [vocab[token] for token in tokens]


def embed(token, char_embedding_table):
    return [char_embedding_table[ch] for ch in token]


def print_namedtuple(nt):
    pprint(dict(nt._asdict()))


def _get_root_from_analysis(analysis):
    if analysis.startswith("+"):
        return "+"
    else:
        return analysis_regex.sub(r"\1", analysis)


def _get_tags_from_analysis(analysis):
    if analysis.startswith("+"):
        return tag_seperator_regex.split(analysis[2:])
    else:
        return tag_seperator_regex.split(analysis_regex.sub(r"\2", analysis))


def load_data(file_path, max_sentence=sys.maxint):
    sentence = []
    sentences = []
    with open(file_path, 'r') as f:
        for line in f:
            trimmed_line = line.decode("utf-8").strip(" \r\n\t")
            if trimmed_line.startswith("<S>"):
                sentence = []
            elif trimmed_line.startswith("</S>"):
                if len(sentence) > 0:
                    sentences.append(sentence)
                    if len(sentences) > max_sentence:
                        return sentences
            elif "<DOC>" in trimmed_line or trimmed_line.startswith("</DOC>") or trimmed_line.startswith(
                    "<TITLE>") or trimmed_line.startswith("</TITLE>"):
                pass
            else:
                parses = trimmed_line.split(" ")
                surface = parses[0]
                analyzes = parses[1:]
                roots = [_get_root_from_analysis(analysis) for analysis in analyzes]
                tags = [_get_tags_from_analysis(analysis) for analysis in analyzes]
                current_word = WordStruct(surface, roots, tags)
                sentence.append(current_word)
    return sentences


def create_vocab_words(sentences):
    surface_word2id = defaultdict(int)
    surface_word2id[SENTENCE_BEGIN_TAG] = len(surface_word2id) + 1
    surface_word2id[SENTENCE_END_TAG] = len(surface_word2id) + 1
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


def create_vocab_chars(sentences):
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


test_manual_labeled = load_data("test.merge")
test = load_data("test.1.2.dis")
train = load_data("train.merge")

char2id_surface, char2id_root = create_vocab_chars(train)
surface_word2id, root2id, tag2id = create_vocab_words(train)

dy.init()
model = dy.Model()
trainer = dy.AdamTrainer(model)
SURFACE_CHARS_LOOKUP = model.add_lookup_parameters((len(char2id_surface)+2, 20))
ROOT_CHARS_LOOKUP = model.add_lookup_parameters((len(char2id_root)+2, 20))
TAGS_LOOKUP = model.add_lookup_parameters((len(tag2id)+2, 20))
fwdRNN_surface = dy.LSTMBuilder(1, 20, 100, model)
bwdRNN_surface = dy.LSTMBuilder(1, 20, 100, model)
fwdRNN_root = dy.LSTMBuilder(1, 20, 100, model)
bwdRNN_root = dy.LSTMBuilder(1, 20, 100, model)
fwdRNN_tag = dy.LSTMBuilder(1, 20, 100, model)
bwdRNN_tag = dy.LSTMBuilder(1, 20, 100, model)
fwdRNN_context = dy.LSTMBuilder(1, 200, 200, model)
bwdRNN_context = dy.LSTMBuilder(1, 200, 200, model)


def propogate(sentence):
    dy.renew_cg()
    fwdRNN_surface_init = fwdRNN_surface.initial_state()
    bwdRNN_surface_init = bwdRNN_surface.initial_state()
    fwdRNN_root_init = fwdRNN_root.initial_state()
    bwdRNN_root_init = bwdRNN_root.initial_state()
    fwdRNN_tag_init = fwdRNN_tag.initial_state()
    bwdRNN_tag_init = bwdRNN_tag.initial_state()
    fwdRNN_context_init = fwdRNN_context.initial_state()
    bwdRNN_context_init = bwdRNN_context.initial_state()

    # CONTEXT REPRESENTATIONS
    surface_words_rep = []

    #SENTENCE BEGIN TAG REPRESENTATION
    encoded_sentence_begin_tag = encode(SENTENCE_BEGIN_TAG, char2id_surface)
    sentence_begin_tag_char_embeddings = embed(encoded_sentence_begin_tag, SURFACE_CHARS_LOOKUP)
    fw_exps_surface_sentence_begin_tag = fwdRNN_surface_init.transduce(sentence_begin_tag_char_embeddings)
    bw_exps_surface_sentence_begin_tag = bwdRNN_surface_init.transduce(reversed(sentence_begin_tag_char_embeddings))
    sentence_begin_tag_rep = dy.concatenate([fw_exps_surface_sentence_begin_tag[-1], bw_exps_surface_sentence_begin_tag[-1]])
    surface_words_rep.append(sentence_begin_tag_rep)

    for index, word in enumerate(sentence):
        encoded_surface_word = encode(word.surface_word, char2id_surface)
        surface_word_char_embeddings = embed(encoded_surface_word, SURFACE_CHARS_LOOKUP)
        fw_exps_surface_word = fwdRNN_surface_init.transduce(surface_word_char_embeddings)
        bw_exps_surface_word = bwdRNN_surface_init.transduce(reversed(surface_word_char_embeddings))
        surface_word_rep = dy.concatenate([fw_exps_surface_word[-1], bw_exps_surface_word[-1]])
        surface_words_rep.append(surface_word_rep)

    # SENTENCE END TAG REPRESENTATION
    encoded_sentence_end_tag = encode(SENTENCE_END_TAG, char2id_surface)
    sentence_end_tag_char_embeddings = embed(encoded_sentence_end_tag, SURFACE_CHARS_LOOKUP)
    fw_exps_surface_sentence_end_tag = fwdRNN_surface_init.transduce(sentence_end_tag_char_embeddings)
    bw_exps_surface_sentence_end_tag = bwdRNN_surface_init.transduce(reversed(sentence_end_tag_char_embeddings))
    sentence_end_tag_rep = dy.concatenate(
        [fw_exps_surface_sentence_end_tag[-1], bw_exps_surface_sentence_end_tag[-1]])
    surface_words_rep.append(sentence_end_tag_rep)

    fw_exps_context = fwdRNN_context_init.transduce(surface_words_rep)
    bw_exps_context = bwdRNN_context_init.transduce(reversed(surface_words_rep))
    scores = []
    # MORPH ANALYSIS REPRESENTATIONS
    for index, word in enumerate(sentence):
        encoded_roots = [encode(root, char2id_root) for root in word.roots]
        encoded_tags = [encode(tag, tag2id) for tag in word.tags]
        roots_embeddings = [embed(root, ROOT_CHARS_LOOKUP) for root in encoded_roots]
        tags_embeddings = [embed(tag, TAGS_LOOKUP) for tag in encoded_tags]
        analysis_representations = None
        for root_embedding, tag_embedding in zip(roots_embeddings, tags_embeddings):
            fw_exps_root = fwdRNN_root_init.transduce(root_embedding)
            bw_exps_root = bwdRNN_root_init.transduce(reversed(root_embedding))
            root_representation = dy.rectify(dy.concatenate([fw_exps_root[-1], bw_exps_root[-1]]))
            fw_exps_tag = fwdRNN_tag_init.transduce(tag_embedding)
            bw_exps_tag = bwdRNN_tag_init.transduce(reversed(tag_embedding))
            tag_representation = dy.rectify(dy.concatenate([fw_exps_tag[-1], bw_exps_tag[-1]]))
            if analysis_representations:
                analysis_representations = dy.concatenate([analysis_representations, dy.rectify(
                    dy.colwise_add(
                        dy.reshape(root_representation, (root_representation.dim()[0][0], 1)),
                        tag_representation))], 1)
            else:
                analysis_representations = dy.rectify(
                    dy.colwise_add(
                        dy.reshape(root_representation, (root_representation.dim()[0][0], 1)),
                        tag_representation))
        left_context_rep = fw_exps_context[index]
        right_context_rep = bw_exps_context[len(sentence) - index]
        context_rep = dy.tanh(
            dy.colwise_add(
                dy.reshape(left_context_rep, (left_context_rep.dim()[0][0], 1)),
                right_context_rep))
        scores.append((dy.transpose(context_rep) * analysis_representations)[0])

    return scores


def get_loss(sentence):
    scores = propogate(sentence)
    errs = []
    for score in scores:
        err = dy.pickneglogsoftmax(score, 0)
        errs.append(err)
    return dy.esum(errs)


def predict(sentence):
    selected_indices = []
    scores = propogate(sentence)
    for score in scores:
        probs = dy.softmax(score)
        selected_indices.append(np.argmax(probs.npvalue()))
    return selected_indices

def calculate_acc(sentences, labels=None):
    corrects = 0
    non_ambigious_count = 0
    total = 0
    if not labels:
        labels = [[0 for w in sentence] for sentence in sentences]
    for sentence, sentence_labels in zip(sentences, labels):
        predicted_labels = predict(sentence)
        corrects += [1 for l1, l2 in zip(sentence_labels, predicted_labels) if l1 == l2 ].count(1)
        non_ambigious_count += [1 for w in sentence if len(w.roots) == 1].count(1)
        total += len(sentence)
    return (corrects * 1.0 / total), ((corrects - non_ambigious_count) * 1.0 / (total - non_ambigious_count) )


train_size = int(math.floor(0.9 * len(train)))
dev = train[train_size:]
train = train[:train_size]

num_tagged = cum_loss = 0
for epoch in xrange(5):
    random.shuffle(train)
    for i, sentence in enumerate(train, 1):
        dy.renew_cg()
        if i > 0 and i % 100 == 0:  # print status
            acc, amb_acc = calculate_acc(dev)
            print epoch, ". epoch ", i, ".instance", " loss: ", cum_loss / num_tagged, " accuracy on dev: ", acc, " ambiguous accuracy on dev: ", amb_acc
            cum_loss = num_tagged = 0
            num_tagged = 0
        loss_exp = get_loss(sentence)
        cum_loss += loss_exp.scalar_value()
        num_tagged += len(sentence)
        loss_exp.backward()
        trainer.update()
    print "epoch %r finished" % epoch
    #trainer.update_epoch(1.0)
