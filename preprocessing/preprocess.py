#!/usr/bin/env python
# coding: utf8

import glob
import time
import json
import multiprocessing
import Queue
#from multiprocessing import Process, Queue, JoinableQueue, cpu_count
from sqlitedict import SqliteDict
import argparse
from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

#PROCESSES = (multiprocessing.cpu_count() * 2) - 1
PROCESSES = 2
STOPWORDS_FILE = "stopwords.txt"

class ArticlePreprocesser(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.ner_tagger = get_NER_tagger()
        self.lemmatizer = WordNetLemmatizer()
        with open(STOPWORDS_FILE, 'r') as _file:
            self.stopwords = _file.readlines()

    def run(self):
        proc_name = self.name
        while True:
            article = self.task_queue.get()
            if article is None:
                # Poison pill means shutdown
                print '%s: Exiting' % proc_name
                self.task_queue.task_done()
                break
            try:
                print '{}: processing article {}'.format(proc_name, article.id + "-" + article.title)
            except:
                print '{}: processing article {}'.format(proc_name, article.id)
            article.sentence_tokenize()
            article.tag_ner(self.ner_tagger)
            article.tag_pos()
            article.lemmatize(self.lemmatizer)
            article.remove_stopwords(self.stopwords)
            article.assemble_bow()
            self.task_queue.task_done()
            self.result_queue.put(article)
        return


class Word:
    def __init__(self, raw):
        self.raw = raw
        self.ner = ''
        self.pos = ''
        self.stem = raw
        self.is_stopword = False


class Article:
    punctuation = [',', '.', '!', '?']

    def __init__(self, filename=None, raw=None):
        self.sentences = []
        self.words = []
        self.ner = []
        # from file:
        if filename is not None:
            self.text = raw
            self.id = filename
            self.url = "file:///" + filename
            self.title = filename
        # from json with specific fields
        else:
            try:
                _json = json.loads(raw)
                self.text = _json["text"]
                self.id = _json["id"]
                self.url = _json["url"]
                self.title = _json["title"]
            except ValueError:
                raise NotAnArticle
        self.text = self.text.encode('utf-8')
        self.title = self.title.encode('utf-8')

    def tokenize(self):
        self.sentences = sent_tokenize(self.text)
        self.words = [[Word(w) for w in word_tokenize(sent)] for sent in self.sentences]

    def tag_ner(self, tagger):
        for sentence in self.words:
            ners = (t[1] for t in tagger.tag((w.raw for w in sentence)))
            for word, ner in zip(sentence, ners):
                word.ner = ner

    def tag_pos(self):
        for sentence in self.words:
            poses = (t[1] for t in pos_tag((w.raw for w in sentence)))
            for word, pos in zip(sentence, poses):
                word.pos = pos

    def remove_stopwords(self, stops):
        for sentence in self.words:
            for word in sentence[:]:
                if not word.ner and word.raw in stops:
                    sentence.remove(word)

    def lemmatize(self, lemmatizer):
        def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            else:
                return ''

        for sentence in self.words:
            for word in sentence:
                lemmatizer_pos = get_wordnet_pos(word.pos)
                word.stem = lemmatizer.lemmatize(word.raw, lemmatizer_pos)

    def assemble_bow(self):
        pass


class NotAnArticle(Exception):
    def __init__(self):
        Exception.__init__(self)


def read_from_directory(path):
    print "reading files from glob pattern: {}".format(path)
    for filename in glob.glob(path):
        with open(filename, 'r') as _file:
            for line in _file:
                try:
                    yield Article(raw=line)
                except NotAnArticle:
                    _file.seek(0)
                    yield Article(filename=filename, raw=_file.read())



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_directory", "-i", help="a path pattern of which files to read as input; e.g. test_data/*/*")
    parser.add_argument("--output_sqlite", "-o")
    return parser.parse_args()


def get_NER_tagger(model_location=None, jar_location=None):
    if model_location is None:
        model_location = "/Users/mgough/hierarchical-topic-modeling/stanford-ner-2017-06-09/classifiers/english.conll.4class.distsim.crf.ser.gz"
    if jar_location is None:
        jar_location = "/Users/mgough/hierarchical-topic-modeling/stanford-ner-2017-06-09/stanford-ner-3.8.0.jar"

    return StanfordNERTagger(model_location, jar_location, encoding='utf-8')


def reader_process(directory, queue):
    for article in read_from_directory(directory):
        print "reading: {}".format(article.title)
        queue.put(article)
    for x in xrange(PROCESSES):
        queue.put(None)


def main():
    t0 = time.time()
    args = parse_arguments()
    output_sqlite = SqliteDict(args.output_sqlite, autocommit=True)
    in_queue = multiprocessing.JoinableQueue()
    out_queue = multiprocessing.Queue()

    # instantiate processes

    # a process to load the queue with file contents
    reader = multiprocessing.Process(target=reader_process, args=(args.corpus_directory, in_queue))
    reader.daemon = True
    reader.start()

    writers = [ArticlePreprocesser(in_queue, out_queue) for cpu in xrange(PROCESSES)]
    for writer in writers:
        writer.daemon = True
        writer.start()

    # write the results as they come in
    while True:
        try:
            processed = out_queue.get(timeout=1)
            try:
                print 'writing processed article: {}'.format(processed.id + "-" + processed.title)
            except:
                print 'writing processed article: {}'.format(processed.id)
            output_sqlite[processed.id] = processed
        except (KeyboardInterrupt, Queue.Empty):
            print "exiting..."
            output_sqlite.close()
            print "program finished in {}".format(time.time() - t0)
            exit()


if __name__ == "__main__":
    main()


