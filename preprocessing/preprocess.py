#!/usr/bin/env python
# coding: utf8

import codecs
import glob
import itertools
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
from sner import Ner
from string import punctuation

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

STOPWORDS_FILE = "stopwords.txt"

class ArticlePreprocesser(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.ner_tagger = get_NER_tagger()
        self.lemmatizer = WordNetLemmatizer()
        with open(STOPWORDS_FILE, 'r') as _file:
            self.stopwords = [w.strip() for w in _file.readlines()]

    def run(self):
        import sys
        reload(sys)
        sys.setdefaultencoding("utf-8")
        proc_name = self.name
        while True:
            article = self.task_queue.get()
            if article is None:
                self.result_queue.put(None)
                # Poison pill means shutdown
                print '%s: Exiting' % proc_name
                self.task_queue.task_done()
                break

            try:
                print '{}: processing article {}'.format(proc_name, article.id + "-" + article.title)

                article.tokenize()
                #print '{} remove stops on {}'.format(proc_name, article.title)
                #article.remove_stopwords(self.stopwords)

                print'{} tagging ner on {}'.format(proc_name, article.title)
                article.tag_ner(self.ner_tagger)

                print '{} tagging pos on {}'.format(proc_name, article.title)
                article.tag_pos()
                print '{} lemmatize on {}'.format(proc_name, article.title)
                article.lemmatize(self.lemmatizer)

                print '{} assemble {}'.format(proc_name, article.title)
                article.assemble_bow(self.stopwords)
                print '{} done with {}'.format(proc_name, article.title)
            except UnicodeDecodeError:
                print '{}: unicode error on {}. aborting'.format(proc_name, article.title)
                continue

            self.task_queue.task_done()
            self.result_queue.put(article)
        return


class Word:
    def __init__(self, raw):
        self.raw = raw.lower()
        self.ner = ''
        self.pos = ''
        self.stem = raw
        self.is_stopword = False

    def __repr__(self):
        return self.raw


class Article:
    def __init__(self, filename=None, raw=None):
        self.sentences = []
        self.words = []
        self.ner = []
        self.bow = []
        # from file:
        if filename is not None:
            self.text = raw.decode('utf-8')
            self.id = filename
            self.url = u"file:///" + filename.decode('utf-8')
            self.title = filename
        # from json with specific fields
        else:
            try:
                _json = json.loads(raw)
                self.text = _json[u"text"]
                self.id = _json[u"id"]
                self.url = _json[u"url"]
                self.title = _json[u"title"]
            except ValueError:
                raise NotAnArticle
        self.text = self.text.encode('utf-8')
        self.title = self.title.encode('utf-8')

    def __repr__(self):
        return self.bow

    def tokenize(self):
        self.sentences = [x for x in sent_tokenize(self.text)]
        self.words = [[Word(w) for w in word_tokenize(sent)] for sent in self.sentences]

    def tag_ner(self, tagger):
        for sentence in self.words:
            ners = (t[1] for t in tagger.get_entities(" ".join([w.raw for w in sentence])))
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
                if word.ner=='O' and word.raw in stops:
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
                if lemmatizer_pos == '':
                    word.stem = lemmatizer.lemmatize(word.raw)
                else:
                    word.stem = lemmatizer.lemmatize(word.raw, lemmatizer_pos)

    def assemble_bow(self, stopwords):
        def grouper(n, iterable, fillvalue=None):
            args = [iter(iterable)] * n
            return itertools.izip_longest(*args, fillvalue='')

        for sentence in self.words:
            for label, group in itertools.groupby(sentence, lambda x: x.ner):
                group = list(group)
                if label == 'O':
                    for word in group[:]:
                        if word.raw in stopwords or word.raw in punctuation:
                            group.remove(word)
                        else:
                            self.bow.append(word.stem)
                else:
                    if len(group) == 1 and str(group[0].raw) in stopwords:
                        continue
                    conglomerate = label + ":" + "_".join([str(g.raw) for g in group if g.raw not in punctuation])
                    if len(group) >3:
                        print "found group of {}S:".format(label)
                        print conglomerate
                        print "\n"
                    self.bow.append(conglomerate)
                #elif label == 'PERSON' and len(group) > 3 and len(group) % 2 == 0:  #hacks! we have multiple first and last names prolly
                #    print "PERSON: {}".format("_".join([w.raw for w in group]))
                        #self.bow.extend(name)
                #    exit()
                #else:
                #    conglomerate = label + ":" + "_".join([str(g.raw) for g in group])
                #    try:
                #        print u"found entity: {}".format(conglomerate)
                #    except:
                #        pass
                #    self.bow.append(conglomerate)

class NotAnArticle(Exception):
    def __init__(self):
        Exception.__init__(self)


def read_from_directory(path):
    print "reading files from glob pattern: {}".format(path)
    for filename in glob.glob(path):
        with codecs.open(filename, 'r', encoding='utf-8') as _file:
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
    parser.add_argument("--processes", "-p", help="number of reader processes to use")
    return parser.parse_args()


def get_NER_tagger(model_location=None, jar_location=None):
    if model_location is None:
        model_location = "/Users/mgough/hierarchical-topic-modeling/stanford-ner-2017-06-09/classifiers/english.conll.4class.distsim.crf.ser.gz"
    if jar_location is None:
        jar_location = "/Users/mgough/hierarchical-topic-modeling/stanford-ner-2017-06-09/stanford-ner-3.8.0.jar"

    tagger = Ner(host='localhost', port=9199)
    return tagger
    #return StanfordNERTagger(model_location, jar_location, encoding='utf-8')


def reader_process(directory, queue, processes):
    for article in read_from_directory(directory):
        #print "reading: {}".format(article.title)
        queue.put(article)
    for x in xrange(processes):
        queue.put(None)


def main():
    t0 = time.time()
    args = parse_arguments()
    if args.processes is not None:
        PROCESSES = int(args.processes)
    else:
        PROCESSES = (multiprocessing.cpu_count() * 2) - 1
    output_sqlite = SqliteDict(args.output_sqlite, autocommit=True)
    in_queue = multiprocessing.JoinableQueue()
    out_queue = multiprocessing.Queue()

    # instantiate processes

    # a process to load the queue with file contents
    reader = multiprocessing.Process(target=reader_process, args=(args.corpus_directory, in_queue, PROCESSES))
    reader.daemon = True
    reader.start()

    writers = [ArticlePreprocesser(in_queue, out_queue) for cpu in xrange(PROCESSES)]
    for writer in writers:
        writer.daemon = True
        writer.start()

    # write the results as they come in
    completed_processes = 0
    written_articles = 0
    while True:
        try:
            processed = out_queue.get()
            if processed is None:
                completed_processes +=1
                if completed_processes == len(writers):
                    raise KeyboardInterrupt
            else:
                try:
                    print 'writing processed article: {}'.format(processed.id + "-" + processed.title)
                except:
                    print 'writing processed article: {}'.format(processed.id)
                output_sqlite[processed.id] = processed
                written_articles += 1
        except (KeyboardInterrupt, Queue.Empty) as e:
            print "exiting..."
            output_sqlite.close()
            print "{} articles saved to {} in {}".format(written_articles, args.output_sqlite, time.time() - t0)
            exit()


if __name__ == "__main__":
    main()


