import os
import sys
import warnings
import operator
import subprocess

import numpy as np
from scipy.cluster.vq import kmeans2, whiten
from sklearn.metrics import accuracy_score

from Utility import LoadSavePickle
from MCMCLDA import MCMCLDA

class Main:
    def __init__(self, video_dir, modec_dir):
        self._video_dir = video_dir
        self._modec_dir = modec_dir

        self._stip_dir = 'stip-2.0-linux/'
        self._stip_result_dir = 'stip_result/'

        self._u = LoadSavePickle()

    def _print(self, message, newLine=True):
        if newLine:
            print message
        else:
            print message,
        sys.stdout.flush()

    # read a directory and list all files with avi as extension
    def _get_videos(self):
        self._print('Checking videos ... ', newLine=False)
        video = []
        if os.path.exists(self._video_dir):
            suffixes = set(['.AVI', '.avi'])
            for dirpath, dirnames, filenames in os.walk(self._video_dir):
                for f in filenames:
                    if os.path.splitext(f)[1] in suffixes:
                        video.append(os.path.join(dirpath, f))
        self._print('%d found' % len(video))
        return video

    # read a directory and list all files with txt as extension
    def _get_txt(self, directory):
        retval = []
        if os.path.exists(directory):
            suffixes = set(['.txt', '.TXT'])
            for dirpath, dirnames, filenames in os.walk(directory):
                for f in filenames:
                    if os.path.splitext(f)[1] in suffixes:
                        retval.append(os.path.join(dirpath, f))
        return retval

    # read an stip_file and returns a list of its content
    def _read_stip_file(self, stip_file):
        with open(stip_file) as f:
            content = f.readlines()
        content = content[2:] # ignore header
        retval = []
        retval.append([c.strip().split(' ') for c in content])
        return retval[0]

    # call kmeans, cluster observations, and return the centroids
    def _cluster(self, obs, nC=1000, nIter=100):
        # whiten (normalize) observations
        feats = np.array(obs).astype(np.float)
        feats = whiten(feats)

        # call k-means clustering
        _, lbl = kmeans2(data=feats, k=nC, iter=nIter, minit="points")
        return lbl

    # creating visual word codebook for each interest points on video
    def _visual_word(self, data, label):
        assert len(data) == len(label)

        for i in range(len(data)):
            data[i].update({"label": label[i]})

        data = sorted(data, key=lambda elem: int(elem["t"]))
        data = sorted(data, key=lambda elem: elem["fname"])

        vword = {}
        fname = data[0].get("fname")
        t = data[0].get("t")
        word = []

        for i in range(len(data)):
            if (data[i].get("t") != t) or (data[i].get("fname") != fname):
                wordlist = vword.get(fname)
                wordlist = wordlist.get("words") if (wordlist is not None) else []
                wordlist.append(word)

                classname = os.path.splitext(os.path.basename(fname))[0][:-4]
                insertdata = {"class": classname, "words": wordlist}
                vword.update({fname: insertdata})

                fname = data[i].get("fname")
                t = data[i].get("t")
                word = []

            word.append(data[i].get("label"))

        if len(word) > 0:
            wordlist = vword.get(fname)
            wordlist = wordlist.get("words") if (wordlist is not None) else []
            wordlist.append(word)

            classname = os.path.splitext(os.path.basename(fname))[0][:-4]
            insertdata = {"class": classname, "words": wordlist}
            vword.update({fname: insertdata})

        return vword

    # create document matrix for MCMCLDA input
    def _create_document_matrix(self, visual_word):
        docs = []
        matrix = []
        cm = set()
        for doc_name in visual_word.keys():
            docs.append((doc_name, visual_word[doc_name].get('class')))
            matrix.append(visual_word[doc_name].get('words')) # or we can use tt frames as 1 input t (bag of frames)
            cm.add(visual_word[doc_name].get('class'))
        cm = sorted([c for c in cm])
        doc = [(docs[i][0], cm.index(docs[i][1])) for i in xrange(len(docs))]
        return doc, matrix, cm

    # create confusion matrix
    def _confusion_matrix(self, Ytrue, Ypred, n):
        if (type(Ytrue) is not np.array): Ytrue = np.array(Ytrue)
        if (type(Ypred) is not np.array): Ypred = np.array(Ypred)
        return np.bincount(n * Ytrue + Ypred, minlength=n*n).reshape(n, n)

    # run Harris3D interest point detector or just calculating HOG and HOF from MODEC results
    def _run_stip(self, videos):
        if self._modec_dir is None:
            self._print('Running Harris3D interest point detector ...')
        else:
            self._print('Calculating HOG and HOF ...')
        nos = 0
        for v in videos:
            nos = nos + 1
            self._print('- Running on video %d of %d: %s' % (nos, len(videos), v))

            video_basename = os.path.splitext(os.path.basename(v))[0]
            outputfile = os.path.join(self._stip_result_dir, video_basename + '.txt')

            f = open('video-list.txt', 'w')
            f.write(video_basename)
            f.close()

            stip = os.path.join(self._stip_dir, 'bin/stipdet')
            if self._modec_dir is not None:
                retval = subprocess.call([stip, '-i', 'video-list.txt', '-vpath', self._video_dir, '-fpath', self._modec_dir, '-o', outputfile, '-mode', '1', '-vis', 'no'])
            else:
                retval = subprocess.call([stip, "-i", "video-list.txt", "-vpath", self._video_dir, "-o", outputfile, "-det", "harris3d", "-vis", "no"])

            if os.path.isfile('video-list.txt'): os.unlink('video-list.txt')
            sys.stdout.flush()

    # create visual word codebook from interest points features
    def _create_codebook(self):
        self._print('Creating visual word codebook ...')

        self._print('- Reading HOG and HOF features ...', newLine=False)
        # read all stip_file and vectorized its HOG + HOF
        files = self._get_txt(self._stip_result_dir)
        dt = []
        vec = []
        for f in files:
            V = self._read_stip_file(f)
            i = 0
            for v in V:
                i = i + 1
                dt.append({"fname": os.path.basename(f), "idx": i, "t": int(v[6])})
                vec.append(v[9:])
        self._print('%d found' % len(files))

        self._print('- Clustering ...')
        lbl = self._cluster(vec)
        self._u.dump('clusterinfo', { 'label': lbl, 'data': dt })

        self._print('- Populating visual words codebook ...')
        vis = self._visual_word(dt, lbl)
        self._u.dump('codebook', { 'visual_word': vis })

        return vis

    # randomly (stratified) split data for training and testing
    def _split_train_test(self, visual_word):
        self._print('Splitting training and testing data ...')
        docs = {}
        for doc_name in visual_word.keys():
            label = visual_word[doc_name].get('class')
            cnt = 0 if not docs.has_key(label) else docs.get(label)
            cnt += 1
            docs.update({label: cnt})
        # number of testing data
        ntest = {}
        for label in docs.keys():
            ntest.update({label: 3 if docs.get(label) >= 10 else 2})

        train = {}
        test = {}
        train_name = []
        test_name = []

        for doc_name in visual_word.keys():
            label = visual_word[doc_name].get('class')
            cnt = docs.get(label)
            if cnt <= ntest.get(label):
                test.update({doc_name: visual_word[doc_name]})
                test_name.append(doc_name)
            else:
                train.update({doc_name: visual_word[doc_name]})
                train_name.append(doc_name)
            cnt -= 1
            docs.update({label: cnt})

        # print 'Train = ', train_name
        # print 'Test = ', test_name
        return train, test

    # run all this
    def run(self, nIter=10):
        video = self._get_videos()
        assert (len(video) > 0)

        self._run_stip(video)
        visual_word = self._create_codebook()

        for it in xrange(nIter):
            self._print('-- Iteration %d --' % it)
            # randomly split training and testing data
            train_vis, test_vis = self._split_train_test(visual_word)

            # training
            docs, matrix, cm = self._create_document_matrix(train_vis)
            self._print('Training MCMCLDA ...')
            model = MCMCLDA(docs, matrix, cm)
            model.train()

            # testing
            docs, matrix, cm = self._create_document_matrix(test_vis)
            gt  = []
            est = []
            for i in xrange(len(docs)):
                self._print('Classifying %s (%s) as ' % (docs[i][0], cm[docs[i][1]]), newLine=False)
                gt.append(docs[i][1])
                score = []
                for c in cm:
                    docs[i] = (docs[i][0], cm.index(c))
                    model = MCMCLDA([docs[i]], [matrix[i]], cm)
                    score.append(model.test(est=c))
                label, max_score = max(enumerate(score), key=operator.itemgetter(1))
                est.append(label)
                self._print('%s (likelihood = %f)' % (cm[label], max_score))
            confmat = self._confusion_matrix(gt, est, len(cm))
            print confmat
            self._print('Accuracy = %f' % accuracy_score(gt, est))

if __name__ == '__main__':
    with warnings.catch_warnings():
        # ignore deprecated library warnings
        warnings.simplefilter("ignore")
        warnings.warn("deprecated", DeprecationWarning)

        if len(sys.argv) > 1:
            video_dir = sys.argv[1]
            modec_dir = sys.argv[2] if len(sys.argv) > 2 else None

            main = Main(video_dir, modec_dir)
            main.run()
        else:
            print 'Usage: python Main.py video_dir [ modec_result_dir ]'


