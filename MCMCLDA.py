import numpy as np
import scipy.special as scp
import sklearn.hmm as hmm

from Utility import LoadSavePickle

class MCMCLDA:
    def __init__(self, docs, matrix, cm, n_topics=100, n_poses=100, vocab_size=1000, alpha=5, beta=0.01):
        self.n_topics = n_topics        # N_z
        self.n_poses = n_poses          # N_y
        self.vocab_size = vocab_size    # N_x

        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = np.ones(len(cm))

        self.alpha_est = self.alpha
        self.beta_est = self.beta
        self.gamma_est = self.gamma

        self.docs = docs
        self.matrix = np.array(matrix)
        self.cm = cm
        self.n_docs = len(docs)

        self.samples = None
        self.n_samples = 0

        # HMM model
        self.model = None

        # number of times that document m and pose y co-occur
        self.nmy = np.zeros((self.n_docs, self.n_poses))
        # number of times that pose y and word w co-occur
        self.nyw = np.zeros((self.n_poses, self.vocab_size))
        # number of times that pose y and motion pattern z co-occur
        self.nyz = np.zeros((self.n_poses, self.n_topics))
        # number of times that motion pattern z_t is followed by mottion pattern z_{t-1)
        self.nzt_min = np.zeros((self.n_topics, self.n_topics, len(self.cm)))
        # number of times that motion pattern z_{t+1} is followed by motion pattern z_t
        self.nzt_pls = np.zeros((self.n_topics, self.n_topics, len(self.cm)))

        self.nm = np.zeros(self.n_docs)
        self.nw = np.zeros(self.vocab_size)
        self.ny = np.zeros(self.n_poses)
        self.nz = np.zeros(self.n_topics)
        self.nz_min = np.zeros((self.n_topics, len(self.cm)))
        self.nz_pls = np.zeros((self.n_topics, len(self.cm)))

        self.poses = {}
        self.topics = {}
        self.labels = np.zeros((self.n_docs,), dtype=np.int)
        self.true_labels = [d[1] for d in self.docs]
        self.labels = np.copy(self.true_labels)

        for m in xrange(self.n_docs):
            # loop through document's time
            for t in xrange(len(self.matrix[m])):
                # randomize motion segment / topic
                z = np.random.randint(self.n_topics)

                self.topics[(m, t)] = z
                self.nz[z] += 1

                if (t > 0):
                    self.nzt_min[(z, self.topics[(m, t-1)], self.labels[m])] += 1
                    self.nzt_pls[(self.topics[(m, t-1)], z, self.labels[m])] += 1
                    self.nz_min[(self.topics[(m, t-1)], self.labels[m])] += 1
                    self.nz_pls[(z, self.labels[m])] += 1

                for i, w in enumerate(self.matrix[m][t][:]):
                    # randomize pose
                    y = np.random.randint(self.n_poses)
                    self.nw[w] += 1
                    self.nmy[m, y] += 1
                    self.nm[m] += 1
                    self.nyw[y, w] += 1
                    self.ny[y] += 1
                    self.nyz[y, z] += 1
                    self.poses[(m, t, i)] = y

    def _save_parameter(self, filename="params"):
        params = {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'model': self.model
        }
        u = LoadSavePickle()
        u.dump(filename, params)

    def _load_parameter(self, paramtype='params'):
        if (paramtype == 'malgireddy'):
            self.alpha = 0.34
            self.beta = 0.001
            self.gamma = np.array([0.04, 0.05, 0.16, 0.22, 0.006, 0.04, 0.13, 0.05, 0.14, 0.45])
            self.model = None
        else:
            u = LoadSavePickle()
            params = u.load(paramtype)
            self.alpha = params.get('alpha')
            self.beta = params.get('beta')
            self.gamma = params.get('gamma')
            self.model = params.get('model')

    def _topic_likelihood(self, m, t):
        z = self.topics[(m, t)]

        yt = np.zeros(self.n_poses)
        for i, w in enumerate(self.matrix[m][t][:]):
            y = self.poses[(m, t, i)]
            yt[y] += 1

        left = 1
        for y in xrange(self.n_poses):
            if (yt[y] > 0):
                u = scp.gamma(self.nyz[y, :] + self.alpha)
                self.nyz[y, z] -= yt[y]
                v = scp.gamma(self.nyz[y, :] + self.alpha)
                self.nyz[y, z] += yt[y]
                left *= u / v

        _nyz = 0
        _nyzt = 0
        for y in xrange(self.n_poses):
            if (yt[y] > 0):
                _nyz += self.nyz[y, :]
                self.nyz[y, z] -= yt[y]
                _nyzt += self.nyz[y, :]
                self.nyz[y, z] += yt[y]
        right = scp.gamma(_nyzt + self.n_poses * self.alpha) / scp.gamma(_nyz + self.n_poses * self.alpha)

        p_y = left * right
        p_y /= np.sum(p_y) if np.sum(p_y) > 0 else 1
        return p_y

    def _topic_prior(self, c, m, t):
        left = np.zeros(self.n_topics)
        right = np.zeros(self.n_topics)

        if (t > 0) and (t < len(self.matrix[m])-1):
            zt = self.topics[(m, t)]
            z_min = self.topics[(m, t-1)]
            z_pls = self.topics[(m, t+1)]

            if (self.nz_min[(z_min, c)] > 0): self.nz_min[(z_min, c)] -= 1 ##
            if (self.nz_pls[(z_pls, c)] > 0): self.nz_pls[(z_pls, c)] -= 1 ##

            for z in xrange(self.n_topics):
                aflag = False
                if self.nzt_min[(z, z_min, c)] > 0: ##
                    self.nzt_min[(z, z_min, c)] -= 1
                    aflag = True

                bflag = False
                if self.nzt_pls[(z, z_pls, c)] > 0: ##
                    self.nzt_pls[(z, z_pls, c)] -= 1
                    bflag = True

                if (z <> z_min):
                    # 1st condition
                    left[z] = (self.nzt_min[(z, z_min, c)] + self.gamma[c]) / (self.nz_min[(z_min, c)] + self.n_topics * self.gamma[c])
                    right[z] = (self.nzt_pls[(z, z_pls, c)] + self.gamma[c]) / (self.nz_pls[(z_pls, c)] + self.n_topics * self.gamma[c])
                elif (z == z_min):
                    if (z == z_pls):
                        # 2nd condition
                        left[z] = (self.nzt_min[(z, z_min, c)] + 1 + self.gamma[c]) / (self.nz_min[(z_min, c)] + 1 + self.n_topics * self.gamma[c])
                        right[z] = (self.nzt_pls[(z, z_pls, c)] + self.gamma[c]) / (self.nz_pls[(z_pls, c)] + self.n_topics * self.gamma[c])
                    else:
                        # 3rd condition
                        left[z] = (self.nzt_min[(z, z_min, c)] + self.gamma[c]) / (self.nz_min[(z_min, c)] + self.n_topics * self.gamma[c])
                        right[z] = (self.nzt_pls[(z, z_pls, c)] + self.gamma[c]) / (self.nz_pls[(z_pls, c)] + 1 + self.n_topics * self.gamma[c])

                if aflag: self.nzt_min[(z, z_min, c)] += 1 ##
                if bflag: self.nzt_pls[(z, z_pls, c)] += 1 ##

            self.nz_min[(z_min, c)] += 1
            self.nz_pls[(z_pls, c)] += 1

        p_z = left * right
        p_z /= np.sum(p_z) if np.sum(p_z) > 0 else 1
        return p_z

    def _pose_distribution(self, z, w):
        left = (self.nyw[:, w] + self.beta) / (self.ny + self.beta * self.vocab_size)
        right = (self.nyz[:, z] + self.alpha)
        p_y = left * right
        p_y /= np.sum(p_y) if np.sum(p_y) > 0 else 1 # normalize to obtain probabilities
        return p_y

    def _topic_distribution(self, c, m, z, t):
        left = self._topic_likelihood(m, t)
        right = self._topic_prior(c, m, t)
        p_z = left * right
        p_z /= np.sum(p_z) if np.sum(p_z) > 0 else 1 # normalize to obtain probabilities
        return p_z

    def _sample_index(self, p):
        return np.random.multinomial(1, p).argmax()

    def _run(self, maxiter=6000, burnin=5000, lag=100, verbose=False):
        assert (burnin < maxiter)
        assert (lag <= maxiter - burnin)
        assert (lag > 0)

        self.samples = []

        if maxiter > 1: print "Starting Gibbs sampling (maxiter=%d, burnin=%d, lag=%d)" % (maxiter, burnin, lag),

        for it in xrange(1, maxiter + 1):
            if maxiter > 1: print "\nIteration %d of %d" % (it, maxiter),

            for m in xrange(self.n_docs):
                c = self.labels[m]
                if verbose: print "\nDocument %d (est=%d, tru=%d)" % (m, c, self.true_labels[m])

                for t in xrange(len(self.matrix[m])):
                    z = self.topics[(m, t)]
                    if verbose: print "\tt = %d, z_old = %d" % (t, z)

                    # update pose y
                    for i, w in enumerate(self.matrix[m][t][:]):
                        y = self.poses[(m, t, i)]
                        self.nmy[m, y] -= 1
                        self.nm[m] -= 1
                        self.nyw[y, w] -= 1
                        self.ny[y] -= 1
                        if (self.nyz[y, z] > 0): self.nyz[y, z] -= 1

                        p_y = self._pose_distribution(z, w)
                        y = self._sample_index(p_y)

                        self.nmy[m, y] += 1
                        self.nm[m] += 1
                        self.nyw[y, w] += 1
                        self.ny[y] += 1
                        self.nyz[y, z] += 1
                        self.poses[(m, t, i)] = y

                    # update topic (motion segment) z
                    if (self.nz[z] > 0): self.nz[z] -= 1 ##

                    z_old = z
                    p_z = self._topic_distribution(c, m, z, t)
                    z = self._sample_index(p_z)

                    if (t > 0): ##
                        if (self.nzt_min[(z_old, self.topics[(m, t-1)], c)] > 0): self.nzt_min[(z_old, self.topics[(m, t-1)], c)] -= 1
                        if (self.nzt_pls[(self.topics[(m, t-1)], z_old, c)] > 0): self.nzt_pls[(self.topics[(m, t-1)], z_old, c)] -= 1
                        if (self.nz_min[(z_old, c)] > 0): self.nz_min[(z_old, c)] -= 1
                        if (self.nz_pls[(z_old, c)] > 0): self.nz_pls[(z_old, c)] -= 1

                        self.nzt_min[(z, self.topics[(m, t-1)], c)] += 1
                        self.nzt_pls[(self.topics[(m, t-1)], z, c)] += 1
                        self.nz_min[(z, c)] += 1
                        self.nz_pls[(z, c)] += 1

                    for i, w in enumerate(self.matrix[m][t][:]):
                        if (self.nyz[y, z_old] > 0): self.nyz[y, z_old] -= 1 ##
                        self.nyz[y, z] += 1

                    self.nz[z] += 1
                    self.topics[(m, t)] = z

                    if verbose: print "\tz_new =", self.topics[(m, t)]

            # sampling after burn-in iteration and lags
            if (it > burnin) and ((it-burnin) % lag == 0):
                if verbose: print "Sampled on iteration #%d" % it
                # hyperparameters estimation
                self._est_params()
                # sampling z for Viterbi
                sample = []
                for m in xrange(self.n_docs):
                    obs = []
                    for t in xrange(len(self.matrix[m])):
                        obs.append(self.topics[(m, t)])
                    sample.append({'obs': obs, 'class': self.cm[self.true_labels[m]]})
                self.samples.append({'iter': it, 'sample': sample})

        if verbose: print "Sampling on %d iterations" % (len(self.samples))

    def _est_params(self):
        self.alpha_est = (self.alpha_est * (np.sum(scp.psi(self.nyz[:, :] + self.alpha_est)) - self.n_poses * self.n_topics * scp.psi(self.alpha_est))) / (self.n_topics * (np.sum(scp.psi(self.ny[:] - self.n_topics * self.alpha_est)) - self.n_poses * scp.psi(self.n_topics * self.alpha_est)))
        self.beta_est = (self.beta_est * (np.sum(scp.psi(self.nyw[:, :] + self.beta_est)) - self.vocab_size * self.n_poses * scp.psi(self.beta_est))) / (self.n_poses * (np.sum(scp.psi(self.nw[:] + self.n_poses * self.beta_est)) - self.vocab_size * scp.psi(self.n_poses * self.beta_est)))

        for c in xrange(len(self.cm)):
            nmc = 0
            for m in xrange(self.n_docs):
                if self.labels[m] == c:
                    nmc += 1
            self.gamma_est[c] = (self.gamma_est[c] * (scp.psi(nmc + self.gamma_est[c])) - self.n_docs * scp.psi(self.gamma_est[c])) / ((scp.psi(self.n_docs + np.sum(self.gamma_est))) - self.n_docs * scp.psi(np.sum(self.gamma_est)))

    def _get_sample(self):
        # get samples per activity class
        obs = {}
        for c in self.cm:
            obs.update({c: []})
        for s in self.samples:
            for t in s['sample']:
                obs[t['class']].append(t['obs'])

        # padding on observations features length
        sz = max([len(self.matrix[m]) for m in xrange(self.n_docs)])
        for c in self.cm:
            for i in xrange(len(obs[c])):
                while (len(obs[c][i]) < sz):
                    obs[c][i].append(-1)

        return obs

    def _classify(self, X, est):
        assert self.model is not None

        # padding
        sz = self.model[est].covars_[0].shape[1]
        for x in X:
            while (len(x) < sz):
                x.append(-1)

        score = self.model[est].score(X)
        return score

    def train(self):
        # Gibbs sampling on training data
        #self._load_parameter(paramtype='malgireddy')
        self._run(verbose=False)

        # estimate MCMCLDA parameters from samples
        self.alpha = self.alpha_est
        self.beta = self.beta_est
        self.gamma = self.gamma_est

        print
        print 'Alpha =', self.alpha
        print 'Beta =', self.beta
        print 'Gamma =', self.gamma

        # learning HMM
        print "Learning HMM with Viterbi algorithm ..."
        obs = self._get_sample()
        self.model = {}
        for c in self.cm:
            print "-", c, np.array(obs[c]).shape
            X = np.array(obs[c])
            self.model.update({c: hmm.GaussianHMM(n_components=len(self.cm))})
            self.model[c].fit([X])

        # save parameters
        print "Saving parameters ..."
        self._save_parameter()

        print "Training done."

    def test(self, est):
        self._load_parameter()
        self._run(maxiter=1, burnin=0, lag=1)
        obs = self._get_sample()
        score = self._classify(obs[est], est)
        return score
