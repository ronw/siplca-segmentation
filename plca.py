# Copyright (C) 2009-2010 Ron J. Weiss (ronw@nyu.edu)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""plca: Probabilistic Latent Component Analysis

This module implements a number of variations of the PLCA algorithms
described in [2] and [3] with Dirichlet priors over the parmaters.

PLCA is a variant of non-negative matrix factorization which
decomposes a (2D) probabilitity distribution (arbitrarily normalized
non-negative matrix in the NMF case) V into the product of
distributions over the columns W = {w_k}, rows H = {h_k}, and mixing
weights Z = diag(z_k).  See [1-3] for more details.

References
----------
[1] R. J. Weiss and J. P. Bello. "Identifying Repeated Patterns in
    Music Using Sparse Convolutive Non-Negative Matrix
    Factorization". In Proc. International Conference on Music
    Information Retrieval (ISMIR), 2010.

[2] P. Smaragdis and B. Raj. "Shift-Invariant Probabilistic Latent
    Component Analysis". Technical Report TR2007-009, MERL, December
    2007.

[3] P. Smaragdis, B. Raj, and M. Shashanka. "Sparse and
    shift-invariant feature extraction from non-negative data".  In
    Proc. ICASSP, 2008.

Copyright (C) 2009-2010 Ron J. Weiss <ronw@nyu.edu>

LICENSE: This module is licensed under the GNU GPL. See COPYING for details.
"""

import functools
import logging

import numpy as np
import scipy as sp
import scipy.signal

import matplotlib.pyplot as plt
import plottools 

logger = logging.getLogger('plca')
EPS = np.finfo(np.float).eps
#EPS = 1e-100

def kldivergence(V, WZH):
    #return np.sum(V * np.log(V / WZH) - V + WZH)
    return np.sum(WZH - V * np.log(WZH))

def normalize(A, axis=None):
    Ashape = A.shape
    norm = A.sum(axis) + EPS
    if axis:
        nshape = list(Ashape)
        nshape[axis] = 1
        norm.shape = nshape
    return A / norm

def shift(a, shift, axis=None, circular=True):
    """Shift array along a given axis.

    If circular is False, zeros are inserted for elements rolled off
    the end of the array.

    See Also
    --------
    np.roll
    """
    aroll = np.roll(a, shift, axis)
    if not circular and shift != 0:
        if axis is None:
            arollflattened = aroll.flatten()
            if shift > 0:
                arollflattened[:shift] = 0
            elif shift < 0:
                arollflattened[shift:] = 0
            aroll = np.reshape(arollflattened, aroll.shape)
        else:
            index = [slice(None)] * a.ndim
            if shift > 0:
                index[axis] = slice(0, shift)
            elif shift < 0:
                index[axis] = slice(shift, None)
            aroll[index] = 0
    return aroll

def _fix_negative_values(x, fix=EPS):
    x[x < 0] = fix
    return x


class PLCA(object):
    """Probabilistic Latent Component Analysis

    Methods
    -------
    analyze
        Performs PLCA decomposition using the EM algorithm from [2].
    reconstruct(W, Z, H, norm=1.0)
        Reconstructs input matrix from the PLCA parameters W, Z, and H.
    plot(V, W, Z, H)
        Makes a pretty plot of V and the decomposition.

    initialize()
        Randomly initializes the parameters
    do_estep(W, Z, H)
        Performs the E-step of the EM parameter estimation algorithm.
    do_mstep()
        Performs the M-step of the EM parameter estimation algorithm.

    Notes
    -----
    You probably don't want to initialize this class directly.  Most
    interactions should be through the statis methods analyze,
    reconstruct, and plot.

    Subclasses that want to use a similar interface (e.g. SIPLCA)
    should also implement initialize, do_estep, and do_mstep.

    Examples
    --------
    Generate some random data:
    >>> F = 20
    >>> T = 100
    >>> rank = 3
    >>> means = [F/4.0, F/2.0, 3.0*F/4]
    >>> f = np.arange(F)
    >>> trueW = plca.normalize(np.array([np.exp(-(f - m)**2 / F)
    ...                                  for m in means]).T, 0)
    >>> trueZ = np.ones(rank) / rank
    >>> trueH = plca.normalize(np.random.rand(rank, T), 1)
    >>> V = plca.PLCA.reconstruct(trueW, trueZ, trueH)

    Perform the decomposition:
    >>> W, Z, H, norm, recon, divergence = plca.PLCA.analyze(V, rank=rank)
    INFO:plca:Iteration 0: divergence = 8.784769
    INFO:plca:Iteration 50: divergence = 8.450114
    INFO:plca:Iteration 99: final divergence = 8.449504
    
    Plot the parameters:
    >>> plt.figure(1)
    >>> plca.PLCA.plot(V, W, Z, H)
    >>> plt.figure(2)
    >>> plca.PLCA.plot(V, trueW, trueZ, trueH)

    W, Z, H and trueW, trueZ, trueH should be the same modulo
    permutations along the rank dimension.

    See Also
    --------
    SIPLCA : Shift-Invariant PLCA
    SIPLCA2 : 2D Shift-Invariant PLCA
    """
    def __init__(self, V, rank, alphaW=0, alphaZ=0, alphaH=0, minpruneiter=0,
                 **kwargs):
        """
        Parameters
        ----------
        V : array, shape (`F`, `T`)
            Matrix to analyze.
        rank : int
            Rank of the decomposition (i.e. number of columns of `W`
            and rows of `H`).
        alphaW, alphaZ, alphaH : float or appropriately shaped array
            Sparsity prior parameters for `W`, `Z`, and `H`.  Negative
            values lead to sparser distributions, positive values
            makes the distributions more uniform.  Defaults to 0 (no
            prior).

            **Note** that the prior is not parametrized in the
            standard way where the uninformative prior has alpha=1.
        """
        self.V = V.copy()
        self.rank = rank

        self.F, self.T = self.V.shape

        self.R = np.zeros((self.F, self.T, self.rank))

        self.alphaW = 1 + alphaW
        self.alphaZ = 1 + alphaZ
        self.alphaH = 1 + alphaH
        #print self.alphaW, self.alphaZ, self.alphaH

        self.minpruneiter = minpruneiter

    @classmethod
    def analyze(cls, V, rank, niter=100, convergence_thresh=1e-9,
                printiter=50, plotiter=None, plotfilename=None,
                initW=None, initZ=None, initH=None,
                updateW=True, updateZ=True, updateH=True, **kwargs):
        """Iteratively performs the PLCA decomposition using the EM algorithm

        Parameters
        ----------
        V : array, shape (`F`, `T`)
            Matrix to analyze.
        niter : int
            Number of iterations to perform.  Defaults to 100.
        convergence_thresh : float
        updateW, updateZ, updateH : boolean
            If False keeps the corresponding parameter fixed.
            Defaults to True.
        initW, initZ, initH : array
            Initial settings for `W`, `Z`, and `H`.  Unused by default.
        printiter : int
            Prints current divergence once every `printiter` iterations.
            Defaults to 50.
        plotiter : int or None
            If not None, the current decomposition is plotted once
            every `plotiter` iterations.  Defaults to None.
        kwargs : dict
            Arguments to pass into the class's constructor.

        Returns
        -------
        W : array, shape (`F`, `rank`)
            Set of `rank` bases found in `V`, i.e. P(f | z).
        Z : array, shape (`rank`)
            Mixing weights over basis vector activations, i.e. P(z).
        H : array, shape (`rank`, `T`)
            Activations of each basis in time, i.e. P(t | z).
        norm : float
            Normalization constant to make `V` sum to 1.
        recon : array
            Reconstruction of `V` using `W`, `Z`, and `H`
        divergence : float
        """
        norm = V.sum()
        V /= norm
    
        params = cls(V, rank, **kwargs)
        iW, iZ, iH = params.initialize()
    
        W = iW if initW is None else initW.copy()
        Z = iZ if initZ is None else initZ.copy()
        H = iH if initH is None else initH.copy()
    
        params.W = W
        params.Z = Z
        params.H = H
    
        olddiv = np.inf
        for n in xrange(niter):
            div, WZH = params.do_estep(W, Z, H)
            if n % printiter == 0:
                logger.info('Iteration %d: divergence = %f', n, div)
            if plotiter and n % plotiter == 0:
                params.plot(V, W, Z, H, n)
                if not plotfilename is None:
                    plt.savefig('%s_%04d.png' % (plotfilename, n))
            if div > olddiv:
                logger.debug('Warning: Divergence increased from %f to %f at '
                             'iteration %d!', olddiv, div, n)
                #import pdb; pdb.set_trace()
            elif n > 0 and olddiv - div < convergence_thresh:
                logger.info('Converged at iteration %d', n)
                break
            olddiv = div
    
            nW, nZ, nH = params.do_mstep(n)
    
            if updateW:  W = nW
            if updateZ:  Z = nZ
            if updateH:  H = nH
    
            params.W = W
            params.Z = Z
            params.H = H

        if plotiter:
            params.plot(V, W, Z, H, n)
            if not plotfilename is None:
                plt.savefig('%s_%04d.png' % (plotfilename, n))
        logging.info('Iteration %d: final divergence = %f', n, div)
        recon = norm * WZH
        return W, Z, H, norm, recon, div

    @staticmethod
    def reconstruct(W, Z, H, norm=1.0):
        """Computes the approximation to V using W, Z, and H"""
        return norm * np.dot(W * Z, H)

    @classmethod
    def plot(cls, V, W, Z, H, curriter=-1):
        WZH = cls.reconstruct(W, Z, H)
        plottools.plotall([V, WZH], subplot=(3,1), align='xy', cmap=plt.cm.hot)
        plottools.plotall(9  * [None] + [W, Z, H], subplot=(4,3), clf=False,
                          align='', cmap=plt.cm.hot, colorbar=False)
        plt.draw()

    def initialize(self):
        """Initializes the parameters

        W and H are initialized randomly.  Z is initialized to have a
        uniform distribution.
        """
        W = normalize(np.random.rand(self.F, self.rank), 0)
        Z = np.ones(self.rank) / self.rank
        H = normalize(np.random.rand(self.rank, self.T), 1)
        return W, Z, H

    def do_estep(self, W, Z, H):
        """Performs the E-step of the EM parameter estimation algorithm.
        
        Computes the posterior distribution over the hidden variables.
        """
        WZH = self.reconstruct(W, Z, H)
        kldiv = kldivergence(self.V, WZH)
        #loglik = (np.sum(self.V * np.log(WZH))
        #          + (self.alphaW-1)*np.sum(np.log(W))
        #          + (self.alphaZ-1)*np.sum(np.log(Z))
        #          + (self.alphaH-1)*np.sum(np.log(H)))

        for z in xrange(self.rank):
            self.R[:,:,z] = np.outer(W[:,z] * Z[z], H[z,:])
        self.R /= self.R.sum(2)[:,:,np.newaxis]
        
        return kldiv, WZH

    def do_mstep(self, curriter):
        """Performs the M-step of the EM parameter estimation algorithm.

        Computes updated estimates of W, Z, and H using the posterior
        distribution computer in the E-step.
        """
        VR = self.R * self.V[:,:,np.newaxis]
        Z = normalize(_fix_negative_values(VR.sum(1).sum(0) + self.alphaZ - 1))
        W = normalize(_fix_negative_values(VR.sum(1) + self.alphaW - 1), 0)
        H = normalize(_fix_negative_values(VR.sum(0).T + self.alphaH - 1), 1)
        return self._prune_undeeded_bases(W, Z, H, curriter)

    def _prune_undeeded_bases(self, W, Z, H, curriter):
        """Discards bases which do not contribute to the decomposition"""
        threshold = 10 * EPS
        zidx = np.argwhere(Z > threshold).flatten()
        if len(zidx) < self.rank and curriter >= self.minpruneiter:
            logger.info('Rank decreased from %d to %d during iteration %d',
                        self.rank, len(zidx), curriter)
            self.rank = len(zidx)
            Z = Z[zidx]
            W = W[:,zidx]
            H = H[zidx,:]
            self.R = self.R[:,:,zidx]
        return W, Z, H


class SIPLCA(PLCA):
    """Sparse Shift-Invariant Probabilistic Latent Component Analysis

    Decompose V into \sum_k W_k * z_k h_k^T where * denotes
    convolution.  Each basis W_k is a matrix.  Therefore, unlike PLCA,
    `W` has shape (`F`, `win`, `rank`). This is the model used in [1].

    See Also
    --------
    PLCA : Probabilistic Latent Component Analysis
    SIPLCA2 : 2D SIPLCA
    """
    def __init__(self, V, rank, win=1, circular=False, **kwargs):
        """
        Parameters
        ----------
        V : array, shape (`F`, `T`)
            Matrix to analyze.
        rank : int
            Rank of the decomposition (i.e. number of columns of `W`
            and rows of `H`).
        win : int
            Length of each of the convolutive bases.  Defaults to 1,
            i.e. the model is identical to PLCA.
        circular : boolean
            If True, data shifted past `T` will wrap around to
            0. Defaults to False.
        alphaW, alphaZ, alphaH : float or appropriately shaped array
            Sparsity prior parameters for `W`, `Z`, and `H`.  Negative
            values lead to sparser distributions, positive values
            makes the distributions more uniform.  Defaults to 0 (no
            prior).

            **Note** that the prior is not parametrized in the
            standard way where the uninformative prior has alpha=1.
        """
        PLCA.__init__(self, V, rank, **kwargs)

        self.win = win
        self.circular = circular

        self.R = np.zeros((self.F, self.T, self.rank, self.win))

    @staticmethod
    def reconstruct(W, Z, H, norm=1.0, circular=False):
        if W.ndim == 2:
            W = W[:,np.newaxis,:]
        if H.ndim == 1:
            H = H[np.newaxis,:]
        F, rank, win = W.shape
        rank, T = H.shape
    
        WZH = np.zeros((F, T))
        for tau in xrange(win):
            WZH += np.dot(W[:,:,tau] * Z, shift(H, tau, 1, circular))
        return norm * WZH

    @classmethod
    def plot(cls, V, W, Z, H, curriter=-1):
        rank = len(Z)
        nrows = rank + 2
        WZH = cls.reconstruct(W, Z, H)
        plottools.plotall([V, WZH] + [cls.reconstruct(W[:,z,:], Z[z], H[z,:])
                                      for z in xrange(len(Z))], 
                          title=['V (Iteration %d)' % curriter,
                                 'Reconstruction'] +
                          ['Basis %d reconstruction' % x
                           for x in xrange(len(Z))],
                          colorbar=False, grid=False, cmap=plt.cm.hot,
                          subplot=(nrows, 2), order='c', align='xy')
        plottools.plotall([None] + [Z], subplot=(nrows, 2), clf=False,
                          plotfun=lambda x: plt.bar(np.arange(len(x)) - 0.4, x),
                          xticks=[[], range(rank)], grid=False,
                          colorbar=False, title='Z')

        plots = [None] * (3*nrows + 2)
        titles = plots + ['W%d' % x for x in range(rank)]
        wxticks = [[]] * (3*nrows + rank + 1) + [range(0, W.shape[2], 10)]
        plots.extend(W.transpose((1, 0, 2)))
        plottools.plotall(plots, subplot=(nrows, 6), clf=False, order='c',
                          align='xy', cmap=plt.cm.hot, colorbar=False, 
                          ylabel=r'$\parallel$', grid=False,
                          title=titles, yticks=[[]], xticks=wxticks)
        
        plots = [None] * (2*nrows + 2)
        titles=plots + ['H%d' % x for x in range(rank)]
        plots.extend(H)
        plottools.plotall(plots, subplot=(nrows, 3), order='c', align='xy',
                          grid=False, clf=False, title=titles, yticks=[[]],
                          colorbar=False, cmap=plt.cm.hot, ylabel=r'$*$',
                          xticks=[[]]*(3*nrows-1) + [range(0, V.shape[1], 100)])
        plt.draw()

    def initialize(self):
        W, Z, H = super(SIPLCA, self).initialize()
        W = np.random.rand(self.F, self.rank, self.win)
        W /= W.sum(2).sum(0)[np.newaxis,:,np.newaxis]
        return W, Z, H

    def do_estep(self, W, Z, H):
        WZH = self.reconstruct(W, Z, H, circular=self.circular)
        kldiv = kldivergence(self.V, WZH)
        #loglik = (np.sum(self.V * np.log(WZH))
        #          + (self.alphaW-1)*np.sum(np.log(W))
        #          + (self.alphaZ-1)*np.sum(np.log(Z))
        #          + (self.alphaH-1)*np.sum(np.log(H)))

        for tau in xrange(self.win):
            Ht = shift(H, tau, 1, self.circular) * Z[:,np.newaxis]
            for z in xrange(self.rank):
                self.R[:,:,z,tau] = np.outer(W[:,z,tau], Ht[z,:])
        self.R /= self.R.sum(3).sum(2)[:,:,np.newaxis,np.newaxis]

        return kldiv, WZH

    def do_mstep(self, curriter):
        VR = self.R * (self.V[:,:,np.newaxis,np.newaxis] + EPS)

        Z = normalize(_fix_negative_values(VR.sum(3).sum(1).sum(0)
                                           + self.alphaZ - 1))

        W = _fix_negative_values(VR.sum(1) + self.alphaW - 1)
        W /= W.sum(2).sum(0)[np.newaxis,:,np.newaxis] + EPS

        H = np.zeros((self.rank, self.T))
        for tau in xrange(self.win):
            H += np.sum(shift(VR[:,:,:,tau], -tau, 1, self.circular), 0).T
        H = normalize(_fix_negative_values(H + self.alphaH - 1), 1)

        return self._prune_undeeded_bases(W, Z, H, curriter)


class SIPLCA2(SIPLCA):
    """Sparse 2D Shift-Invariant Probabilistic Latent Component Analysis
 
    Shift invariance is over both rows and columns of `V`.  Unlike
    PLCA and SIPLCA, the activations for each basis `H_k` describes
    when the k-th basis is active in time *and* at what vertical
    (frequency) offset.  Therefore, unlike PLCA and SIPLCA, `H` has
    shape (`rank`, `win[1]`, `T`).

    Note that this is not the same as the 2D-SIPLCA decomposition
    described in Smaragdis and Raj, 2007.  `W` has the same shape as
    in SIPLCA, regardless of `win[1]`.

    See Also
    --------
    PLCA : Probabilistic Latent Component Analysis
    SIPLCA : Shift-Invariant PLCA
    """
    def __init__(self, V, rank, win=1, circular=False, **kwargs):
        """
        Parameters
        ----------
        V : array, shape (`F`, `T`)
            Matrix to analyze.
        rank : int
            Rank of the decomposition (i.e. number of columns of `W`
            and rows of `H`).
        win : int or tuple of 2 ints
            `win[0]` is the length of the convolutive bases.  `win[1]`
            is maximum frequency shift.  Defaults to (1, 1).
        circular : boolean or tuple of 2 booleans
            If `circular[0]` (`circular[1]`) is True, data shifted
            horizontally (vertically) past `T` (`F`) will wrap around
            to 0.  Defaults to (False, False).
        alphaW, alphaZ, alphaH : float or appropriately shaped array
            Sparsity prior parameters for `W`, `Z`, and `H`.  Negative
            values lead to sparser distributions, positive values
            makes the distributions more uniform.  Defaults to 0 (no
            prior).

            **Note** that the prior is not parametrized in the
            standard way where the uninformative prior has alpha=1.
        """
        PLCA.__init__(self, V, rank, **kwargs)
        self.rank = rank

        try:
            self.winF, self.winT = win
        except:
            self.winF = self.winT = win
        # Needed for compatibility with SIPLCA.
        self.win = self.winT  

        try:
            self.circularF, self.circularT = circular
        except:
            self.circularF = self.circularT = circular

        self.R = np.zeros((self.F, self.T, self.rank, self.winF, self.winT))

    @staticmethod
    def reconstruct(W, Z, H, norm=1.0, circular=False):
        if W.ndim == 2:
            W = W[:,np.newaxis,:]
        if Z.ndim == 0:
            Z = Z[np.newaxis]
        if H.ndim == 2:
            H = H[np.newaxis,:,:]
        F, rank, winT = W.shape
        rank, winF, T = H.shape
    
        try:
            circularF, circularT = circular
        except:
            circularF = circularT = circular
    
        recon = 0
        for z in xrange(rank):
            recon += sp.signal.fftconvolve(W[:,z,:] * Z[z], H[z,:,:])
    
        WZH = recon[:F,:T]
        if circularF:
            WZH[:winF-1,:] += recon[F:,:T]
        if circularT:
            WZH[:,:winT-1] += recon[:F,T:]
        if circularF and circularT:
            WZH[:winF-1,:winT-1] += recon[F:,T:]
    
        return norm * WZH

    def initialize(self):
        W, Z, H = super(SIPLCA2, self).initialize()
        W = np.random.rand(self.F, self.rank, self.winT)
        W /= W.sum(2).sum(0)[np.newaxis,:,np.newaxis]
        H = np.random.rand(self.rank, self.winF, self.T)
        H /= H.sum(2).sum(1)[:,np.newaxis,np.newaxis]
        return W, Z, H

    def do_estep(self, W, Z, H):
        WZH = SIPLCA2.reconstruct(W, Z, H,
                                  circular=[self.circularF, self.circularT])
        kldiv = kldivergence(self.V, WZH)
        #loglik = (np.sum(self.V * np.log(WZH))
        #          + (self.alphaW-1)*np.sum(np.log(W))
        #          + (self.alphaZ-1)*np.sum(np.log(Z))
        #          + (self.alphaH-1)*np.sum(np.log(H)))

        WZ = W * Z[np.newaxis,:,np.newaxis]
        for tauF in xrange(self.winF):
            Wshift = shift(WZ, tauF, 0, self.circularF)
            for tauT in xrange(self.winT):
                Hshift = shift(H[:,tauF,:], tauT, 1, self.circularT)
                self.R[:,:,:,tauF,tauT] = (
                    Wshift[:,:,tauT][:,:,np.newaxis]
                    * Hshift[np.newaxis,:,:]).transpose((0,2,1))
        self.R /= (EPS +
            self.R.sum(4).sum(3).sum(2)[:,:,np.newaxis,np.newaxis,np.newaxis])

        return kldiv, WZH

    def do_mstep(self, curriter):
        VR = self.R * self.V[:,:,np.newaxis,np.newaxis,np.newaxis]
        Z = normalize(_fix_negative_values(
            VR.sum(4).sum(3).sum(1).sum(0) + self.alphaZ - 1))

        W = np.zeros((self.F, self.rank, self.winT))
        for tauF in xrange(self.winF):
            W += shift(VR[:,:,:,tauF,:], -tauF, 0, self.circularF).sum(1)
        W = _fix_negative_values(W + self.alphaW - 1)
        W /= W.sum(2).sum(0)[np.newaxis,:,np.newaxis]

        H = np.zeros((self.rank, self.winF, self.T))
        for tauT in xrange(self.winT):
            H += shift(VR[:,:,:,:,tauT], -tauT, 1,
                       self.circularT).sum(0).transpose((1,2,0))
        H = _fix_negative_values(H + self.alphaH - 1)
        H /= H.sum(2).sum(1)[:,np.newaxis,np.newaxis]

        return self._prune_undeeded_bases(W, Z, H, curriter)
