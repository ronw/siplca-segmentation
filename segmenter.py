# Copyright (C) 2009-2010 Ron J. Weiss (ronw@nyu.edu)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Music structure segmentation using SI-PLCA

This module contains an implementation of the algorithm for music
structure segmentation described in [1].  It is based on
Shift-invariant Probabilistic Latent Component Analysis, a variant of
convolutive non-negative matrix factorization (NMF).  See plca.py for
more details.

Examples
--------
>>> import segmenter
>>> wavfile = '/path/to/come_together.wav'
>>> rank = 4  # rank corresponds to the number of segments
>>> win = 60  # win controls the length of each chroma pattern
>>> niter = 200  # number of iterations to perform
>>> np.random.seed(123)  # Make this reproduceable
>>> labels = segmenter.segment_wavfile(wavfile, win=win, rank=rank,
...                                    niter=niter, plotiter=10)
INFO:plca:Iteration 0: divergence = 10.065992
INFO:plca:Iteration 50: divergence = 9.468196
INFO:plca:Iteration 100: divergence = 9.421632
INFO:plca:Iteration 150: divergence = 9.409279
INFO:root:Iteration 199: final divergence = 9.404961
INFO:segmenter:Removing 2 segments shorter than 32 frames

.. image::come_together-segmentation.png

>>> print labels
0.0000 21.7480 segment0
21.7480 37.7640 segment1
37.7640 55.1000 segment0
55.1000 76.1440 segment1
76.1440 95.1640 segment0
95.1640 121.2360 segment1
121.2360 158.5360 segment2
158.5360 180.8520 segment1
180.8520 196.5840 segment0
196.5840 255.8160 segment3

See Also
--------
segmenter.extract_features : Beat-synchronous chroma feature extraction
segmenter.segment_song : Performs segmentation
segmenter.evaluate_segmentation : Evaluate frame-wise segmentation
segmenter.convert_labels_to_segments : Generate HTK formatted list of segments
                                       from frame-wise labels
plca.SIPLCA : Implementation of Shift-invariant PLCA

References
----------
 [1] R. J. Weiss and J. P. Bello. "Identifying Repeated Patterns in
     Music Using Sparse Convolutive Non-Negative Matrix
     Factorization". In Proc. International Conference on Music
     Information Retrieval (ISMIR), 2010.

Copyright (C) 2009-2010 Ron J. Weiss <ronw@nyu.edu>

LICENSE: This module is licensed under the GNU GPL. See COPYING for details.
"""

import glob
import logging
import optparse
import os
import sys

import numpy as np
import scipy as sp
import scipy.io

import plca

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('segmenter')

try:
    from mlabwrap import mlab
    mlab.addpath('coversongs')
except:
    logger.warning('Unable to import mlab module.  Feature extraction '
                   'and evaluation will not work.')


def extract_features(wavfilename, fctr=400, fsd=1.0, type=1):
    """Computes beat-synchronous chroma features from the given wave file

    Calls Dan Ellis' chrombeatftrs Matlab function.
    """
    x,fs = mlab.wavread(wavfilename, nout=2)
    feats,beats = mlab.chrombeatftrs(x, fs, fctr, fsd, type, nout=2)
    return feats, beats.flatten()

def segment_song(seq, win, nrep=1, minsegments=3, maxlowen=10, maxretries=5,
                 uninformativeWinit=False, uninformativeHinit=True, 
                 normalize_frames=True, **kwargs):
    """Segment the given feature sequence using SI-PLCA

    Parameters
    ----------
    seq : array, shape (F, T)
        Feature sequence to segment.
    win : int
        Length of patterns in frames.
    nrep : int
        Number of times to repeat the analysis.  The repetition with
        the lowest reconstrucion error is returned.  Defaults to 1.
    minsegments : int
        Minimum number of segments in the output.  The analysis is
        repeated until the output contains at least `minsegments`
        segments is or `maxretries` is reached.  Defaults to 3.
    maxlowen : int
        Maximum number of low energy frames in the SIPLCA
        reconstruction.  The analysis is repeated if it contains too
        many gaps.  Defaults to 10.
    maxretries : int
        Maximum number of retries to perform if `minsegments` or
       `maxlowen` are not satisfied.  Defaults to 5.
    uninformativeWinit : boolean
        If True, `W` is initialized to have a flat distribution.
        Defaults to False.
    uninformativeHinit : boolean
        If True, `H` is initialized to have a flat distribution.
        Defaults to True.
    normalize_frames : boolean
        If True, normalizes each frame of `seq` so that the maximum
        value is 1.  Defaults to True.
    kwargs : dict
        Keyword arguments passed to plca.SIPLCA.analyze.  See
        plca.SIPLCA for more details.

    Returns
    -------
    labels : array, length `T`
        Segment label for each frame of `seq`.
    W : array, shape (`F`, `rank`, `win`)
        Set of `F` x `win` shift-invariant basis functions found in `seq`.
    Z : array, length `rank`
        Set of mixing weights for each basis.
    H : array, shape (`rank`, `T`)
        Activations of each basis in time.
    segfun : array, shape (`rank`, `T`)
        Raw segmentation function used to generate segment labels from
        SI-PLCA decomposition.  Corresponds to $\ell_k(t)$ in [1].
    norm : float
        Normalization constant to make `seq` sum to 1.

    Notes
    -----
    The experimental results reported in [1] were found using the
    default values for all keyword arguments while varying kwargs.

    """
    seq = seq.copy()
    if normalize_frames:
        seq /= seq.max(0) + np.finfo(float).eps
    
    if 'alphaWcutoff' in kwargs and 'alphaWslope' in kwargs:
        kwargs['alphaW'] = create_sparse_W_prior((seq.shape[0], win),
                                                 kwargs['alphaWcutoff'],
                                                 kwargs['alphaWslope'])
        del kwargs['alphaWcutoff']
        del kwargs['alphaWslope']

    rank = kwargs['rank']
    F, T = seq.shape
    if uninformativeWinit:
        kwargs['initW'] = np.ones((F, rank, win)) / F*win
    if uninformativeHinit:
        kwargs['initH'] = np.ones((rank, T)) / T
        
    outputs = []
    for n in xrange(nrep):
        outputs.append(plca.SIPLCA.analyze(seq, win=win, **kwargs))
    div = [x[-1] for x in outputs]
    W, Z, H, norm, recon, div = outputs[np.argmin(div)]

    # Need to rerun segmentation if there are too few segments or
    # if there are too many gaps in recon (i.e. H)
    lowen = seq.shape[0] * np.finfo(float).eps
    nlowen_seq = np.sum(seq.sum(0) <= lowen)
    if nlowen_seq > maxlowen:
        maxlowen = nlowen_seq
    nlowen_recon = np.sum(recon.sum(0) <= lowen)
    nretries = maxretries
    while (len(Z) < minsegments or nlowen_recon > maxlowen) and nretries > 0:
        nretries -= 1
        logger.info('Redoing SIPLCA analysis (len(Z) = %d, number of '
                    'low energy frames = %d).', len(Z), nlowen_recon)
        outputs = []
        for n in xrange(nrep):
            outputs.append(plca.SIPLCA.analyze(seq, win=win, **kwargs))
        div = [x[-1] for x in outputs]
        W, Z, H, norm, recon, div = outputs[np.argmin(div)]
        nlowen_recon = np.sum(recon.sum(0) <= lowen)

    labels, segfun = nmf_analysis_to_segmentation(seq, win, W, Z, H, **kwargs)
    return labels, W, Z, H, segfun, norm

def create_sparse_W_prior(shape, cutoff, slope):
    """Constructs sparsity parameters for W (alphaW) to learn pattern length

    Follows equation (6) in the ISMIR paper referenced in this
    module's docstring.
    """

    # W.shape is (ndim, nseg, nwin)
    prior = np.zeros(shape[-1])
    prior[cutoff:] = prior[0] + slope * np.arange(shape[-1] - cutoff)

    alphaW = np.zeros((shape[0], 1, shape[-1]))
    alphaW[:,:] = prior
    return alphaW
    
def nmf_analysis_to_segmentation(seq, win, W, Z, H, min_segment_length=32,
                                 use_Z_for_segmentation=True, **ignored_kwargs):
    if not use_Z_for_segmentation:
        Z = np.ones(Z.shape)

    segfun = []
    for n, (w,z,h) in enumerate(zip(np.transpose(W, (1, 0, 2)), Z, H)):
        reconz = plca.SIPLCA.reconstruct(w, z, h)
        score = np.sum(reconz, 0) 

        # Smooth it out
        score = np.convolve(score, np.ones(min_segment_length), 'same')
        # kernel_size = min_segment_length
        # if kernel_size % 2 == 0:
        #     kernel_size += 1
        # score = sp.signal.medfilt(score, kernel_size)
        segfun.append(score)

    # Combine correlated segment labels
    C = np.reshape([np.correlate(x, y, mode='full')[:2*win].max()
                    for x in segfun for y in segfun],
                   (len(segfun), len(segfun)))

    segfun = np.array(segfun)
    segfun /= segfun.max()

    labels = np.argmax(np.asarray(segfun), 0)
    remove_short_segments(labels, min_segment_length)

    return labels, segfun

def remove_short_segments(labels, min_segment_length):
    """Remove segments shorter than min_segment_length."""
    segment_borders = np.nonzero(np.diff(labels))[0]
    short_segments_idx = np.nonzero(np.diff(segment_borders)
                                    < min_segment_length)[0]
    logger.info('Removing %d segments shorter than %d frames',
                len(short_segments_idx), min_segment_length)
    # Remove all adjacent short_segments.
    segment_borders[short_segments_idx]

    for idx in short_segments_idx:
        start = segment_borders[idx]
        try:
            end = segment_borders[idx + 1] + 1
        except IndexError:
            end = len(labels)

        try:
            label = labels[start - 1]
        except IndexError:
            label = labels[end]

        labels[start:end] = label

def evaluate_segmentation(labels, gtlabels, Z):
    """Calls Matlab to evaluate the given segmentation labels

    labels and gtlabels are arrays containing a numerical label for
    each frame of the sound (as returned by segment_song).

    Returns a dictionary containing name-value pairs of the form
    'metric name': value.
    """

    # Matlab is really picky about the shape of these vectors.  Make
    # sure labels is a row vector.
    nlabels = max(labels.shape)
    if labels.ndim == 1:
        labels = labels[np.newaxis,:]
    elif labels.shape[0] == nlabels:
        labels = labels.T

    perf = {}
    perf['pfm'], perf['ppr'], perf['prr'] = mlab.eval_segmentation_pairwise(
        labels, gtlabels, nout=3)
    perf['So'], perf['Su'] = mlab.eval_segmentation_entropy(labels, gtlabels,
                                                            nout=2)

    perf['nlabels'] = len(np.unique(labels))
    perf['effrank'] = len(Z)
    perf['nsegments'] = np.sum(np.diff(labels) != 0) + 1

    for k,v in perf.iteritems():
        perf[k] = float(v)

    return perf

def compute_effective_pattern_length(w):
    wsum = w.sum(0)
    # Find all taus in w that contain significant probability mass.
    nonzero_idx, = np.nonzero(wsum > wsum.min())
    winlen = nonzero_idx[-1] - nonzero_idx[0] + 1
    return winlen

def convert_labels_to_segments(labels, frametimes):
    """Covert frame-wise segmentation labels to a list of segments in HTK
    format"""
    
    # nonzero points in diff(labels) correspond to the final frame of
    # a segment (so just index into labels to find the segment label)
    boundaryidx = np.concatenate(([0], np.nonzero(np.diff(labels))[0],
                                  [len(labels) - 1]))
    boundarytimes = frametimes[boundaryidx]
    # Fix timing of first beat.
    boundarytimes[0] = 0;

    segstarttimes = boundarytimes[:-1]
    segendtimes = boundarytimes[1:]
    seglabels = labels[boundaryidx[1:]]

    segments = ['%.4f %.4f segment%d' % (start, end, label)
                for start,end,label in zip(segstarttimes,segendtimes,seglabels)]
    return '\n'.join(segments)
    
def segment_wavfile(wavfile, **kwargs):
    """Convenience function to compute segmentation of the given wavfile

    Keyword arguments are passed into segment_song.

    Returns a string containing list of segments in HTK label format.
    """
    features, beattimes = extract_features(wavfile)
    labels, W, Z, H, segfun, norm = segment_song(features, **kwargs)
    print sorted(labels)
    segments = convert_labels_to_segments(labels, beattimes)
    return segments
