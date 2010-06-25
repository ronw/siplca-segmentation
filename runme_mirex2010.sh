#!/bin/sh

./segmenter.py -i "$1" -o "$2" \
    -niter 200 -rank 15 -win 70 -alphaZ -0.015 \
    -normalize_frames True -viterbi_segmenter True

