#!/bin/sh

export PYTHONPATH=$PYTHONPATH:mlabwrap-1.1/build/lib.*

./segmenter.py -i "$1" -o "$2" \
    -niter 200 -rank 15 -win 60 -alphaZ -0.01 \
    -normalize_frames True -viterbi_segmenter True

