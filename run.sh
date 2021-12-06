#!/bin/bash

PYTHONHOME="/vol/research/xmodal_dl/txtreid-env/bin"
HOME="/vol/research/xmodal_dl/TextReID"

echo $HOME
echo 'args:' $@

$PYTHONHOME/python $HOME/train_net.py --root $HOME $@
