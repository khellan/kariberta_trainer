# KariBERTa Trainer

This approach is meant to be used to train a Norwegian RoBERTa language model.

## Getting started

These instructions are for pipenv. The Pipenv file should be simple enough to understand for anyone using conda, virtualenv or something else. I have not included Pipfile.lock since it would be different between a Windows or Mac development box and a Linux server.

## Training data

Download a suitable corpus of Norwegian training data.

I used these Norwegian Bokmål corpuses:
http://pcai056.informatik.uni-leipzig.de/downloads/corpora/nob_wikipedia_2014_1M.tar.gz
http://pcai056.informatik.uni-leipzig.de/downloads/corpora/nob-no_web_2017_1M.tar.gz
http://pcai056.informatik.uni-leipzig.de/downloads/corpora/nob_news_2013_1M.tar.gz
https://traces1.inria.fr/oscar/files/Compressed/no_dedup.txt.gz

I split the combined using split_sentences.py

## Development server
pipenv install

## Training server
I used Ubuntu. A similar approach should work for CentOS as well. 

'''
sudo apt install python3 python3-distutils python3-pip
pip3 install pipenv
git clone https://www.github.com/nvidia/apex
cd <your training directory>
pipenv install --python /usr/bin/python3
pipenv run pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ~/apex
'''

## Training

### Training the tokenizer

Before training, the tokenizer needs to be trained. There is no need for a GPU and my MacBook Air did this just as quickly as a beefy server.

'''
pipenv run python train_tokenizer.py
'''

### Training the language model

This may be run on a tiny set on a development box, but use a GPU server to train anything useful.

'''
cd <your training directory>
./train.sh
'''

I like to run that in tmux with two other tmux windows open for observation:
1. *top* to see CPU and memory use
2. *watch -d -c -n 1 nvidia-smi* to see GPU usage
