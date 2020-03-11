from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from sklearn.model_selection import train_test_split

def load(name):
    with open(name, 'r') as input_file:
        dataset = [line.strip() for line in input_file]
    return dataset

def save(subset, name):
    with open(name, 'w') as output_file:
        [output_file.write(line + '\n') for line in subset]

def split_subset(subset, name):
    test, train = train_test_split(subset, test_size=0.2, shuffle=True)
    trainset_name = f'data/test_{name}.txt'
    testset_name = f'data/train_{name}.txt'

    save(train, trainset_name)
    save(test, testset_name)

split_subset(f'data/nob_sentences.txt', 'sentences')
