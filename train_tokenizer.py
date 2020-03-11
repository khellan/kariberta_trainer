from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

#paths = ['data/train_sentences.txt']
paths = ['data/train/t5.txt']
#paths = [str(x) for x in Path("./data/").glob("train_subset_*.txt")]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, vocab_size=25_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save("models", "KariBERTa-small")
