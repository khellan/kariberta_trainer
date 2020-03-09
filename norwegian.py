from pathlib import Path
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tokenizers.processors import BertProcessing


class NorwegianDataset(Dataset):
    def read_example_file(self, src_file, i, number_of_files):
        print(f"🔥{i} / {number_of_files}: {src_file}")
        lines = src_file.open(encoding="utf-8").read().splitlines()
        examples = [x.ids for x in self.tokenizer.encode_batch(lines)]
        return examples

    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        # or use the RobertaTokenizer from `transformers` directly.

        # src_file = Path(file_path)
        # print("🔥", src_file)
        # lines = src_file.open(encoding="utf-8").read().splitlines()
        # self.examples = [x.ids for x in tokenizer.encode_batch(lines)]
        path = Path(file_path)
        print("🔥", path)
        filenames = path.glob("*")
        number_of_files = len(list(filenames))
        filenames = path.glob("*")
        self.examples = []
        for i, src_file in enumerate(filenames):
            self.examples += self.read_example_file(src_file, i, number_of_files)
        print("All files read")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


class PartitionedDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        # or use the RobertaTokenizer from `transformers` directly.
        self.tokenizer = tokenizer
        path = Path(file_path)
        print("🔥", path)
        self.partition_map = {i: filename for i, filename in enumerate(path.glob("*"))}
        self.partition_start_index = {}
        self.number_of_samples = 0
        for partition, src_file in self.partition_map.items():
            print("🔥", src_file)
            self.partition_start_index[partition] = self.number_of_samples
            self.number_of_samples += len(
                src_file.open(encoding="utf-8").read().splitlines()
            )
        print("--- partition map ---", self.partition_map)
        print("--- partition start index ---", self.partition_start_index)
        print(f"Number of samples: {self.number_of_samples}")

    def __number_of_partitions__(self):
        return len(self.partition_map)

    def __local_index__(self, item_index):
        partition = self.__get_partition__(item_index)
        local_index = item_index - self.partition_start_index[partition]
        print(
            f"Partition for item {item_index} is {partition}, local index is {local_index}"
        )
        return local_index

    def __len__(self):
        return self.number_of_samples

    def __get_partition__(self, item_index):
        for i, start_index in self.partition_start_index.items():
            if item_index < start_index:
                return i - 1
        return i

    def __getitem__(self, item_index):
        partition = self.__get_partition__(item_index)
        print(f"Partition for {item_index} is {partition}")
        src_file = self.partition_map[partition]
        lines = src_file.open(encoding="utf-8").read().splitlines()
        print(f"File {self.partition_map[partition]} has {len(lines)} lines")
        print(f"Looking for line {self.__local_index__(item_index)} of {len(lines)}")
        line = lines[self.__local_index__(item_index)]
        example = self.tokenizer.encode_batch([line])[0].ids
        return torch.tensor(example)


class KariBERTaTokenizer(ByteLevelBPETokenizer):
    def __init__(self, tokenizer_name, max_len=None):
        vocab_file = f"{tokenizer_name}/kariberta-vocab.json"
        merges_file = f"{tokenizer_name}/kariberta-merges.txt"
        self.max_len = max_len if max_len is not None else int(1e12)
        self._pad_token = PAD_TOKEN
        self.mask_token = MASK_TOKEN
        super().__init__(vocab_file, merges_file)
        self.pad_token_id = self.token_to_id(PAD_TOKEN)
        self.mask_token_id = self.token_to_id(MASK_TOKEN)
        self.max_len_single_sentence = self.max_len - self.num_added_tokens(
            False
        )  # take into account special tokens

    @classmethod
    def from_pretrained(cls, tokenizer_name, cache_dir=None):
        tokenizer = KariBERTaTokenizer(tokenizer_name)
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)
        tokenizer.enable_padding()
        return tokenizer

    def num_added_tokens(self, pair=False):
        return self._tokenizer.num_special_tokens_to_add(pair)

    def __len__(self):
        return self._tokenizer.get_vocab_size()

    def convert_tokens_to_ids(self, tokens):
        ids = [self.token_to_id(token) for token in tokens]
        return ids

    def tokenize(self, sentence):
        # return torch.tensor()
        encoding = self._tokenizer.encode(sentence)
        return encoding.tokens

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        cls = [self.token_to_id(START_TOKEN)]
        sep = [self.token_to_id(STOP_TOKEN)]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1=None, already_has_special_tokens=False
    ):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))


def create_norwegian_tokenizer():
    tokenizer = ByteLevelBPETokenizer(
        "./models/KariBERTa-tiny/vocab.json", "./models/KariBERTa-tiny/merges.txt",
    )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")), ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding()
    return tokenizer


PAD_TOKEN = "<pad>"
MASK_TOKEN = "<mask>"
UNKNOWN_TOKEN = "<unk>"
START_TOKEN = "<s>"
STOP_TOKEN = "</s>"


def create_token_masker(tokenizer: ByteLevelBPETokenizer):
    special_tokens = [START_TOKEN, PAD_TOKEN, STOP_TOKEN, UNKNOWN_TOKEN, MASK_TOKEN]
    special_token_ids = {tokenizer.token_to_id(token) for token in special_tokens}
    special_token_ids

    def get_special_tokens_mask(token_ids: torch.tensor):
        return [1 if token_id in special_token_ids else 0 for token_id in token_ids]

    return get_special_tokens_mask
