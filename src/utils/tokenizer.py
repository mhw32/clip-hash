import torch
from DeBERTa import deberta


class DebertaV3Tokenizer:

    def __init__(self, model='base'):
        vocab_path, vocab_type = deberta.load_vocab(pretrained_id=model)
        self.tokenizer = deberta.tokenizers[vocab_type](vocab_path)

    def _tokenize(self, text):
        pieces = list(text)  # convert hash to letters
        vocab = self.tokenizer.vocab

        def _norm(x):
            if x not in vocab or x=='<unk>':
                return '[UNK]'
            else:
                return x
        
        pieces = [_norm(p) for p in pieces]
        return pieces

    def tokenize(self, input_str, max_seq_len=512):
        tokens = self._tokenize(input_str)
        tokens = tokens[:max_seq_len-2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1]*len(input_ids)
        paddings = max_seq_len-len(input_ids)
        input_ids = input_ids + [0]*paddings
        attention_mask = attention_mask + [0]*paddings

        output = dict(
            input_ids=torch.tensor(input_ids, dtype=torch.int),
            attention_mask= torch.tensor(attention_mask, dtype=torch.int),
        )
        return output
