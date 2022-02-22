import pandas as pd
from tqdm.auto import tqdm

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from collections import Counter, OrderedDict

def build_vocab(args, min_freq:int=1, max_vocab_size:int=None):
    # Load data from csv file
    path = args.train_data_path
    if path.endswith('.csv'):
        df = pd.read_csv(path, header=None)

        if args.text_column_index is not None:
            text_column = df.columns[args.text_column_index]
        else:
            raise ValueError('text_column_index is not specified')
            
        texts = df[text_column].values    
    else:
        raise ValueError('Invalid data path, only support csv file')

    # Tokenize text and build OrderedDict
    counter = Counter()
    tokenizer = get_tokenizer("basic_english")

    for i, text in enumerate(tqdm(texts, total=len(texts), desc=f'Tokenizing from {path}')):
        tokens = tokenizer(text.lower())
        counter.update(tokens)
    
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    # Build vocab
    vocabulary = vocab(ordered_dict, min_freq=min_freq)

    vocabulary.insert_token('<pad>', 0)
    vocabulary.insert_token('<sos>', 1)
    vocabulary.insert_token('<eos>', 2)
    vocabulary.insert_token('<unk>', 3)

    vocabulary.set_default_index(vocabulary['<unk>'])

    return vocabulary