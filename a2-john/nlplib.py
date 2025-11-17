import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from typing import Dict, List, Callable, Optional, Any

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os
import time
###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

def build_tokenizer(train_file, tokenize_fun=lowercase_tokenizer, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             tokenize_fun:      The function that maps a text to a list of string tokens.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """

    special_tokens = [pad_token, unk_token, bos_token, eos_token]
    
    if max_voc_size is not None and max_voc_size < len(special_tokens):
        raise ValueError('max_voc_size should be at least ' + str(len(special_tokens)))

    count = Counter()
    
    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = tokenize_fun(line)
            
            if model_max_length is not None and model_max_length > 0:
                tokens = tokens[:model_max_length]

            count.update(tokens)

    # pop the special tokens from the counter
    for tok in special_tokens:
        count.pop(tok, None)
    
    if max_voc_size is None:
        slots = len(count)   
    else:
        slots = max(0, max_voc_size - len(special_tokens))

    most_common_toks =  [tok for tok, _ in count.most_common(slots)]
    
    combined_vocab = special_tokens + most_common_toks

    str_to_int = {tok: i for i, tok in enumerate(combined_vocab)}
    int_to_str = {i: tok for i, tok in enumerate(combined_vocab)}

    return A1Tokenizer(
        str_to_int=str_to_int,
        int_to_str=int_to_str,
        tokenize_fun=tokenize_fun,
        pad_token=pad_token,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        model_max_length=model_max_length
    )
class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(
            self,
            str_to_int: Dict[str, int],
            int_to_str: Dict[int, str],
            tokenize_fun: Callable[[str], List[str]],
            pad_token: str = "<PAD>",
            unk_token: str = "<UNK>",
            bos_token: str = "<BOS>",
            eos_token: str = "<EOS>",
            model_max_length: Optional[int] = None,
            ):
        # TODO: store all values you need in order to implement __call__ below.

        self.str_to_int = str_to_int
        self.int_to_str = int_to_str 
        self.tokenize_fun = tokenize_fun

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.pad_token_id = str_to_int[pad_token]
        self.unk_token_id = str_to_int[unk_token]
        self.bos_token_id = str_to_int[bos_token]
        self.eos_token_id = str_to_int[eos_token]
        self.model_max_length = model_max_length


    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
           
           Args:
             texts:           The texts to tokenize.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')
        
        # TODO: Your work here is to split the texts into words and map them to integer values.
        # 
        # - If `truncation` is set to True, the length of the encoded sequences should be 
        #   at most self.model_max_length.
        # - Encoded sequences should start with the beginning-of-sequence dummy; non-truncated
        #   sequences should end with the end-of-sequence dummy; out-of-vocabulary tokens should
        #   be encoded with the 'unknown' dummy.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an 
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.
        
        if isinstance(texts, str):
            texts = [texts] # make it a list of one text if needed
        
        all_ids = []
        for text in texts:
            tokens = self.tokenize_fun(text)
            # start with the beginning of sequence token
            ids = [self.bos_token_id]
            # map each token
            for tok in tokens:
                ids.append(
                    self.str_to_int.get(tok, self.unk_token_id)
                )
            
            # truncation
            if truncation and self.model_max_length is not None:
                # if truncating, leave space for eos token
                max_len = self.model_max_length - 1
                ids = ids[:max_len]

            # add the end of sequence token
            if self.model_max_length is None or len(ids) < self.model_max_length:
                ids.append(self.eos_token_id)
            all_ids.append(ids)
            
        # padding
        if padding:
            max_len = max(len(seq) for seq in all_ids)
        else:
            max_len = None

        input_ids = []
        attention_mask = []

        for seq in all_ids:
            if max_len is not None:
                orig_len = len(seq)
                pad_length = max_len - len(seq)
                if pad_length > 0:
                    seq = seq + [self.pad_token_id] * pad_length
                mask = [1] * orig_len + [0] * pad_length
            else:
                mask = [1] * len(seq)

            input_ids.append(seq)
            attention_mask.append(mask) 

        # convert to tensors if needed 
        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

        return BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask})

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.str_to_int)

    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=None, embedding_size=None, hidden_size=None, pad_token_id=0, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.pad_token_id = pad_token_id

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_size,
            padding_idx=config.pad_token_id
        )
        self.rnn = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            batch_first=True,
        )
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, X):
        """The forward pass of the RNN-based language model.
        
           Args:
             X:  The input tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
        """
        embedded = self.embedding(X)
        rnn_out, _ = self.rnn(embedded)
        out = self.unembedding(rnn_out)
        return out


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')


    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
    
        train_loader = DataLoader(self.train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
        val_loader = DataLoader(self.eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)

        # TODO: Your work here is to implement the training loop.
        
        # for each training epoch (use args.num_train_epochs here):
        #   for each batch B in the training set:
        #
        #       PREPROCESSING AND FORWARD PASS:
        #       input_ids = apply your tokenizer to B
	    #       X = all columns in input_ids except the last one
	    #       Y = all columns in input_ids except the first one
	    #       put X and Y onto the GPU (or whatever device you use)
        #       apply the model to X
        #   	compute the loss for the model output and Y
        #
        #       BACKWARD PASS AND MODEL UPDATE:
        #       optimizer.zero_grad()
        #       loss.backward()
        #       optimizer.step()
        for epoch in range(args.num_train_epochs):
            self.model.train()
            steps = 0
            total_loss = 0.0
            
            print(f"\n==== Epoch {epoch+1}/{args.num_train_epochs} ====")

            start_time = time.time()

            for batch in train_loader:
                encoding = self.tokenizer(batch['text'], truncation=True, padding=True, return_tensors="pt")


                input_ids = encoding.input_ids.to(device)
                attention_mask = encoding.attention_mask.to(device)
                X = input_ids[:, :-1]
                Y = input_ids[:, 1:]
                X_mask = attention_mask[:, :-1]

                X = X.to(device)
                Y = Y.to(device)
                outputs = self.model(X, attn_mask=X_mask)
                loss = loss_func(
                    outputs.contiguous().view(-1, outputs.shape[-1]),
                    Y.contiguous().view(-1)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                steps += 1 
                total_loss += loss.item()
                batch_loss = total_loss / steps
                if steps % 100 == 0:
                    print(f"\r[Train] Epoch {epoch+1} Step {steps} - Loss: {batch_loss:.4f} - Elapsed time: {time.time() - start_time:.2f}s", end='')
            
            avg_loss = total_loss / steps
            elapsed_time = time.time() - start_time
            print(f"\n[Train] Epoch {epoch+1} completed - Loss: {avg_loss:.4f} - Elapsed time: {elapsed_time:.2f}s")
            # validation
            # set the model to evaluation mode
            self.model.eval()
            val_total_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    encoding = self.tokenizer(val_batch['text'], truncation=True, padding=True, return_tensors="pt")

                    input_ids = encoding.input_ids.to(device)
                    attention_mask = encoding.attention_mask.to(device)

                    X_val = input_ids[:, :-1]
                    Y_val = input_ids[:, 1:]
                    X_val_mask = attention_mask[:, :-1]

                    X_val = X_val.to(device)
                    Y_val = Y_val.to(device)
                    
                    val_outputs = self.model(X_val, attn_mask=X_val_mask)
                    val_loss = loss_func(
                        val_outputs.contiguous().view(-1, val_outputs.size(-1)),
                        Y_val.contiguous().view(-1)
                    )
                    val_steps += 1
                    val_total_loss += val_loss.item()
                    val_batch_loss = val_total_loss / val_steps
                    if val_steps % 100 == 0:
                        print(f"\r[Val] Epoch {epoch+1} Step {val_steps} - Loss: {val_batch_loss:.4f} - Elapsed time: {time.time() - start_time:.2f}s", end='')

            val_avg_loss = val_total_loss / val_steps
            elapsed_time = time.time() - start_time
            print(f"\n[Val] Epoch {epoch+1} completed - Loss: {val_avg_loss:.4f} - Elapsed time: {elapsed_time:.2f}s")
        
        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)



def predict_next_word(model, tokenizer, text, k=5):
    print(f'The prompt: {text}')


    model.eval()
    
    tokenized = tokenizer([text], return_tensors="pt")

    input_ids = tokenized.input_ids

    with torch.no_grad():
        logits = model(input_ids)

    last_word_token_logits = logits[0, -2, :]

    probs = torch.softmax(last_word_token_logits, dim=0)
    top_k_probs, top_k_indices = torch.topk(probs, k)

    int_to_srt = tokenizer.int_to_str

    print(f'Top {k} words after the prompt: ')
    for i in range(k):
        token_id = top_k_indices[i].item()
        word = int_to_srt[token_id]
        prob = top_k_probs[i].item()
        print(f'{i+1}. {word}, ID: {token_id}, Probability: {prob}')