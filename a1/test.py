from A1_skeleton import *
import sys


train_file = 'train_test.txt'

if __name__ == "__main__":
    test_texts = ['This is a test.', 'Another test.']
    tokenizer = build_tokenizer(train_file, return_tensors='pt', padding=True,
          truncation=True)
    print(tokenizer(test_texts))
    
    