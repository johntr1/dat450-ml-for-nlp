from A1_skeleton import *
import sys


train_file = 'a1/train.txt'

if __name__ == "__main__":
    test_texts = ['This is a test.', 'Another test.', 'Test']
    test_texts2 = ['This is a test.', 'Another test.']
    
    tokenizer = build_tokenizer(train_file)
    test1 = tokenizer(test_texts, return_tensors='pt', padding=True,
          truncation=False)
    test2 = tokenizer(test_texts2, return_tensors='pt', padding=True,
          truncation=False)
    
    print(test1)

    
    