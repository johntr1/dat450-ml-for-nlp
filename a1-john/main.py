from datasets import load_dataset
from nlplib import build_tokenizer, A1Tokenizer, A1Trainer, A1RNNModel, A1RNNModelConfig 
from transformers import TrainingArguments

TRAIN_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/train.txt"
TEST_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"

"""
tokenizer = build_tokenizer(
    train_file=train_file,
    max_voc_size=10000,
    model_max_length=512
)


encoding = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)

#save the tokenizer
tokenizer.save("tokenizer.json")
"""

tokenizer = A1Tokenizer.from_file("tokenizer.pkl")

config = A1RNNModelConfig(
    vocab_size=10_000,
    embedding_size=128,
    hidden_size=256,
    pad_token_id=tokenizer.pad_token_id
)

model = A1RNNModel(config=config)

args = TrainingArguments(
    output_dir="./a1_model",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    optim="adamw_torch",
    eval_strategy="epoch",
    use_cpu=False,
    no_cuda=False,
    learning_rate=1e-3,
)

print("Loading Dataset...")
dataset = load_dataset("text", data_files={"train": TRAIN_FILE, "test": TEST_FILE})
dataset = dataset.filter(lambda x: x['text'].strip() != '')

print("Dataset Loaded Successfully")

print("Starting Training...")
trainer = A1Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

trainer.train()