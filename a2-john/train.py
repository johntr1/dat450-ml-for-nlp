from datasets import load_dataset
from nlplib import build_tokenizer, A1Tokenizer, A1Trainer, A1RNNModel, A1RNNModelConfig 
from transformerlib import *
from transformers import TrainingArguments

TRAIN_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/train.txt"
TEST_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"


'''
tokenizer = build_tokenizer(
    train_file=TRAIN_FILE,
    max_voc_size=15000,
    model_max_length=512
)


encoding = tokenizer(TEST_FILE, return_tensors="pt", padding=True, truncation=True)

#save the tokenizer
tokenizer.save("tokenizer.json")
'''

tokenizer = A1Tokenizer.from_file("tokenizer.pkl")

cfg = A2ModelConfig(
        vocab_size=15000,
        hidden_size=64,
        intermediate_size=256,
        num_attention_heads=8,
        num_hidden_layers=3,
        rope_theta=100000.0,
        rms_norm_eps=1e-6
    )

model = A2Transformer(cfg)

args = TrainingArguments(
    output_dir="./a2_model_attn_mask",
    num_train_epochs=15, 
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
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


