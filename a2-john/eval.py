import torch
import math
from transformerlib import A2Transformer
from datasets import load_dataset
from nlplib import A1Tokenizer, A1RNNModel 
from torch.utils.data import DataLoader
     
def predict_next_word(model, tokenizer, text, k=5):
    print(f'The prompt: {text}')

    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    
    tokenized = tokenizer([text], return_tensors="pt")

    input_ids = tokenized.input_ids
    input_ids = input_ids.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with torch.no_grad():
        logits = model(input_ids)

    last_word_token_logits = logits[0, -1, :]

    probs = torch.softmax(last_word_token_logits, dim=0)
    top_k_probs, top_k_indices = torch.topk(probs, k)

    int_to_srt = tokenizer.int_to_str

    print(f'Top {k} words after the prompt: ')
    for i in range(k):
        token_id = top_k_indices[i].item()
        word = int_to_srt[token_id]
        prob = top_k_probs[i].item()
        print(f'{i+1}. {word}, ID: {token_id}, Probability: {prob}')
     

def generate_text(model, tokenizer, prompt, max_length, temperature, topk):
    print(f'With k: {topk} and Temperature: {temperature}. The prompt: {prompt}')
    
    eos_token_id = tokenizer.eos_token_id

    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    
    tokenized = tokenizer([prompt], return_tensors="pt")

    input_ids = tokenized.input_ids
    input_ids = input_ids.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids)
        
        last_word_logits = logits[0, -2, :]
        scaled_logits = last_word_logits / temperature

        top_k_probs, top_k_indices = torch.topk(scaled_logits, topk)

        probs = torch.softmax(top_k_probs, dim=0)

        dist = torch.distributions.Categorical(logits=probs)
        sample = dist.sample()
        next_token_id = top_k_indices[sample]
        
        if next_token_id.item() == eos_token_id:
            break
        
        input_ids = torch.cat([input_ids, next_token_id.view(1, 1)], dim=1)
    
    generated_ids = input_ids[0].tolist()

    generated_text = ' '.join([tokenizer.int_to_str[idx] for idx in generated_ids])

    return generated_text



def evaluate_perplexity(model, dataloader, tokenizer, device: torch.device, pad_id: int = 0):
        model.eval()
        model.to(device)

        total_nll = 0.0
        total_tokens = 0
        
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
        


        with torch.no_grad():
            for batch in dataloader:
                input_ids = tokenizer(batch['text'], truncation=True, padding=True, return_tensors="pt").input_ids
                X = input_ids[:, :-1]
                Y = input_ids[:, 1:]
                X = X.to(device)
                Y = Y.to(device)

                outputs = model(X)
                
                # Sum of token nll
                nll_sum = loss_func(
                    outputs.contiguous().view(-1, outputs.shape[-1]),
                    Y.contiguous().view(-1),
                )

                total_nll += nll_sum.item()
                total_tokens += (Y != pad_id).sum().item()
    
        mean_ce = total_nll / max(total_tokens, 1) 
        ppl = math.exp(mean_ce)

        return ppl
    
def main():
    # load the model, tokenizer, and dataloader here
    model = A2Transformer.from_pretrained("./a2_model",
                                        local_files_only=True,
                                        use_safetensors=True)
    dataset = load_dataset("text", data_files={"test": "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"})
    dataset = dataset.filter(lambda x: x['text'].strip() != '')

    tokenizer = A1Tokenizer.from_file("tokenizer.pkl")    

    dataloader = DataLoader(dataset["test"], batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppl = evaluate_perplexity(model, dataloader, tokenizer, device, pad_id=tokenizer.pad_token_id)
    print(f"Perplexity on test set: {ppl}")

    print("Predicting next word...")
    print("===================================")
    prompt = "He lives in Los"
    predict_next_word(model, tokenizer, prompt)


    print("Generating text...")
    print("===================================")
    prompt1 = 'In natural language processing, a Transformer'
    prompt2 = 'Is Stockholm the capital of Sweden? Answer yes or no. The answer is'
    prompt3 = 'Write a Python program that reverses a list.'

    generated1 = generate_text(model, tokenizer, prompt1, max_length=50, temperature=0.8, topk=10)
    print(f'Text generated: {generated1}')
    generated2 = generate_text(model, tokenizer, prompt2, max_length=50, temperature=0.8, topk=10)
    print(f'Text generated: {generated2}')
    generated3 = generate_text(model, tokenizer, prompt3, max_length=50, temperature=0.8, topk=10)
    print(f'Text generated: {generated3}')



if __name__ == "__main__":
    main() 