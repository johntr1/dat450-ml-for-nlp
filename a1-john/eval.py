import torch
import math
from datasets import load_dataset
from nlplib import A1Tokenizer, A1RNNModel 
from torch.utils.data import DataLoader

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
    model = A1RNNModel.from_pretrained("./ten_epoch_model",
                                        local_files_only=True,
                                        use_safetensors=True)
    dataset = load_dataset("text", data_files={"test": "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"})
    dataset = dataset.filter(lambda x: x['text'].strip() != '')

    tokenizer = A1Tokenizer.from_file("tokenizer.pkl")    

    dataloader = DataLoader(dataset["test"], batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppl = evaluate_perplexity(model, dataloader, tokenizer, device, pad_id=tokenizer.pad_token_id)
    print(f"Perplexity on test set: {ppl}")


if __name__ == "__main__":
    main() 