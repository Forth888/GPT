# import different socurces
import os
import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

# getting the existing and trained model
def load_models(model_name):
    print("---------Getting the trained model---------")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model_path = model_name
    model.load_state_dict(torch.load(model_path))
    print("---------Got---------")
    return tokenizer, model

def choose_from_top_k_top_n(probs, k=50, p=0.8):
    index = np.argpartition(probs, -k)[-k:] # return ordered indexes
    top_prob = probs[index]
    top_prob = {i: top_prob[idx] for idx,i in enumerate(index)}

    sorted_top_prob = {k: v for k, v in sorted(top_prob.items(), key=lambda item: item[1], reverse=True)}

    t=0
    f=[]
    pr=[]
    for k,v in sorted_top_prob.items():
        t+=v;
        f.append(v)
        pr.append(v)
        if t>=p:
          break
    top_prob = pr / np.sum(pr)
    token_id = np.random.choice(f, 1, p = top_prob)

    return int(token_id)


# grnerating paraphrases
def generate(model, tokenizer, texts, label, device):
    with torch.no_grad():
        for id in range():
            finished = False
            cur_ids = torch.tensor(tokenizer.encode(label)).unsqueeze(0).to('cpu')
            for i in range(100):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0, -1], dim=0)

                if i < 5:
                    n = 10
                else:
                    n = 5

                next_token_id = choose_from_top_k_top_n(softmax_logits.to('cpu').numpy())  # top-k-top-n sampling
                cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id], dim=1)
            if next_token_id in tokenizer.encode('<|endoftext|>'):
                finished = True
                break

            if finished:
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                print(output_text)
            else:
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                print(output_text)

# the main method
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for Generation model')

    parser.add_argument('--model_name', default='mymodel.pt', type=str, action='store', help='Name of the model file')
    parser.add_argument('--text', type=int, default=5, action='store', help='Number of sentences in outputs')
    parser.add_argument('--label', type=str, action='store', help='Label for which to produce text')
    args = parser.parse_args()

    sentences = args.sentences
    model_name = args.model_name
    label = args.label

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # getting the existing model
    tokenizer, model = load_models(model_name)
    #generate paraphrase
    generate(tokenizer, model, sentences, label, device)