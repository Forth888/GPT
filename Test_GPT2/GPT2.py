# getting the existing and trained model
# import different socurces
import os, csv
import torch
import argparse, time, datetime
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

# the get_model is to get model & tokenizer
def get_init_model(model_name):
    print("---------Getting GPT-2 Model---------")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    print("---------Got---------")
    return tokenizer, model


# processing data to input it in model
class MyDataset(Dataset):
    def __init__(self, data_file_name, data_dir='.data/'):
        super().__init__()

        data_path = os.path.join(data_dir, data_file_name)

        self.data_list = []
        # self.end_of_text_token = " <|endoftext|> "

        with open(data_path, encoding='UTF-8') as csv_file:
            csv_reader = csv.reader(csv_file,delimiter='\n')
            for row in csv_reader:
                #data_str = f"{row[0]}: {row[1]}"
                data_str = f"{row[0]}"
                self.data_list.append(data_str)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]


def get_data(data_set):
    print("---------Processing the dataset---------")
    dataset = MyDataset(data_set)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("---------Processed---------")
    return data_loader


# traing many
def trainer(model, tokenier, train_data, batch_size, epoches, device):
    batch_count = 0
    sum_loss = 0.0


    for epoch in range(epoches):
        start_time_epoch = datetime.datetime.now()
        print(f"---------The {epoch + 1} epoch starts at{start_time_epoch}---------")
        for id, text in enumerate(train_data):
            text = torch.tensor(tokenier.encode(text[0])).unsqueeze(0).to(device)

            outputs = model(text, labels=text)

            loss, logits = outputs[:2]
            loss.backward()
            sum_loss = sum_loss + loss.data

            if id % batch_size == 0:
                batch_count += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            if batch_count == 10:
                print(f"The sum_loss is {sum_loss} and the time is {datetime.datetime.now()} now")
                batch_count = 0
                sum_loss = 0.0
        end_time_eopch = datetime.datetime.now()
        print(f"this epoch end at  {end_time_eopch} and spent{end_time_eopch-start_time_epoch}")

    return model


# save trained models.
def save_trained_model(model,name):
    print("---------Saving the trained model to disk---------")
    torch.save(model.state_dict(),f"{name}.pt")
    print("---------Saved---------")


# the main method
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments of training GPT2 model')

    parser.add_argument('--model_name', default='mymodel.pt', type=str, action='store',
                        help='The name user wants to save the model call')
    parser.add_argument('--training_data', default='mydata.csv', type=str, action='store',
                        help='The name of traing data user wants to use')
    parser.add_argument('--epoches', default=3, type=int, action='store', help='the number of epoch')
    parser.add_argument('--batch_size', default=32, type=int, action='store', help='Batch size')
    parser.add_argument('--learning_rate', default=3e-5, type=float, action='store', help='Learning rate for the model')
    parser.add_argument('--max_len', default=200, type=int, action='store', help='the maximum length of sequence')
    parser.add_argument('--warmup', default=300, type=int, action='store', help='Number of warmup steps')

    args = parser.parse_args()

    saved_model_name = args.model_name
    training_data = args.training_data
    epoches = args.epoches
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    max_len = args.max_len
    warmup = args.warmup

    # get tokenizer, model
    tokenizer, model = get_init_model("gpt2-medium")
    # get processed dataset
    data_processed = get_data(training_data)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model_device = model.to(device)

    # call train
    model_device.train()

    optimizer = AdamW(model_device.parameters(), lr=learning_rate)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup,
                                                                   num_training_steps=-1)

    # train epoches
    #model, tokenier, train_data, batch_size, epoches, device
    model = trainer(model_device, tokenizer, data_processed, batch_size, epoches, device)
    save_trained_model(model, saved_model_name)
