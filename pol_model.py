import json
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.tensorboard import SummaryWriter
import random
import time

LEARNING_RATE = 3e-6
OPTIM = "adam"
WEIGHT_DECAY = 1e-2
MOMENTUM = 0.98
BATCH_SIZE = 16
NUM_WORKERS = 4
LR_STEP = 1
GAMMA = 0.3
ENCODING = { 
    "anarchism": [0,0],
    "anarcho_capitalism": [1,1],
    "antiwork": [2,0],
    "breadtube": [3,0],
    "chapotraphouse": [4,0],
    "communism": [5,0],
    "completeanarchy": [6,0],
    "conservative": [7,1],
    "cringeanarchy": [8,1],
    "democraticsocialism": [9,0],
    "esist": [10,0],
    "fullcommunism": [11,0],
    "goldandblack": [12,1],
    "jordanpeterson": [13,1],
    "keep_track": [14,0],
    "latestagecapitalism": [15,0],
    "latestagesocialism": [16,1],
    "liberal": [17,0],
    "libertarian": [18,1],
    "neoliberal": [19,0],
    "onguardforthee": [20,1],
    "ourpresident": [21,1],
    "political_revolution": [22,0],
    "politicalhumor": [23,0],
    "politics": [24,0],
    "progressive": [25,0],
    "republican": [26,1],
    "sandersforpresident": [27,0],
    "selfawarewolves": [28,0],
    "socialism": [29,0],
    "the_donald": [30,1],
    "the_mueller": [31,0],
    "thenewright": [32,1],
    "voteblue": [33,0],
    "wayofthebern": [34,0],
    "yangforpresidenthq": [35,0]
}

def get_df(file_name, percent=1.0, range_mode=False, rng = (4,256)):
    lst = []
    start_time = time.time()
    with open(file_name) as f:
        header = True
        for line in f:
            if header:
                header = False
            elif percent == 1.0 or random.random() < percent:
                subreddit = line.split(',')[1:2][0]
                score = line.split(',')[6:7][0] 
                body = ','.join(line.split(',')[7:])
                sub_id = ENCODING[subreddit][0]
                pos_id = ENCODING[subreddit][1]
                if int(score) >= 0:
                    if range_mode:
                        if body.count(' ') in range(rng[0],rng[1]):
                            lst.append([sub_id,pos_id,body])
                    else:
                        lst.append([sub_id,pos_id,body])
    delta = time.time() - start_time
    print("Dataframe done! Took {}".format(time.strftime("%H:%M:%S",time.gmtime(delta))))
    return pd.DataFrame(lst, columns =["sub_id", "pos_id", "body"])

class PolReddit(torch.utils.data.Dataset):
    def __init__(self, df, pos_mode = True, test = False):
        super(PolReddit).__init__() 
        self.df = df
        self.body = self.df['body']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if pos_mode:
            self.target = "pos_id"
        else:
            self.target = "sub_id"
        self.test = test

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            self.body[index],
            add_special_tokens=True,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        if self.test:
            return {
                'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
            }
        else:
            return {
                'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'targets': torch.tensor(self.df.loc[index][self.target], dtype=torch.long)
            }

    def __len__(self):
        return len(self.df.index)

def train(trainloader, num_epochs=10, auto_val=False):
    for epoch in range(num_epochs):
        run_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            inputs, labels, mask = data['ids'].to(
                device), data['targets'].to(device), data['mask'].to(device)
            outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)
            loss_val = outputs[0]
            loss_val.backward()
            total += BATCH_SIZE
            optimizer.step()
            run_loss += loss_val.item()
            correct += (outputs[1].data.max(1).indices == labels).sum().item()
            if i % 100 == 0:
                writer.add_scalar('Loss/train', (run_loss/total), i*BATCH_SIZE + epoch*len(trainloader))
                writer.add_scalar('Accuracy/train', (correct/total), i*BATCH_SIZE + epoch*len(trainloader))
                run_loss = 0.0
                correct = 0.0
                total = 0.0
        if auto_val:
            test(val_loader, epoch)
        scheduler.step()

def test(testloader, count=1):
    correct = 0.0
    total = 0.0
    run_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            inputs, labels, mask = data['ids'].to(
                device), data['targets'].to(device), data['mask'].to(device)
            outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)
            total += BATCH_SIZE
            correct += (outputs[1].data.max(1).indices == labels).sum().item()
            run_loss += outputs[0]
        writer.add_scalar('Accuracy/test', correct/total, count)

if __name__ == "__main__":
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    if OPTIM == "sgd":
        optimizer = torch.optim.SGD(
            params=model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    elif OPTIM == "adam":
        optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=GAMMA)
    writer = SummaryWriter()


    train_df = get_df("politics10.csv", percent=0.5)

    train_data, val_data = torch.utils.data.random_split(PolReddit(train_df), [int(.8*len(train_df)), len(train_df)-int(.8*len(train_df))])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory=True, shuffle=True)
    train(train_loader, num_epochs=5, auto_val=True)
    torch.save(model.state_dict(), "run1.pth")
