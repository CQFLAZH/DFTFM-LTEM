import json
import torch
import random
from transformers import BertTokenizer, BertModel
from torch import nn
from tqdm import tqdm
from DFTFM import GlobalValue as g

from type_info import TrainRun
import Utils as tools
F = nn.functional
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#use gpu
trainrun = TrainRun()

class Dataloader:
    _dataset = None
    all_types = None
    type_to_id = None
    id_to_type = None
    tokenizer = None

    typelist = None
    entitylist = None

    entity2typeweights_dict = None
    entity2typescores_dict = None
    #records type weights&scores of each entity

    entity2descriptions_dict = None
    #records each entity's discription

    def __init__(self, split='train', batchsize=g.batchsize, shuffle=True):
        self.batchsize = batchsize
        self.shuffle = shuffle
        if Dataloader._dataset is None:
            descriptions = []
            entities = []
            types = []
            with open("data(Fb15k_demo)/FB15k/descriptions.json", 'r', encoding='utf-8') as f:
                # read file
                entity2des_dict = json.load(f)

            for i in range(len(trainrun.fb_HT)):
                key = trainrun.fb_HT[i]
                type = trainrun.fb_LT[i]
                if key in entity2des_dict:
                    entities.append(key)
                    types.append(type)


            Dataloader.entity2descriptions_dict = entity2des_dict
            Dataloader.entity2typeweights_dict = tools.gen_typeweights(entities, types)
            Dataloader.typelist = tools.gen_typesets(types)
            Dataloader.entitylist = tools.gen_entitysets(entities)
            Dataloader.entity2typescores_dict = tools.gen_typevecs(Dataloader.typelist, Dataloader.entitylist, Dataloader.entity2typeweights_dict)


            Dataloader._dataset = Dataloader.entitylist
            Dataloader.all_types = Dataloader.typelist
            Dataloader.all_types.sort()
            Dataloader.type_to_id = trainrun.type2id
            Dataloader.id_to_type = trainrun.id2type
            Dataloader.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


        split_point = int(len(Dataloader._dataset) * 0.99)
        if split == 'train':
            self.pairs = Dataloader._dataset[:split_point]
        elif split == 'val':
            self.pairs = Dataloader._dataset[split_point:]
        else:
            self.pairs = Dataloader._dataset
        #split datasets for training, validation, testing
        self.i = 0
        self.max_iterations = (len(self.pairs) + batchsize - 1) // batchsize
        #i records the batch number of each epoch

    def __len__(self):
        return self.max_iterations
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.pairs)
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= self.max_iterations:
            raise StopIteration

        ys = []
        texts = []
        pairs = self.pairs[self.i * self.batchsize: (self.i + 1) * self.batchsize]

        for name in pairs:
            ys.append(Dataloader.entity2typescores_dict[name])
            texts.append(Dataloader.entity2descriptions_dict[name])
        ys = torch.tensor([item.detach().numpy() for item in ys])


        encoded = Dataloader.tokenizer.batch_encode_plus(
                texts,
                max_length=g.text_max_length,
                pad_to_max_length=True
        )
        # encoded consists of three parts: 'input_ids', 'token_type_ida', 'attention_mask'

        masks = encoded['attention_mask']
        xs = encoded['input_ids']

        self.i += 1
        return torch.tensor(xs), torch.tensor(masks), torch.tensor(ys)


class Model(nn.Module):
    def __init__(self, num_classes, feature_only=False, preload=False):
        super().__init__()
        self.feature_only = feature_only
        self.num_classes = num_classes
        #num_class records the number of types

        if preload:
            self.preload = torch.load('data(Fb15k_demo)/text-features.pkl')

        else:
            self.preload = None
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.classifier1 = nn.Linear(768, g.outputemb)
            self.classifier2 = nn.Linear(g.outputemb, num_classes)
    
    def forward(self, x, masks, idxs=None):
        if self.preload is not None:
            return self.preload[idxs].to(device)
        if self.feature_only:
            with torch.no_grad():
                x = self.bert(x, attention_mask=masks)[1]
                x1 = self.classifier1(x)
            return x1
        else:
            x = self.bert(x, attention_mask=masks)[1]
            x1 = self.classifier1(x)
            return self.classifier2(x1)

    #forward layers
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

def validate(model):
    loader = Dataloader(split='val')
    val_loss = 0.
    n = 0
    criterion = nn.CrossEntropyLoss()

    for xs, masks, ys in loader:
        xs, masks, ys = xs.to(device), masks.to(device), ys.to(device)
        logits = model(xs, masks)
        loss = criterion(logits, ys)
        val_loss += loss.item()
        n += 1
    
    print(f"val loss: {val_loss / n}")
 
def train():
    LOG_FREQ = 20
    SAVE_FREQ = 200

    loader = Dataloader()
    model = Model(len(loader.all_types)).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)

    validate(model)

    total_it = 0
    for ep in range(g.epoch):
        running_loss = 0
        for i, (xs, masks, ys) in enumerate(loader):
            xs, ys = xs.to(device), ys.to(device)
            masks = masks.to(device)
            total_it += 1
            optim.zero_grad()
            logits = model(xs, masks)
            loss = criterion(logits, ys)
            loss.backward()
            optim.step()
            running_loss += loss.item()

            if i % LOG_FREQ== LOG_FREQ-1:
                print(f"Train {i:05d}/{ep:05d}  Loss {running_loss / LOG_FREQ:.4f} ")
                running_loss = 0.
            #every 200 rounds print the loss and set to zero
            if i % SAVE_FREQ == SAVE_FREQ - 1:
                model.save('data(Fb15k_demo)/text-model.pyt')
            #every 2000 rounds save the model
        validate(model)

def export_features():
    loader = Dataloader(split='all', shuffle=False, batchsize=100)
    model = Model(len(loader.all_types)).to(device)
    model.load('data(Fb15k_demo)/text-model.pyt')
    model.feature_only = True
    model.eval()
    namevecdict={}
    features = []

    for i, (xs, masks, _) in tqdm(enumerate(loader), total=len(loader)):
        xs, masks = xs.to(device), masks.to(device)
        with torch.no_grad():
            feat = model(xs, masks).detach().cpu().tolist()
            features.extend(feat)

    for i in range(len(loader.entitylist)):
        namevecdict[loader.entitylist[i]] = features[i]



    with open('../DFTFM_output/entity_features.json', 'w') as _file:
        json.dump(namevecdict, _file)


    features = torch.tensor(features)
    with open('data(Fb15k_demo)/text-features.pkl', 'wb') as f:
        torch.save(features, f)

    tools.WriteFile()



if __name__ == '__main__':
    train()
    export_features()
