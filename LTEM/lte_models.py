import torch
import json
from torch import nn
from torch.nn import functional as F
import numpy as np


global w
global R
global E
global T


def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param



def get_param1(tensor):
    param = nn.Parameter(tensor)
    nn.init.xavier_normal_(param.data)
    return param





def read_types(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        return len(lines)

def read_eatures(item2id_file, item2fea_file):
    with open(item2id_file, 'r', encoding='utf-8') as file1:
        typedict = dict()
        for row in file1:
            data = row.strip().split()
            typedict[data[0]] = data[1]

    with open(item2fea_file, 'r', encoding='utf-8') as file2:
        typefea = json.load(file2)

    id2dea = dict()
    for key in typedict:
        id = typedict[key]
        if key not in typefea:
            continue
        id2dea[id] = typefea[key]
    return id2dea



class LTEModel(nn.Module):
    def __init__(self, num_ents, num_rels, params=None):
        super(LTEModel, self).__init__()


        self.num_types = read_types("datasets/{}/types.txt".format(params.dataset))
        self.bceloss = torch.nn.BCELoss()
        self.p = params

        if self.p.comprehensive==0:
            self.entities = get_param((num_ents, self.p.init_dim))
            self.weights = get_param((self.p.init_dim, self.p.init_dim))
            self.relations = get_param((num_rels * 2, self.p.init_dim))
            self.types = get_param((self.num_types, self.p.init_dim))

        else:
            self.entity_features = read_eatures("datasets/{}/entities.txt".format(params.dataset), "../DFTFM_output/entity_features.json")
            self.type_features = read_eatures("datasets/{}/types.txt".format(params.dataset), "../DFTFM_output/type_features.json")

            self.weights = torch.randn(self.p.init_dim, self.p.init_dim, requires_grad=True).to('cuda')
            self.relations = torch.randn(num_rels * 2, self.p.init_dim,requires_grad=True).to('cuda')


            entities = list()
            for i in range(num_ents):
                if str(i) not in self.entity_features:
                    entities.append(list(np.zeros(self.p.init_dim)))
                    continue
                feature = self.entity_features[str(i)]
                entities.append(feature)
            self.entities = torch.tensor(entities, dtype=torch.float32, requires_grad=True).to('cuda')

            types = list()
            for i in range(self.num_types):
                if str(i) not in self.type_features:
                    types.append(list(np.zeros(self.p.init_dim)))
                    continue
                feature = self.type_features[str(i)]
                types.append(feature)
            self.types = torch.tensor(types, dtype=torch.float32, requires_grad=True).to('cuda')


            self.entities = get_param1(self.entities)
            self.weights = get_param1(self.weights)
            self.relations = get_param1(self.relations)
            self.types = get_param1(self.types)




        self.device = "cuda"
        self.bias = nn.Parameter(torch.zeros(num_ents))

        self.h_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.t_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.r_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.x_ops = self.p.x_ops
        self.r_ops = self.p.r_ops
        self.diff_ht = False

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

    def exop(self, x, r, x_ops=None, r_ops=None):
        x_tail = x
        if len(x_ops) > 0:
            for x_op in x_ops.split("."):
                x_tail = self.t_ops_dict[x_op](x_tail)

        if len(r_ops) > 0:
            for r_op in r_ops.split("."):
                r = self.r_ops_dict[r_op](r)

        return  x_tail, r


class TransE(LTEModel):
    def __init__(self, num_ents, num_rels, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, params)
        self.loop_emb = get_param([1, self.p.init_dim])
    def forward(self, g, sub, rel, loss_head=False):


        ini_w = self.weights
        ini_e = self.entities
        ini_r = self.relations
        ini_t = self.types

        global W
        global R
        global E
        global T
        W = ini_w
        R = ini_r
        E = ini_e
        T = ini_t

        subs = torch.matmul(ini_t, ini_w)
        rels = ini_r
        ents = ini_e
        ents_t, rels = self.exop(ents - self.loop_emb, rels, self.x_ops, self.r_ops)

        sub_emb = torch.index_select(subs, 0, sub)
        rel_emb = torch.index_select(rels, 0, rel)
        obj_emb = sub_emb + rel_emb

        #all_ent size() :[14951, 50]
        #obj_emb.unsqueeze(1) :[64, 1, 50]
        #obj_emb.unsqueeze(1) - all_ent : [64, 14951, 50]

        x_head = self.p.gamma - \
                 torch.norm(sub_emb.unsqueeze(1) - ents, p=1, dim=2)

        #the size of x:[64, 14951], that is the distance of 64(a batch) vectors to others. p=1:1norm, dim=2 caculate at the last dimension


        head_score = torch.sigmoid(x_head)


        x_tail = self.p.gamma - \
                 torch.norm(obj_emb.unsqueeze(1) - ents_t, p=1, dim=2)


        tail_score = torch.sigmoid(x_tail)

        if loss_head == False:
            return tail_score
        else:
            return head_score


class DistMult(LTEModel):
    def __init__(self, num_ents, num_rels, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, params)

    def forward(self, g, sub, rel, loss_head=False):

        ini_w = self.weights
        ini_e = self.entities
        ini_r = self.relations
        ini_t = self.types

        global W
        global R
        global E
        global T
        W = ini_w
        R = ini_r
        E = ini_e
        T = ini_t

        subs = torch.matmul(ini_t, ini_w)
        rels = ini_r
        ents = ini_e
        ents_t, rels = self.exop(ents, rels, self.x_ops, self.r_ops)

        sub_emb = torch.index_select(subs, 0, sub)
        rel_emb = torch.index_select(rels, 0, rel)
        obj_emb = sub_emb * rel_emb

        # all_ent size() :[14951, 50]
        # obj_emb.unsqueeze(1) :[64, 1, 50]
        # obj_emb.unsqueeze(1) - all_ent : [64, 14951, 50]

        x_head = self.p.gamma - \
                 torch.norm(sub_emb.unsqueeze(1) - ents, p=1, dim=2)

        head_score = torch.sigmoid(x_head)


        x_tail = torch.mm(obj_emb, ents_t.transpose(1, 0))

        x_tail += self.bias.expand_as(x_tail)
        tail_score = torch.sigmoid(x_tail)


        if loss_head == False:
            return tail_score
        else:
            return head_score






class ConvE(LTEModel):
    def __init__(self, num_ents, num_rels, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, params)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.conve_hid_drop)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, g, sub, rel,loss_head=False):


        ini_w = self.weights
        ini_e = self.entities
        ini_r = self.relations
        ini_t = self.types

        global W
        global R
        global E
        global T
        W = ini_w
        R = ini_r
        E = ini_e
        T = ini_t

        subs = torch.matmul(ini_t, ini_w)
        rels = ini_r
        ents = ini_e
        ents_t, rels = self.exop(ents, rels, self.x_ops, self.r_ops)

        sub_emb = torch.index_select(subs, 0, sub)
        rel_emb = torch.index_select(rels, 0, rel)

        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ents_t.transpose(1, 0))
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)


        x_head = self.p.gamma - \
                 torch.norm(sub_emb.unsqueeze(1) - ents, p=1, dim=2)

        head_score = torch.sigmoid(x_head)

        tail_score = score


        if loss_head == False:
            return tail_score
        else:
            return head_score







class FileWriter():
    def __init__(self, params=None):

        self.p = params

    def write_to_file(self):

        dictentity = dict()
        dictrel = dict()
        dicttype = dict()
        w = W.detach().cpu().numpy().tolist()
        t = T.detach().cpu().numpy().tolist()
        r = R.detach().cpu().numpy().tolist()
        e  = E.detach().cpu().numpy().tolist()


        with open('datasets/{}/entities.txt'.format(self.p.dataset), 'r',encoding="UTF-8") as FileEntity:
            countline = 0
            for line in FileEntity:
                dataline = line.strip().split()
                dictentity[dataline[0]] = list(e[countline])
                countline += 1

        with open('datasets/{}/relations.txt'.format(self.p.dataset), 'r',encoding="UTF-8") as FileRel:
            countline = 0
            for line in FileRel:
                dataline = line.strip().split()
                dictrel[dataline[0]] = list(r[countline])
                countline += 1

        with open('datasets/{}/types.txt'.format(self.p.dataset), 'r',encoding="UTF-8") as FileType:
            countline = 0
            for line in FileType:
                dataline = line.strip().split()
                dicttype[dataline[0]] = list(t[countline])
                countline += 1

        with open('../LTEM_output/entities.json', "w") as file1:
            json.dump(dictentity, file1)

        with open('../LTEM_output/relations.json', "w") as file2:
            json.dump(dictrel, file2)

        with open('../LTEM_output/types.json', "w") as file3:
            json.dump(dicttype, file3)

        with open('../LTEM_output/weights.txt', "w") as file4:
            for row in w:
                file4.write(str(row) + "\n")


























