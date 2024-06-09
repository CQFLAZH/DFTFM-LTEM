from collections import defaultdict as ddict

def process(dataset, num_rel, params):
    """
    pre-process dataset
    :param dataset: a dictionary containing 'train', 'valid' and 'test' data.
    :param num_rel: relation number
    :return:
    """

    entity2id = create_dict("datasets/{}/entities.txt".format(params.dataset))
    type2id = create_dict("datasets/{}/types.txt".format(params.dataset))
    entity2type = read_type_file("datasets/{}/{}_Entity_Type_train{}.txt".format(params.dataset,params.dataset,params.errorrate), entity2id, type2id)

    sr2o = ddict(list)
    sr2s = ddict(list)


    for subj, rel, obj in dataset['train']:
        if subj not in entity2type or obj not in entity2type:
            continue
        type_list1 = entity2type[subj]
        type_list2 = entity2type[obj]
        for tp in type_list1:
            sr2o[(tp, rel)].append(obj)
            sr2s[(tp, rel)].append(subj)

        for tp in type_list2:
            sr2o[(tp, rel + num_rel)].append(subj)
            sr2s[(tp, rel + num_rel)].append(obj)
    sr2o_train = {k: list(v) for k, v in sr2o.items()}
    sr2s_train = {k: list(v) for k, v in sr2s.items()}


    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            if subj not in entity2type or obj not in entity2type:
                continue
            type_list1 = entity2type[subj]
            type_list2 = entity2type[obj]
            for tp in type_list1:
                sr2o[(tp, rel)].append(obj)
                sr2s[(tp, rel)].append(subj)
            for tp in type_list2:
                sr2o[(tp, rel + num_rel)].append(subj)
                sr2s[(tp, rel + num_rel)].append(obj)
    sr2o_all = {k: list(v) for k, v in sr2o.items()}
    sr2s_all = {k: list(v) for k, v in sr2s.items()}
    #sr2o_all 为{（a，b）：【c，d，e】。。。}



    triplets = ddict(list)
    for (subj, rel), obj in sr2o_train.items():
        triplets['train'].append({'triple': (subj, rel, -1), 'label': sr2o_train[(subj, rel)], 'entity':sr2s_train[(subj, rel)]  })


    
    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            if subj not in entity2type or obj not in entity2type:
                continue
            type_list1 = entity2type[subj]
            type_list2 = entity2type[obj]
            for tp in type_list1:
                triplets[f"{split}_tail"].append({'triple': (tp, rel, obj), 'label': sr2o_all[(tp, rel)], 'entity':sr2s_all[(tp, rel)] })
            for tp in type_list2:
                triplets[f"{split}_head"].append(
                    {'triple': (tp, rel + num_rel, subj), 'label': sr2o_all[(tp, rel + num_rel)], 'entity': sr2s_all[(tp, rel + num_rel)] })

    triplets = dict(triplets)
    return triplets





    #
    # Sr2o = ddict(set)
    # for subj, rel, obj in dataset["valid"]:
    #     if subj not in entity2type or obj not in entity2type:
    #         continue
    #     type_list1 = entity2type[subj]
    #     type_list2 = entity2type[obj]
    #     for tp in type_list1:
    #         Sr2o[(tp, rel)].add(obj)
    #     for tp in type_list2:
    #         Sr2o[(tp, rel + num_rel)].add(subj)
    # Sr2o_valid = {k: list(v) for k, v in Sr2o.items()}
    #
    # for (subj, rel), obj in Sr2o_valid.items():
    #     triplets['valid'].append({'triple': (subj, rel, obj), 'label': sr2o_train[(subj, rel)]})
    #
    #
    #
    # Sr2o = ddict(set)
    # for subj, rel, obj in dataset["test"]:
    #     if subj not in entity2type or obj not in entity2type:
    #         continue
    #     type_list1 = entity2type[subj]
    #     type_list2 = entity2type[obj]
    #     for tp in type_list1:
    #         Sr2o[(tp, rel)].add(obj)
    #     for tp in type_list2:
    #         Sr2o[(tp, rel + num_rel)].add(subj)
    # Sr2o_test = {k: list(v) for k, v in Sr2o.items()}
    #
    # for (subj, rel), obj in Sr2o_test.items():
    #     triplets['test'].append({'triple': (obj, rel + num_rel, subj), 'label': sr2o_all[(obj, rel + num_rel)]})
    #
    #
    #
    #
    # triplets = dict(triplets)
    # return triplets
    # # triplets['valid']
    # # triplets['test']



#读2个文件组成字典
#文件1.txt
#A  B
def create_dict(filename):
    my_dict = {}

    with open(filename, 'r',encoding="UTF-8") as file:
        for line in file:
            key, value = line.strip().split()
            my_dict[key] = value
    return my_dict



#读取实体类型信息
#{entity：【type1，type2，type3】。。。}
#以编号表示
def read_type_file(filename, entity2id, type2id):
    my_dict = {}

    with open(filename, 'r',encoding="UTF-8") as file:
        for line in file:
            key, value = line.strip().split()
            key = int(entity2id[key])
            value = int(type2id[value])
            if key not in my_dict:
                my_dict[key] = []
            my_dict[key].append(value)

    return my_dict

