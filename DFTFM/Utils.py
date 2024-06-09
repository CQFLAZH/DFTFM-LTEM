import json
import torch
from torch import nn
import math
from DFTFM import GlobalValue as g


# input:
# types(path) list
# output:
# specific types : [type1, type2, type3...]
def gen_typesets(types):
    typessets =set()
    for i in range(len(types)):
        sp = set(types[i].split('/')[1:])
        typessets.update(sp)
    return list(typessets)



# input:
# entity list
# output:
# the set of entities
def gen_entitysets(entities):
    return list(set(entities))



# input:
# entity-type pairs(two lists)
# output:
# the weight scores of types of each entity
def gen_typeweights(entities, types):
    alltypes={}
    for i in range(len(entities)):
        if not entities[i] in alltypes:
            alltypes[entities[i]]=[]
        onetypes = alltypes[entities[i]]
        sp = types[i].split('/')
        onetypes.extend(sp)

    # Record all specific types of entities, each type path, separated by ''

    def split_and_merge_array(arr):
        result = []
        temp = []
        for item in arr:
            if item != '':
                temp.append(item)
            else:
                if temp:
                    result.append(temp)
                    temp = []
        if temp:
            result.append(temp)
        return result


    alltypes1 = {}
    for key in alltypes:
        alltypes1[key] = split_and_merge_array(alltypes[key])

    # list the specific types of each type path

    def transtodict(tplist: list):
        typedict = {}
        for Alist in tplist:
            Adict = calcscores(Alist)
            for key in Adict:
                if not key in typedict:
                    typedict[key] = Adict[key]
                    continue
                if typedict[key] >= Adict[key]:
                    typedict[key] = Adict[key]

        return typedict


    alltypes2 = {}
    for key in alltypes1:
        alltypes2[key] = transtodict(alltypes1[key])

    #caculate the type weights scores

    def apply_softmax_to_dict_values(dictionary):

        values = torch.tensor(list(dictionary.values()), dtype=torch.float)

        softmax = nn.Softmax(dim=0)

        softmax_values = softmax(values)

        for i, key in enumerate(dictionary.keys()):
            dictionary[key] = softmax_values[i].item()

        return dictionary


    alltypes3 ={}
    for key in alltypes2:
        alltypes3[key] = apply_softmax_to_dict_values(alltypes2[key])
    return alltypes3





# input:
#typeweights: {entity1:{type1: score1, type2: score2}...}
#typesets: [type1, type2, type3...]
#entitysets: [entity1,entity2,entity3...]
# output:
# {entity1:[0,0.5,0.1...], entity2....}
def gen_typevecs(typesets, entitysets, typeweights):

    vecsets = {}
    for i in range(len(entitysets)):
        vecsets[entitysets[i]]=torch.zeros(len(typesets))

    for key in typeweights:
        for key1 in typeweights[key]:
            typeindex = typesets.index(key1)
            vecsets[key][typeindex] = typeweights[key][key1]

    return vecsets



def calcscores(onetype:list):

    type_scores = {}
    Division = 0
    for i in range(len(onetype)):
        Division+=math.exp(i)

    for i in range(len(onetype)):
        type_scores[onetype[i]]=math.exp(i)/Division

    return type_scores



def gen_type_ent_set(filename):
    data_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip('\n').split('\t')
            data1 = line[0]
            data2 = line[1]
            if data2 in data_dict:
                data_dict[data2].add(data1)
            else:
                data_dict[data2] = {data1}
    return data_dict



def calculate_vector_sum(dict1, dict2):
    result_dict = {}
    for key, value_set in dict2.items():
        vector_sum = [0] * len(dict1[next(iter(dict1))])
        count = 0
        for data_key in value_set:
            if data_key in dict1:
                vector_sum = [x + y for x, y in zip(vector_sum, dict1[data_key])]
                count += 1
        if count > 0:
            vector_sum = [x / count for x in vector_sum]
        result_dict[key] = vector_sum
    return result_dict



def WriteFile():
    with open("../DFTFM_output/entity_features.json", 'r') as file:
        dict1 = json.load(file)

    dict2 = gen_type_ent_set("data(Fb15k_demo)/FB15k_Entity_Types/FB15k_Entity_Type_train{}.txt".format(g.errorrate))
    typedict = calculate_vector_sum(dict1, dict2)

    with open('../DFTFM_output/type_features.json', 'w') as _file:
        json.dump(typedict, _file)


