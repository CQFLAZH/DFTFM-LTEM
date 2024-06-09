from DFTFM import GlobalValue as g


class TrainRun:
    def __init__(self):


        self.relation2id = {}
        self.entity2id = {}
        self.type2id = {}
        self.id2relation ={}
        self.id2entity = {}
        self.id2type = {}
        # {int:int}

        self.fb_HT = []
        self.fb_LT = []
        #triplets info

        self.et2type = {}
        #entity-type info




        self.entity_num = self.Read_data("data(Fb15k_demo)/FB15k/entity2id.txt", self.entity2id, self.id2entity)
        self.relation_num = self.Read_data("data(Fb15k_demo)/FB15k/relation2id.txt", self.relation2id, self.id2relation)
        self.type_num = self.Read_data("data(Fb15k_demo)/FB15k/type2id.txt", self.type2id, self.id2type)



        with open("data(Fb15k_demo)/FB15k_Entity_Types/FB15k_Entity_Type_train{}.txt".format(g.errorrate), 'r', encoding='utf-8') as reader1:
            res1=reader1.readlines()
            for line in res1:
                sp = line.strip('\n').split('\t')
                heade = sp[0]
                tailet = sp[1]
                self.add(heade, tailet)

        print("entity_number = %s", self.entity_num)
        print("relation_number = %s", self.relation_num)
        print("type_number = %s", self.type_num)



    def Read_data(self, file_name: str, data2id: dict, id2data: dict):
        count= 0
        with open(file_name, 'r', encoding='utf-8') as f:
            res=f.readlines()
            for line in res:
                sp = line.strip().split()
                data2id[sp[0]]=int(sp[1])
                id2data[int(sp[1])]=sp[0]
                count+=1
        return count



    def add(self, head, tail):
        self.fb_HT.append(head)
        self.fb_LT.append(tail)

        key = self.entity2id[head]
        if not key in self.et2type:
            self.et2type[key] = set()
        type_set = self.et2type[key]
        type_set.add(self.type2id[tail])























