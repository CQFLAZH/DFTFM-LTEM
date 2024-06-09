
import json
import numpy as np
import math
import LTEM.Globalvalue as g

def errordetection(testdataset):

    with open("LTEM_output/entities.json",'r') as entity_features:
        entity_dict = json.load(entity_features)

    with open("LTEM_output/relations.json", 'r') as relation_features:
        relation_dict = json.load(relation_features)


    with open("LTEM_output/types.json", 'r') as type_features:
        type_dict = json.load(type_features)
        Type_dict = {}
        for key in type_dict:
            Type_dict[key] = type_dict[key]



    wmatrix = []
    with open("LTEM_output/weights.txt", 'r') as weight_features:
        for row in weight_features:
            wmatrix.append(json.loads(row))



    entitylist = []
    typelist = []

    #

    with open("LTEM/datasets/{}/{}_Entity_Type_test_noise.txt".format(testdataset,testdataset), 'r', encoding="UTF-8") as file:
        for line in file:
            key, value = line.strip().split()
            e1 = np.array(entity_dict[key])
            entitylist.append(e1)
            t1 = np.array(Type_dict[value])
            t2 = np.array(wmatrix)
            typelist.append(np.matmul(t1,t2))


    def distance_between(vector1, vector2):
        # 计算向量之间的欧氏距离（二范数距离）
        if len(vector1) != len(vector2):
            raise ValueError("Unequal length")

        squared_distance = sum((x - y)**2 for x, y in zip(vector1, vector2))
        distance = math.sqrt(squared_distance)
        return distance

    dist_list = {}
    for i in range(len(entitylist)):
        dist = distance_between(entitylist[i], typelist[i])
        dist_list[i] = dist

    def sort_dict_by_value(dictionary):
        sorted_keys = sorted(dictionary, key=lambda k: dictionary[k], reverse=True)
        return sorted_keys

    sorted_keys = sort_dict_by_value(dist_list)


    print("noisecount   recall_rate   precision")

    if testdataset == "fb15k":
        count_false = 16000
    if testdataset == "yago":
        count_false = 46000
    if testdataset == "dbpedia":
        count_false = 4084


    count_pass = 0
    count = 0
    rec = 0
    recall = 0.05
    pre = 0
    for i in range(len(sorted_keys)):
        count_pass = count_pass+1
        if sorted_keys[i] >= count_false:
            count = count+1
        rec  = count/count_false
        pre  = count/count_pass

        if rec - recall < 0.0005 and rec - recall > -0.0005:
            print(str(count) + '  ' + str(rec) + '  ' + str(pre))
            recall = recall+ 0.05




    # for i in range(int(len(sorted_keys)/1000)):
    #
    #
    #     the_list = sorted_keys[0:(i+1)*1000]
    #
    #     recall_rate = len(the_list)/len(sorted_keys)
    #     if testdataset == "fb15k":
    #         count = sum(1 for num in the_list if num > 16000)
    #     if testdataset == "yago":
    #         count = sum(1 for num in the_list if num > 46000)
    #     precision = count/len(the_list)
    #     print(str(count)+'  '+str(recall_rate)+'  '+str(precision))
    #


if __name__ == '__main__':

    args = g.parser.parse_args()
    print(args.dataset)
    errordetection(args.dataset)











































# def increment_second_element(file_path):
#     try:
#         lines = []
#         with open(file_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 parts = line.strip().split()
#                 if len(parts) == 2 and parts[1].isdigit():
#                     second_element = int(parts[1])
#                     parts[1] = str(second_element - 1)
#                 lines.append(" ".join(parts))
#
#         with open(file_path, 'w', encoding='utf-8') as file:
#             file.write("\n".join(lines))
#
#         print("处理完成并写入文件成功.")
#     except FileNotFoundError:
#         print("文件未找到.")
#     except Exception as e:
#         print("发生错误:", str(e))
#
# # 示例：处理文件中每行的第二个元素数值加1并写入文件
# file_path = "datasets/FB15k-237/types.txt"  # 请将此处的文件路径替换为你要处理的文件路径
# increment_second_element(file_path)
