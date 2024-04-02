import numpy as np
import torch
import csv
import json
from pathlib import Path
from PIL import Image
import multiprocessing as mp
import faiss

import open_clip
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def check_dir(dir_path):
    """
    检查文件夹路径是否存在，不存在则创建

    Args:
        dir_path (str): 待检查的文件夹路径
    """
    if not dir_path.exists():
        try:
            dir_path.mkdir(parents=True)
        except Exception as e:
            raise e

def load_json(json_path):
    """
    以只读的方式打开json文件

    Args:
        config_path: json文件路径

    Returns:
        A dictionary

    """
    with open(json_path, 'r', encoding='UTF-8') as f:
        return json.load(f)
    
def save_json(save_path, data):
    """
    Saves the data to a file with the given filename in the given path

    Args:
        :param save_path: The path to the folder where you want to save the file
        :param filename: The name of the file to save
        :param data: The data to be saved

    """
    with open(save_path, 'w', encoding='UTF-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def is_str_Length_valid(str_list) -> bool:
    """
    判断字符串长度是否超过77

    Args:
        str_list (list): 字符串列表

    Returns:
        bool: 是否超过77
    """
    try:
        for str in str_list:
            tokenizer(str)
        return True
    except:
        return False

def prepare_img_caption_pairs(NFT_name,  img_caption_dict, img_base_path="/ShuXun_SSD/") -> list:
    """
    准备图片-描述对
    返回两个列表，一个是图片路径列表，一个是描述列表
    """
    # 找到test_img_caption_dict 的key中包含NFT_name的所有图片名称
    img_path_list = [img_name for img_name in img_caption_dict.keys() if NFT_name in img_name]
    # 拼凑出图片的路径
    target_img_path_list = [img_base_path + img_name for img_name in img_path_list]
    # 提取描述列表
    target_des_list = [img_caption_dict[img_name] for img_name in img_path_list]
    return target_img_path_list, target_des_list


def prepare_img_caption_pairs_for_NFT1000(NFT_name, img_base_path="/ShuXun_SSD/") -> tuple:
    """
    准备图片-描述对
    返回两个列表，一个是图片路径列表，一个是描述列表
    """
    # 找到test_img_caption_dict 的key中包含NFT_name的所有图片名称
    NFT_caption_path = Path(img_base_path).joinpath("NFT1000", NFT_name, "caption", "_caption_dict.json")
    caption_dict = load_json(NFT_caption_path).get("caption_dict")
    img_base_path = Path(img_base_path).joinpath("NFT1000", NFT_name, "img/")
    target_img_path_list = [img_base_path.joinpath(img_name) for img_name in caption_dict.keys()]
    target_des_list = list(caption_dict.values())
    print(f"图片数量：{len(target_img_path_list)}，描述数量：{len(target_des_list)}")

    return target_img_path_list, target_des_list

def tensorlize_imgs(img_path_list) -> torch.Tensor:
    """
    使用模型提取图片特征，返回图片特征向量列表

    Args:
        img_path_list (list): 图片路径列表

    Returns:
        torch.Tensor: 图片特征向量列表
    """

    images = []
    for img_path in img_path_list:

        image = Image.open(img_path).convert("RGB")
            # 首先将图片预处理成模型需要的格式
        images.append(preprocess(image))
        # 把图片加载进cuda中
    image_input = torch.tensor(np.stack(images)).cuda(device=device)
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        return image_features
            
def tensorlize_texts(text_tokens_list) -> torch.Tensor:
    """
    使用模型提取单句文本特征，返回文本特征向量列表

    Args:
        text_tokens_list (list): 文本列表

    Returns:
        torch.Tensor: 文本特征向量列表
    """
    text_tokens = tokenizer(text_tokens_list).cuda(device=device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        return text_features

def load_features(device, imgTensor_path, desTensor_path) -> tuple:
    """
    加载图片和描述的tensor 向量到指定cuda中

    Args:
        device (str): cuda
        img_path (str): 图片tensor路径
        des_path (str): 描述的tensor路径

    Returns:
        torch.Tensor: 图片特征向量
    """
    # 加载json文件
    # img_features = torch.load(imgTensor_path, map_location=device)
    # caption_features = torch.load(desTensor_path, map_location=device)
    img_features = torch.load(imgTensor_path)
    caption_features = torch.load(desTensor_path)
    return img_features, caption_features

def load_des_tensor(device, desTensor_path):
    """
    加载描述的tensor 向量到指定cuda中

    Args:
        device (str): cuda
        img_path (str): 描述的tensor路径

    Returns:
        torch.Tensor: 图片特征向量
    """
    # 加载json文件
    NFT_tensor_data = load_json(desTensor_path)
    des_features = NFT_tensor_data['des_tensors'] 
    des_tensors = torch.tensor(des_features).to(device)
    # 将所有张量以第二个维度作为基准堆叠在一起
    des_tensors = torch.stack(tuple(des_tensors), dim=1)
    return des_tensors

def calculate_cosine_similarity_topk(img_features, caption_features, k = 10) -> tuple:
    """
    计算图片特征和描述特征的余弦相似度，并返回topk的结果

    Args:
        img_features (torch.tensor): 图像特征向量
        des_features (torch.tensor): 描述特征向量
        k (int, optional): 前k位结果. Defaults to 10.

    Returns:
        tuple: (topk的相似度，topk的索引)
    """
    # 对每一个特征向量进行归一化，使其范数为1
    img_features /= img_features.norm(dim=-1, keepdim=True)

    # 归一化描述特征
    caption_features /= caption_features.norm(dim=-1, keepdim=True)
    # similarity = des_features.cpu().numpy() @ img_features.cpu().numpy().T
    # 每个图像向量都与文本向量计算余弦相似度
    text_probs = (100.0 * img_features @ caption_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(k, dim=-1)
    return top_probs, top_labels


def tensorlize_valid_subsentence(model, text) -> torch.Tensor:
    """
    截取有效的子句，然后求特征向量值

    Args:
        model (CLIP): 使用的 CLIP 模型。
        text (str): 输入的文本。

    Returns:
        torch.Tensor: 文本特征向量。
    """
    text_tensor = None
    # 标记为False时，表示该句子无法被模型处理，需要进行拆分
    flag = False
    words = text.split()
    text_length = len(words)
    while not flag:
        try:
            text_tensor = tensorlize_texts(model, text)
            flag = True
        except:
            text_length -= 1
            text = ' '.join(words[:text_length])
    return text_tensor

def legalize_text(text) -> str:
    """
    截取有效长度的句子

    Args:
        text (str): 输入的文本。

    Returns:
        str: 截断之后的文本。
    """
    # 标记为False时，表示该句子无法被模型处理，需要进行拆分
    flag = False
    words = text.split()
    text_length = len(words)
    while not flag:
        try:
            tokenizer(text)
            flag = True
        except:
            text_length -= 1
            text = ' '.join(words[:text_length])
    return text

def handle_long_texts(model, text_list) -> list:
    """
    处理长文本，将长文本分块，然后对每块文本进行向量化，最后将这些向量平均。

    Args:
        model (CLIP): 使用的 CLIP 模型。
        text_list (list): 输入的文本列表。
        window_size (int): 窗口宽度
        step_size (int): 窗口移动的步长

    Returns:
        list: 文本特征向量列表。
    """
    # 先将长句子截断为有效短句子
    legal_text_list = list(map(legalize_text, text_list))
    text_features = tensorlize_texts(model, legal_text_list)
    return text_features


def load_des_tensor(device, desTensor_path):
    """
    加载描述的tensor 向量到指定cuda中

    Args:
        device (str): cuda
        img_path (str): 描述的tensor路径

    Returns:
        torch.Tensor: 图片特征向量
    """
    # 加载json文件
    NFT_tensor_data = load_json(desTensor_path)
    des_features = NFT_tensor_data['des_tensors']
    des_tensors = torch.tensor(des_features).to(device)
    return des_tensors

def create_gpu_index_use_n_gpu(feature_tensors, gpus=[1]):
    # 构建索引，这里我们选用暴力检索的方法FlatL2为例，L2代表构建的index采用的相似度度量方法为L2范数，即欧氏距离
    # 这里必须传入一个向量的维度，创建一个空的索引
    feature_dim = feature_tensors.shape[1]
    index_flat = faiss.IndexFlatL2(feature_dim)  
    # gpus用于指定使用的gpu号
    gpu_index_flat = faiss.index_cpu_to_gpus_list(index_flat, gpus=gpus)
    gpu_index_flat.add(feature_tensors)   # 把向量数据加入索引
    return gpu_index_flat

def search_top_k(index_flat, query_tensor, top_k=10):
    _, index_matrix = index_flat.search(query_tensor, top_k)  # 实际搜索
    return index_matrix

def retrieve_within_all_collections_faiss(index_matrix, save_path, csv_name):
        # 定义表头文件
    headers = ['collection', 'top1', 'top5', 'top10', 'top15', 'top20', 'top30', 'top40', 'top50']
    # 加载特征向量
    data = ["all_collections"]
    top_K = [1, 5, 10, 15, 20, 30, 40, 50]
    # 创建一个全零二维矩阵，用于存储检索结果
    result = np.zeros((index_matrix.shape[0], len(top_K)))
    for i in range(index_matrix.shape[0]):
        for j in range(len(top_K)):
            top_k = top_K[j]
            # 如果行号在top_k中，就记为1，否则记为0
            if i in index_matrix[i, :top_k]:
                result[i, j] = 1

    # 按列求和,结果换算成百分比并保留两位小数
    for i in range(len(top_K)):
        retrieval_result = np.around(np.sum(result[:, i]) / index_matrix.shape[0], decimals=6) * 100
        formatted_result = float("{:.6f}".format(retrieval_result))  # 格式化为字符串并转换回浮点数
        data.append(formatted_result)

    print(data)
    # 将结果写入csv文件
    with open(save_path.joinpath(csv_name),'a+', encoding='UTF-8', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerow(data)
        

def extract_representations(NFT_name_list,  test_NFT_dict_info, tensor_cache_storage_path):
    """
    提取NFT的特征向量

    Args:
        NFT_list (list): 被提取向量的NFT列表
    """
    for nft_name in NFT_name_list:
        print("开始处理：", nft_name, "......")
        # 检查路径是否存在
        target_path = tensor_cache_storage_path.joinpath(nft_name)
        check_dir(target_path)
        # img_path_list, des_list = prepare_img_caption_pairs(nft_name, test_NFT_dict_info)
        img_path_list, des_list = prepare_img_caption_pairs_for_NFT1000(nft_name)

        # 以下措施是为了防止描述过长，导致CUDA out of memory
        # 以3000长度子列表为单位，将img_path_list, des_list分割成多个子列表
        sub_length = 1000
        img_path_list = [img_path_list[i:i + sub_length] for i in range(0, len(img_path_list), sub_length)]
        des_list = [des_list[i:i + sub_length] for i in range(0, len(des_list), sub_length)]

        image_features_CPU = torch.tensor([])
        des_features_CPU = torch.tensor([])
        # 提取图片特征向量
        print(f"开始提{nft_name}取图片特征向量……")
        for index, img_path_sublist in enumerate(img_path_list):
            print(f"开始提取第{index}个子列表的图片特征向量")
            image_features_CPU_item = tensorlize_imgs(img_path_sublist).cpu()
            image_features_CPU = torch.cat((image_features_CPU, image_features_CPU_item), 0)
        print(f"图片特征向量提取完成\n")

        print(f"开始提{nft_name}取描述特征向量……")
        for index, des_sublist in enumerate(des_list):
            print(f"开始提取第{index}个子列表的描述特征向量")
            des_features_CPU_item = tensorlize_texts(des_sublist).cpu()
            des_features_CPU = torch.cat((des_features_CPU, des_features_CPU_item), 0)
        print(f"caption列表特征向量提取完成\n")

        # 将特征向量存为pth文件
        torch.save(image_features_CPU, target_path.joinpath("image_features.pth"))
        torch.save(des_features_CPU, target_path.joinpath("caption_features.pth"))

        print("处理完成：", nft_name, "\n")

def gather_tensor(NFT_name_list, cache_base_path):
    img_tensor_list = torch.tensor([])
    caption_tensor_list = torch.tensor([])
    for NFT_name in NFT_name_list:
        print(f"开始聚合{NFT_name}的特征")
        img_tensor_cache_path = Path(cache_base_path, NFT_name, "image_features.pth")
        caption_tensor_cache_path = Path(cache_base_path, NFT_name, "caption_features.pth")
        img_tensor = torch.load(img_tensor_cache_path)
        caption_tensor = torch.load(caption_tensor_cache_path)
        img_tensor_list = torch.cat([img_tensor_list, img_tensor], dim=0)
        caption_tensor_list = torch.cat([caption_tensor_list, caption_tensor], dim=0)
        print(f"聚合{NFT_name}的特征完成\n")
    print(img_tensor_list.shape)
    print(caption_tensor_list.shape)

    # 存起来
    torch.save(img_tensor_list, Path(cache_base_path, "img_gathered_features.pth"))
    torch.save(caption_tensor_list, Path(cache_base_path, "caption_gathered_features.pth"))
    print("聚合完成")

def retrieve_within_single_collection(NFT_name_list, tensor_cache_storage_path, save_path, csv_name):
    """
    在单个集合中检索

    Args:
        dataset_base_path (Path object): 存放检索结果的目标文件夹
        target_dataset_path (Path object): 
    """

    # 定义表头文件
    headers = ['collection', 'top1', 'top10', 'top20', 'top30', 'top40', 'top50', 'top60', 'top70', 'top80', 'top90', 'top100']
    # 加载特征向量
    data = []
    for nft_name in NFT_name_list:
        # 找到路径下的tensor文件
        img_tensor_path = tensor_cache_storage_path.joinpath(nft_name,"image_features.pth")
        caption_tensor_path = tensor_cache_storage_path.joinpath(nft_name,"caption_features.pth")
        image_features, caption_features = load_features(device, img_tensor_path, caption_tensor_path)
        top_K = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        print("开始计算：", nft_name, "...")
        data_item = [nft_name]
        for k in top_K:
            counter = 0
            _, top_labels = calculate_cosine_similarity_topk(image_features, caption_features, k)
            for index, top_index in enumerate(top_labels):
                if index in top_index:
                    counter += 1
            precision = round(counter / (len(top_labels)) * 100, 2)
            data_item.append(precision)
            print("top", k, ":", precision, "%")
        print(data_item)
        data.append(data_item)
        print("计算完成：", nft_name)
        break

    # 将数据写入csv文件
    with open(save_path.joinpath(f"precision_with_{csv_name}.csv"), 'a+', newline='') as f:
        f_csv = csv.writer(f)
        # 写入表头
        f_csv.writerow(headers)
        # 写入数据
        f_csv.writerows(data)

def retrieve_within_all_collections(NFT_name_list, tensor_cache_storage_path, save_path, csv_name):
# 加载特征向量

    all_image_features = torch.tensor([]).to(device)
    all_caption_features = torch.tensor([]).to(device)
    for nft_name in NFT_name_list:
        # 找到路径下的tensor文件
        img_tensor_path = tensor_cache_storage_path.joinpath(nft_name,"image_features.pth")
        caption_tensor_path = tensor_cache_storage_path.joinpath(nft_name,"caption_features.pth")
        image_features, caption_features = load_features(device, img_tensor_path, caption_tensor_path)

        # 放入all_image_features中
        all_image_features = torch.cat((all_image_features, image_features), 0)
        # 放入all_des_features中
        all_caption_features = torch.cat((all_caption_features, caption_features), 0)

    print(all_image_features.shape)
    print(all_caption_features.shape)

    data = ["all_collections"]

    top_K = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for k in top_K:
        counter = 0
        _, top_labels = calculate_cosine_similarity_topk(all_image_features, all_caption_features, k)
        for index, top_index in enumerate(top_labels):
            if index in top_index:
                counter += 1
        precision = round(counter / len(top_labels) * 100, 2)
        data.append(precision)
        print("top", k, ":", precision, "%")
    # 将数据写入csv文件
    with open(save_path.joinpath(f"precision_with_{csv_name}.csv"), 'a+', newline='') as f:
        f_csv = csv.writer(f)
        # 写入数据
        f_csv.writerows(data)


def retrieve_within_single_collection_faiss(NFT_name_list, tensor_cache_storage_path, save_path, csv_name, gpus=[6,7]):
    """
    在单个集合中检索

    Args:
        dataset_base_path (Path object): 存放检索结果的目标文件夹
        target_dataset_path (Path object): 
    """

    # 定义表头文件
    headers = ['collection', 'item_num', 'dimension',  'top1', 'top5', 'top10', 'top15', 'top20', 'top30', 'top40', 'top50']
    top_K = [1, 5, 10, 15, 20, 30, 40, 50]
    # 加载特征向量
    data = []
    for nft_name in NFT_name_list:
        # 找到路径下的tensor文件
        # print("开始加载：", nft_name, "tensor文件")
        img_tensor_path = tensor_cache_storage_path.joinpath(nft_name,"image_features.pth")
        caption_tensor_path = tensor_cache_storage_path.joinpath(nft_name,"caption_features.pth")
        img_features, caption_features = load_features(device, img_tensor_path, caption_tensor_path)
        img_features = np.array(img_features).astype('float32')     # 数据库向量
        caption_features = np.array(caption_features).astype('float32')      # 查询向量

        # print("开始计算：", nft_name, "...")
        data_item = [nft_name, img_features.shape[0], img_features.shape[1]]
        # print("开始创建索引")
        img_gpu_index_flat = create_gpu_index_use_n_gpu(img_features, gpus=gpus)
        # print("开始检索")
        index_matrix = search_top_k(img_gpu_index_flat, caption_features, top_k=50)
        # print("开始统计检索结果")
        result = np.zeros((index_matrix.shape[0], len(top_K)))
        for i in range(index_matrix.shape[0]):
            for j in range(len(top_K)):
                top_k = top_K[j]
                # 如果行号在top_k中，就记为1，否则记为0
                if i in index_matrix[i, :top_k]:
                    result[i, j] = 1

        # 按列求和,结果换算成百分比并保留两位小数
        for i in range(len(top_K)):
            retrieval_result = np.around(np.sum(result[:, i]) / index_matrix.shape[0], decimals=6) * 100
            formatted_result = float("{:.6f}".format(retrieval_result))  # 格式化为字符串并转换回浮点数
            data_item.append(formatted_result)
        print(data_item)
        data.append(data_item)
        print("计算完成：", nft_name)

    # 将数据写入csv文件
    with open(save_path.joinpath(f"precision_with_{csv_name}.csv"), 'a+', newline='') as f:
        f_csv = csv.writer(f)
        # 写入表头
        f_csv.writerow(headers)
        # 写入数据
        f_csv.writerows(data)

def retrieve_within_all_collections_faiss(NFT_name_list, tensor_cache_storage_path, save_path, csv_name, gpus=[6,7]):
    
    top_K = [1, 5, 10, 15, 20, 30, 40, 50]

    # 判断聚合文件是否存在
    if not Path(tensor_cache_storage_path, "img_gathered_features.pth").exists() or not Path(tensor_cache_storage_path, "caption_gathered_features.pth").exists():
        gather_tensor(NFT_name_list, tensor_cache_storage_path)

    # 加载聚合文件
    img_gathered_features_path = Path(tensor_cache_storage_path, "img_gathered_features.pth")
    caption_gathered_features_path = Path(tensor_cache_storage_path, "caption_gathered_features.pth")
    all_image_features, all_caption_features = load_features(device, img_gathered_features_path, caption_gathered_features_path)
    data = ["all_collections", all_image_features.shape[0], all_image_features.shape[1]]

    img_features = np.array(all_image_features).astype('float32') 
    caption_features = np.array(all_caption_features).astype('float32') 

    # 统计index_matrix中每一行的行号是不是在top_K中，如果在就记为1，不在就记为0, 将结果存为一个二维矩阵，最后二维矩阵每一列的和就是每个top_K的检索结果
    # 创建一个全零二维矩阵，用于存储检索结果
    img_gpu_index_flat = create_gpu_index_use_n_gpu(img_features, gpus=gpus)
    index_matrix = search_top_k(img_gpu_index_flat, caption_features, top_k=50)
    result = np.zeros((index_matrix.shape[0], len(top_K)))
    for i in range(index_matrix.shape[0]):
        for j in range(len(top_K)):
            top_k = top_K[j]
            # 如果行号在top_k中，就记为1，否则记为0
            if i in index_matrix[i, :top_k]:
                result[i, j] = 1

    # 按列求和,结果换算成百分比并保留两位小数
    for i in range(len(top_K)):
        retrieval_result = np.around(np.sum(result[:, i]) / index_matrix.shape[0], decimals=6) * 100
        formatted_result = float("{:.6f}".format(retrieval_result))  # 格式化为字符串并转换回浮点数
        data.append(formatted_result)

    print(data)
    # 将结果写入csv文件
    with open(save_path.joinpath(f"precision_with_{csv_name}.csv"),'a+', encoding='UTF-8', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(data)


if __name__ == "__main__":

    test_NFT_list = [
            "LordSocietyNFT",
            "RugBurn",
            "Dr.Ji",
            "RoaringLeaders",
            "Gooniez Gang",
            "OctoHedz",
            "Meta Bounty Huntress",
            "CyberRonin Haruka",
            "Imps",
            "YOLO Bunny",
            "Wicked Apes",
            "Dead Army Skeleton Klub",
            "RSTLSS ✕ CrypToadz",
            "Aki Story",
            "LuckyManekiNFT",
            "Cool Ape Club",
            "Habbo Avatars",
            "BaoSociety",
            "WaveCatchers",
            "TEST NFT",
            "Dopey Ducklings",
            "Angry Apes Society",
            "0xVampire",
            "Chill Frogs",
            "Shiba Social Club",
            "KevinPunks",
            "Bunny Buddies",
            "T Thugs",
            "Strange Times",
            "Joker Charlie Club Genesis",
            "Space Riders",
            "MetaStonez",
            "Degenz",
            "Rink Rat Ice Club",
            "DizzyDragons",
            "Junglebayapeclub",
            "Cheers UP",
            "Crypto Coven",
            "loomlocknft",
            "Blockchain Bandits",
            "Cosmos",
            "Savage Droids",
            "Kryptoria Alpha Citizens",
            "DeBox Guardians Penguin",
            "GalaxyFightClub",
            "Lora Of Flower Garden",
            "Project Draca",
            "Tokenmon",
            "DormantDragon",
            "Tasty Bones",
            "Fat Rat Mafia",
            "Immortalz",
            "Aneroverse",
            "ZooFrenzToken",
            "Project Godjira Generation 2",
            "Bad Bears",
            "ChiptoPunks",
            "NeoTokyoPunks",
            "Elderly Ape Retirement Club",
            "Rainbow Cats",
            "Social BEES University",
            "0xAzuki",
            "The Royal Cubs",
            "Permies",
            "CryptoPhunksV2",
            "Rareland",
            "Forever Fomo Duck Squad",
            "FVCK_AVATAR",
            "Yin Yang Gang",
            "JPunks OG-Rex",
            "CryptoMutts",
            "CyberTurtles",
            "OkayBearsYachtClub",
            "VoltedDragonsSailorsClub",
            "Ghost Boy",
            "Project Shura",
            "HUXLEY Robots",
            "SpriteClub",
            "HypnoDuckzGenesis",
            "Sipher INU",
            "Dogs Unchained",
            "Space Punks",
            "Yakuza Inc.",
            "Sad Girls Bar",
            "Bapes Clan",
            "J48BAFORMS",
            "Bit Monsters",
            "Nudie Community",
            "DigiDaigakuSpirits",
            "TheWhitelist",
            "Unemployables",
            "Long Neckie Ladies",
            "Iron Paw Gang",
            "Ghost Buddy NFT",
            "CakedApes",
            "Moshi Mochi",
            "Avastar",
            "ForgottenRunesWarriorsGuild",
            "Phoenixes",
            "GalaXY Kats",
            "Goat Soup",
            "Coalition Crew 2.0",
            "ThePicaroons",
            "ALTAVA Second Skin Metamorphosis",
            "alinft-official",
            "Ape Reunion",
            "ShadesOfYou",
            "SmallBrosNFT Official",
            "0bits",
            "Monfters Club",
            "Ghidorah Godz",
            "Superlative Secret Society",
            "ShinseiGalverse",
            "Based Fish Mafia",
            "Raccoon Mafia",
            "Super Creators By IAC",
            "DeadHeads",
            "RichKids",
            "Sad Bots",
            "Dippies",
            "DSC E_MATES 4 DA NEXT LEVEL",
            "IROIRO",
            "The Jims",
            "Hor1zon",
            "Japanese Born Ape Society",
            "Lazy Bunny NFT",
            "Bears Deluxe",
            "Okay Duck Yacht Club",
            "Mutant Hounds",
            "Rebel Society",
            "Gazer",
            "JunkYardDogs",
            "GoldSilverPirates",
            "Haki",
            "Bad Kids Alley",
            "COOLDOGS",
            "Ethlizards",
            "Super Puma",
            "Genzee",
            "Bufficorn Buidl Brigade",
            "KumaVerse",
            "The Divine Order Of the Zodiac",
            "AI Rein",
            "Choadz",
            "BlockchainBikers",
            "HOPE",
            "Tokyo Alternative Girls",
            "Divine Anarchy",
            "APE DAO REMIX!",
            "Stoner Ape Club"
        ]
    
    ############################################################################################################################################################################

    # # 统计CLIP_ViT_B_32的finetune三种版本在NFT1000_mini的推理结果
    # img_caption_dict_path = "/ShuXun_SSD/NFT1000/_index/NFT1000_mini_img_caption_dict.json"
    search_result_storage_path = Path("/mnt/main/baiwm/Img_Retrieval/Data/search_results_faiss/NFT1000_mini")

    var_tuple_list = [
        ("/mnt/main/baiwm/Img_Retrieval/models/2024_03_30-20_34_43-model_ViT-L-14-NFT1000_mini_finetuned_all_compontents_dynamic_mask_hardCase_p0_5_ckp10/checkpoints/epoch_10.pt", "ViT-L-14", Path("/mnt/main/baiwm/Img_Retrieval/Data/Tensor_cache/NFT1000_mini/CLIP_ViT_L_14_NFT1000_mini_finetuned_all_compontents_dynamic_mask_hardCase_p0_5_ckp10"), "NFT1000_mini_CLIP_ViT_L_14_NFT1000_mini_all_compontents_dynamic_mask_hardCase_finetuned_p0_5_ckp10"),
        ("/mnt/main/baiwm/Img_Retrieval/models/2024_03_31-11_09_04-model_ViT-L-14-META-NFT1000_mini_finetuned_all_compontents_dynamic_mask_hardCase_p0_5_ckp10/checkpoints/epoch_10.pt", "ViT-L-14", Path("/mnt/main/baiwm/Img_Retrieval/Data/Tensor_cache/NFT1000_mini/Meta_CLIP_ViT_L_14_NFT1000_mini_finetuned_all_compontents_dynamic_mask_hardCase_p0_5_ckp10"), "NFT1000_mini_Meta_CLIP_ViT_L_14_NFT1000_mini_all_compontents_dynamic_mask_hardCase_finetuned_p0_5_ckp10"),

    ]

    ############################################################################################################################################################################

    

    for var_tuple in var_tuple_list:
        _, _, tensor_cache_storage_path, csv_name = var_tuple

        # data_dict = load_json(img_caption_dict_path)
        # test_NFT_name_list = data_dict["project_name_list"].get("test_list")
        # test_NFT_dict_info = data_dict.get("test_dict")
        # print(f"开始使用{csv_name}提取特征的向量")
        # extract_representations(test_NFT_list, test_NFT_dict_info, tensor_cache_storage_path)
        gpu_list = [5,6,7]
        retrieve_within_single_collection_faiss(test_NFT_list, tensor_cache_storage_path, search_result_storage_path, csv_name=csv_name, gpus=gpu_list)
        retrieve_within_all_collections_faiss(test_NFT_list, tensor_cache_storage_path, search_result_storage_path, csv_name=csv_name, gpus=gpu_list)
