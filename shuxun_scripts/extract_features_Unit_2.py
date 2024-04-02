import os
CUDA_TO_USE = "4,5,6,7"
CUDA_NUM = len(CUDA_TO_USE.split(","))
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_TO_USE

import numpy as np
import torch
import csv
import json
from pathlib import Path
from PIL import Image
import torch.multiprocessing as mp
import open_clip


def divide_list_into_sublists(input_list, sub_num):
    """
    将列表划分成sub_num个子列表

    Args:
        target_list (list): 待划分的列表
        sub_num (int): 子列表的数量

    Returns:
        list: 划分好的子列表
    """
    target_list = []
    sub_len = len(input_list) // sub_num
    for i in range(sub_num):
        if i == sub_num - 1:
            target_list.append(input_list[i * sub_len:]) 
        else:
            target_list.append(input_list[i * sub_len: (i + 1) * sub_len])
    
    return target_list


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
    img_features = torch.load(imgTensor_path, map_location=device)
    caption_features = torch.load(desTensor_path, map_location=device)
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


def extract_representations(NFT_name_list, tensor_cache_storage_path):
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
        img_path_list, des_list = prepare_img_caption_pairs_for_NFT1000(nft_name)

        # 以下措施是为了防止描述过长，导致CUDA out of memory
        # 以3000长度子列表为单位，将img_path_list, des_list分割成多个子列表
        sub_length = 5000
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

        for index, des_sublist in enumerate(des_list):
            print(f"开始提取第{index}个子列表的描述特征向量")
            des_features_CPU_item = tensorlize_texts(des_sublist).cpu()
            des_features_CPU = torch.cat((des_features_CPU, des_features_CPU_item), 0)
        print(f"caption列表特征向量提取完成\n")

        # 将特征向量存为pth文件
        torch.save(image_features_CPU, target_path.joinpath("image_features.pth"))
        torch.save(des_features_CPU, target_path.joinpath("caption_features.pth"))

        print("处理完成：", nft_name, "\n")


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

class NFT1000_Extractor():
    def __init__(self, GPU_ID, model_path, tokenizer_type, test_NFT_list, tensor_cache_storage_path, NFT_base_path):
        self.NFT_base_path = NFT_base_path
        self.GPU_ID = GPU_ID
        self.model_path = model_path
        self.tokenizer_type = tokenizer_type
        self.test_NFT_list = test_NFT_list
        self.tensor_cache_storage_path = tensor_cache_storage_path
        self.tensor_cache_storage_path.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(f"cuda:{str(GPU_ID)}" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.tokenizer_type, pretrained=self.model_path, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(self.tokenizer_type)
        self.model.to(self.device)
        print(f"模型已加载到{self.device}")

    def tensorlize_imgs(self, img_path_list) -> torch.Tensor:
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
            images.append(self.preprocess(image))
            # 把图片加载进cuda中
        image_input = torch.tensor(np.stack(images)).cuda(device=self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
            return image_features
                
    def tensorlize_texts(self, text_tokens_list) -> torch.Tensor:
        """
        使用模型提取单句文本特征，返回文本特征向量列表

        Args:
            text_tokens_list (list): 文本列表

        Returns:
            torch.Tensor: 文本特征向量列表
        """
        text_tokens = self.tokenizer(text_tokens_list).cuda(device=self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
            return text_features

    def prepare_img_caption_pairs(self, NFT_name) -> tuple:
        """
        准备图片-描述对
        返回两个列表，一个是图片路径列表，一个是描述列表
        """

        # 找到test_img_caption_dict 的key中包含NFT_name的所有图片名称
        NFT_caption_path = self.NFT_base_path.joinpath(NFT_name, "caption", "_caption_dict.json")

        caption_dict = load_json(NFT_caption_path).get("caption_dict")

        img_base_path = self.NFT_base_path.joinpath(NFT_name, "img/")
        target_img_path_list = [img_base_path.joinpath(img_name) for img_name in caption_dict.keys()]
        target_des_list = list(caption_dict.values())

        return target_img_path_list, target_des_list


    def extract_representations(self):
        """
        提取NFT的特征向量

        Args:
            NFT_list (list): 被提取向量的NFT列表
        """
        for nft_name in self.test_NFT_list:
            print("开始处理：", nft_name, "......")
            # 检查路径是否存在
            target_path = self.tensor_cache_storage_path.joinpath(nft_name)
            check_dir(target_path)
            img_path_list, des_list = self.prepare_img_caption_pairs(nft_name)

            # 以下措施是为了防止描述过长，导致CUDA out of memory
            # 以特定长度子列表为单位，将img_path_list, des_list分割成多个子列表
            sub_length = 1000
            img_path_list = [img_path_list[i:i + sub_length] for i in range(0, len(img_path_list), sub_length)]
            des_list = [des_list[i:i + sub_length] for i in range(0, len(des_list), sub_length)]

            image_features_CPU = torch.tensor([])
            des_features_CPU = torch.tensor([])

            
            # 提取图片特征向量
            for index, img_path_sublist in enumerate(img_path_list):
                print(f"开始提取{nft_name}第{index}个子列表的图片特征向量")
                image_features_CPU_item = self.tensorlize_imgs(img_path_sublist).cpu()
                image_features_CPU = torch.cat((image_features_CPU, image_features_CPU_item), 0)
            print(f"图片特征向量提取完成\n")

            # 提取描述特征向量
            for index, des_sublist in enumerate(des_list):
                print(f"开始提{nft_name}取第{index}个子列表的描述特征向量")
                des_features_CPU_item = self.tensorlize_texts(des_sublist).cpu()
                des_features_CPU = torch.cat((des_features_CPU, des_features_CPU_item), 0)
            print(f"caption列表特征向量提取完成\n")

            # 将特征向量存为pth文件
            torch.save(image_features_CPU, target_path.joinpath("image_features.pth"))
            torch.save(des_features_CPU, target_path.joinpath("caption_features.pth"))

            print("处理完成：", nft_name, "\n")

class NFT1000_mini_Extractor(NFT1000_Extractor):
    def __init__(self, GPU_ID, model_path, tokenizer_type, test_NFT_list, tensor_cache_storage_path, NFT_base_path, img_caption_dict):
        super().__init__(GPU_ID, model_path, tokenizer_type, test_NFT_list, tensor_cache_storage_path, NFT_base_path)
        self.img_caption_dict = img_caption_dict

    def prepare_img_caption_pairs(self, NFT_name) -> tuple:
        """
        准备图片-描述对
        返回两个列表，一个是图片路径列表，一个是描述列表
        """
        # 找到test_img_caption_dict 的key中包含NFT_name的所有图片名称
        img_path_list = [img_name for img_name in self.img_caption_dict.keys() if NFT_name in img_name]
        # 拼凑出图片的路径
        target_img_path_list = [self.NFT_base_path.joinpath(img_name) for img_name in img_path_list]
        # 提取描述列表
        target_des_list = [self.img_caption_dict[img_name] for img_name in img_path_list]
        return target_img_path_list, target_des_list


def NFT1000_worker(i, model_path, tokenizer_type, test_NFT_list, tensor_cache_storage_path, NFT_base_path):
    extractor = NFT1000_Extractor(i, model_path, tokenizer_type, test_NFT_list, tensor_cache_storage_path)
    extractor.extract_representations()

def NFT1000_mini_worker(i, model_path, tokenizer_type, test_NFT_list, tensor_cache_storage_path, NFT_base_path, img_caption_dict):
    extractor = NFT1000_mini_Extractor(i, model_path, tokenizer_type, test_NFT_list, tensor_cache_storage_path, NFT_base_path, img_caption_dict)
    extractor.extract_representations()


if __name__ == "__main__":

    NFT_base_path   = Path("/mnt/main/baiwm/Img_Retrieval/DataSet")
    NFT1000_mini_img_caption_dict_path = Path("/mnt/main/baiwm/Img_Retrieval/DataSet/NFT1000/_index/NFT1000_mini_img_caption_dict.json")
    NFT1000_mini_img_caption_dict = load_json(NFT1000_mini_img_caption_dict_path)
    img_caption_dict = NFT1000_mini_img_caption_dict.get("test_dict")
    # test_list = NFT1000_mini_img_caption_dict.get("project_name_list").get("test_list")
    test_list = [
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

    test_list = divide_list_into_sublists(test_list, CUDA_NUM)

    var_tuple_list = [
        ("/mnt/main/baiwm/Img_Retrieval/models/2024_03_31-11_09_04-model_ViT-L-14-META-NFT1000_mini_finetuned_all_compontents_dynamic_mask_hardCase_p0_5_ckp10/checkpoints/epoch_16.pt", "ViT-L-14", Path("/mnt/main/baiwm/Img_Retrieval/Data/Tensor_cache/NFT1000_mini/META_CLIP_ViT_L_14_NFT1000_mini_finetuned_all_compontents_dynamic_mask_hardCase_p0_5_ckp10"), "NFT1000_mini_META_CLIP_ViT_L_14_NFT1000_mini_all_compontents_dynamic_mask_hardCase_finetuned_p0_5_ckp10"),
    ]
    # # 提取NFT1000特征向量
    # for var_tuple in var_tuple_list:

    #     model_path, tokenizer_type, tensor_cache_storage_path, csv_name = var_tuple
        
    #     with mp.Pool(CUDA_NUM) as pool:
    #         for i in range(CUDA_NUM):
    #             pool.apply_async(NFT1000_worker, args=(i, model_path, tokenizer_type, test_list[i], tensor_cache_storage_path, NFT_base_path))
    #         pool.close()
    #         pool.join()

    # 提取NFT1000_mini特征向量
    for var_tuple in var_tuple_list:

        model_path, tokenizer_type, tensor_cache_storage_path, csv_name = var_tuple
        
        with mp.Pool(CUDA_NUM) as pool:
            for i in range(CUDA_NUM):
                pool.apply_async(NFT1000_mini_worker, args=(i, model_path, tokenizer_type, test_list[i], tensor_cache_storage_path, NFT_base_path, img_caption_dict))
            pool.close()
            pool.join()

    print(f"{csv_name}特征向量提取完成\n")



