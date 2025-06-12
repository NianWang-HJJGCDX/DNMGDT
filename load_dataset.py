
import torch.utils.data as data
from glob2 import glob
import math
import torchvision.transforms as transforms
import random
import torch
from PIL import Image
import torchvision.transforms.functional as TF

class SemiDatasets(data.Dataset):
    def __init__(self, data_dir_sys, data_dir_real, data_dir_sys_gt=None,data_real_gt=None,data_dir_sys_hazy2=None, data_dir_sys_gt2= None,test_hazy_dir=None, test_gt_dir=None,
                 istrain=True, flip=True):
        super(SemiDatasets, self).__init__()
        self.scale_size = 320
        self.size = 256
        self.hazy_img_list = glob(data_dir_sys + '*')
        self.unhazy_img_list = glob(data_dir_real + '*')
        self.clean_img_list = []
        self.unclean_img_list_NLD = []
        self.unclean_img_list_IDRLP = []
        self.unclean_img_list_IDE = []
        self.unclean_img_list_DCP = []
        self.unclean_img_list_FGT = [] # the fused result of multi-priors
        self.isTrain = istrain
        self.Flip = flip
        if self.isTrain:
            if self.isTrain:
                for img in self.unhazy_img_list:
                    img_name_NLD = img.replace('hazy','NLD')
                    img_name_DCP = img.replace('hazy', 'DCP')
                    img_name_IDRLP = img.replace('hazy', 'IDRLP')
                    img_name_IDE = img.replace('hazy', 'IDE')
                    img_name_FGT = img.replace('hazy', 'FGT')
                    self.unclean_img_list_NLD.append(img_name_NLD)
                    self.unclean_img_list_DCP.append(img_name_DCP)
                    self.unclean_img_list_IDRLP.append(img_name_IDRLP)
                    self.unclean_img_list_IDE.append(img_name_IDE)
                    self.unclean_img_list_FGT.append(img_name_FGT)
        else:
            self.unhazy_img_list = glob(data_dir_sys + '*.jpg')
            for img in self.unhazy_img_list:
                img_name = img.split('\\')[-1]
                gt_name = img_name.replace('hazy','GT')
                gt_path = data_dir_sys_gt + gt_name
                self.clean_img_list.append(gt_path)


    def name(self):
        return 'SemiDatasets72'

    def initialize(self, opt):
        pass
    def imgae_entropy(self,img):
        gray = img.convert('L')
        hist = gray.histogram()
        tot_pixels = sum(hist)
        prob = [float(h)/tot_pixels for h in hist]
        entropy = 0
        for p in prob:
            if p != 0:
                entropy =  entropy-p * math.log2(p)
        return entropy
    def imgae_saturation(self,img):
        gray = img.convert('L')
        hist = gray.histogram()
        tot_pixels = sum(hist)
        prob = [float(h)/tot_pixels for h in hist]
        entropy = 0
        for p in prob:
            if p != 0:
                entropy =  entropy-p * math.log2(p)
        return entropy

    def __getitem__(self, index):
        if self.isTrain:
            index1 = torch.randint(0,len(self.unhazy_img_list),(1,)).item()
            index2 = torch.randint(0, len(self.hazy_img_list), (1,)).item()
            src = Image.open(self.hazy_img_list[index2]).convert('RGB')
            w, h = src.size
            gt_img = src.crop((int(w / 2), 0, w, h))
            hazy_img = src.crop((0, 0, int(w / 2), h))
            un_img = Image.open(self.unhazy_img_list[index1]).convert('RGB')
            un_clean_img_NLD = Image.open(self.unclean_img_list_NLD[index1]).convert('RGB')
            un_clean_img_DCP = Image.open(self.unclean_img_list_DCP[index1]).convert('RGB')
            un_clean_img_IDRLP = Image.open(self.unclean_img_list_IDRLP[index1]).convert('RGB')
            un_clean_img_IDE = Image.open(self.unclean_img_list_IDE[index1]).convert('RGB')
            un_clean_img_FGT = Image.open(self.unclean_img_list_FGT[index1]).convert('RGB')

            rand_hor = random.randint(0, 1)

            hazy_img = transforms.RandomHorizontalFlip(rand_hor)(hazy_img)
            gt_img = transforms.RandomHorizontalFlip(rand_hor)(gt_img)
            un_img = transforms.RandomHorizontalFlip(rand_hor)(un_img)

            un_clean_img_NLD = transforms.RandomHorizontalFlip(rand_hor)(un_clean_img_NLD)
            un_clean_img_DCP = transforms.RandomHorizontalFlip(rand_hor)(un_clean_img_DCP)
            un_clean_img_IDRLP = transforms.RandomHorizontalFlip(rand_hor)(un_clean_img_IDRLP)
            un_clean_img_IDE = transforms.RandomHorizontalFlip(rand_hor)(un_clean_img_IDE)
            un_clean_img_FGT = transforms.RandomHorizontalFlip(rand_hor)(un_clean_img_FGT)

            hazy_img = hazy_img.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            gt_img = gt_img.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            un_img = un_img.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            un_clean_img_NLD = un_clean_img_NLD.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            un_clean_img_IDRLP = un_clean_img_IDRLP.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            un_clean_img_IDE = un_clean_img_IDE.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            un_clean_img_FGT = un_clean_img_FGT.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            un_clean_img_DCP = un_clean_img_DCP.resize((self.scale_size, self.scale_size), Image.BICUBIC)


            i, j, h, w = transforms.RandomCrop.get_params(hazy_img, output_size=(self.size, self.size))

            gt_img1 = TF.crop(gt_img, i, j, h, w)
            hazy_img1 = TF.crop(hazy_img, i, j, h, w)
            un_clean_img1_NLD = TF.crop(un_clean_img_NLD, i, j, h, w)
            un_clean_img1_IDRLP = TF.crop(un_clean_img_IDRLP, i, j, h, w)
            un_clean_img1_DCP = TF.crop(un_clean_img_DCP, i, j, h, w)
            un_clean_img1_IDE = TF.crop(un_clean_img_IDE, i, j, h, w)
            un_clean_img1_FGT = TF.crop(un_clean_img_FGT, i, j, h, w)
            un_img_1 = TF.crop(un_img, i, j, h, w)


            e_NLD = self.imgae_entropy(un_clean_img1_NLD)
            e_DCP = self.imgae_entropy(un_clean_img1_DCP)
            e_BCCR = self.imgae_entropy(un_clean_img1_IDRLP)
            e_IDE = self.imgae_entropy(un_clean_img1_IDE)
            e_hazy = self.imgae_entropy(un_img)

            hazy_img1 = transforms.ToTensor()(hazy_img1)
            gt_img1 = transforms.ToTensor()(gt_img1)
            un_img_1 = transforms.ToTensor()(un_img_1)

            un_clean_img1_NLD = transforms.ToTensor()(un_clean_img1_NLD)
            un_clean_img1_IDRLP = transforms.ToTensor()(un_clean_img1_IDRLP)
            un_clean_img1_DCP = transforms.ToTensor()(un_clean_img1_DCP)
            un_clean_img1_IDE = transforms.ToTensor()(un_clean_img1_IDE)
            un_clean_img1_FGT = transforms.ToTensor()(un_clean_img1_FGT)

            hazy_img1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(hazy_img1)
            gt_img1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(gt_img1)
            un_img_1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_img_1)
            un_clean_img1_NLD = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_img1_NLD)
            un_clean_img1_IDRLP = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_img1_IDRLP)
            un_clean_img1_IDE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_img1_IDE)
            un_clean_img1_FGT = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_img1_FGT)
            un_clean_img1_DCP = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_img1_DCP)
            return hazy_img1,gt_img1,un_img_1,un_clean_img1_NLD,un_clean_img1_IDRLP,un_clean_img1_DCP,un_clean_img1_IDE,un_clean_img1_FGT,e_NLD,e_BCCR,e_DCP,e_IDE,e_hazy
        else:
            hazy_img = Image.open(self.unhazy_img_list[index]).convert('RGB')
            clean_img = Image.open(self.clean_img_list[index]).convert('RGB')
            w_s = 512
            h_s = 512
            hazy_img = hazy_img.resize((w_s, h_s), Image.BICUBIC)
            clean_img = clean_img.resize((w_s, h_s), Image.BICUBIC)
            transform_list = []
            transform_list += [transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]
            trans = transforms.Compose(transform_list)

            return trans(hazy_img), trans(clean_img)

    def __len__(self):
        return len(self.hazy_img_list)