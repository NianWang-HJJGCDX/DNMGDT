
import torch.utils.data as data
from glob2 import glob
import math
import torchvision.transforms as transforms
import random
import torch
from PIL import Image,ImageOps
import torchvision.transforms.functional as TF

class SemiDatasets72(data.Dataset):
    def __init__(self, data_dir_sys, data_dir_real, data_dir_sys_gt=None,data_real_gt=None,data_dir_sys_hazy2=None, data_dir_sys_gt2= None,test_hazy_dir=None, test_gt_dir=None,
                 istrain=True, flip=True):
        super(SemiDatasets72, self).__init__()
        self.scale_size = 320
        self.size = 256
        self.hazy_img_list = glob(data_dir_sys + '*')
        self.unhazy_img_list = glob(data_dir_real + '*')

        self.un_len = self.unhazy_img_list.__len__()
        self.clean_img_list = []
        self.unclean_img_list_NLD = []
        self.unclean_img_list_BCCR = []
        self.unclean_img_list_BCP = []
        self.unclean_img_list_IDE = []
        self.unclean_img_list_DCP = []
        self.unclean_img_list_FGT = []

        self.cap_list = []
        self.dcp_list = []
        self.nld_list = []
        self.gt_img_list = []
        self.clearness_NLD = []
        self.clearness_BDCP = []
        self.clearness_DCP = []
        self.clearness_BCCR= []
        self.name_NLD = dict()
        self.name_BDCP = dict()
        self.name_DCP = dict()
        self.name_BCCR = dict()
        self.un_clean_CLAHE = []
        self.un_clean_img_list = glob('D:\OTS_BETA\clear\clear\*.jpg')
        self.isTrain = istrain
        self.Flip = flip
        if self.isTrain:
            if self.isTrain:


                for img in self.unhazy_img_list:
                    img_name_NLD = img.replace('hazy','IDRLP')
                    img_name_DCP = img.replace('hazy', 'DCP')
                    #IDRLP
                    img_name_BCCR = img.replace('hazy', 'BCCR')
                    img_name_BCP = img.replace('hazy', 'BDCP')
                    img_name_IDE = img.replace('hazy', 'IDE')
                    #FGT_1
                    img_name_FGT = img.replace('hazy', 'FGT47')
                    self.unclean_img_list_NLD.append(img_name_NLD)
                    # self.unclean_img_list_CAP.append(img_name_CAP)
                    self.unclean_img_list_BCCR.append(img_name_BCCR)
                    self.unclean_img_list_DCP.append(img_name_DCP)
                    self.unclean_img_list_BCP.append(img_name_BCP)
                    self.unclean_img_list_IDE.append(img_name_IDE)
                    self.unclean_img_list_FGT.append(img_name_FGT)



        else:
            self.unhazy_img_list = glob(data_dir_sys + '*.jpg')
            for img in self.unhazy_img_list:
                img_name = img.split('\\')[-1]
                gt_name = img_name.replace('hazy','GT')
                self.clean_img_list.append(data_dir_sys_gt + gt_name)
            if data_dir_sys_hazy2 is not None:
                self.hazy_img_list2 = glob(data_dir_sys_hazy2 + '*.png')
                for img in self.hazy_img_list2:
                    img_name = img.split('\\')[-1]
                    gt_name = img_name.split('_')[0]
                    gt_name = gt_name + '.png'
                    self.clean_img_list.append(data_dir_sys_gt2 + gt_name)
                self.unhazy_img_list.extend(self.hazy_img_list2)

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
    def augData(self, data, target1, target2, target3, target4, target5, target6, clean=None, enhance=True):

        if self.isTrain:
            if enhance:
                rand_hor = random.randint(0, 1)
                rand_rot = random.randint(0, 3)
                data = transforms.RandomHorizontalFlip(rand_hor)(data)
                target1 = transforms.RandomHorizontalFlip(rand_hor)(target1)
                target2 = transforms.RandomHorizontalFlip(rand_hor)(target2)
                target3 = transforms.RandomHorizontalFlip(rand_hor)(target3)
                target4 = transforms.RandomHorizontalFlip(rand_hor)(target4)
                target5 = transforms.RandomHorizontalFlip(rand_hor)(target5)
                target6 = transforms.RandomHorizontalFlip(rand_hor)(target6)
                if clean is not None:
                    clean = transforms.RandomHorizontalFlip(rand_hor)(clean)
                if rand_rot:
                    data = transforms.functional.rotate(data, 90 * rand_rot)
                    target1 = transforms.functional.rotate(target1, 90 * rand_rot)
                    target2 = transforms.functional.rotate(target2, 90 * rand_rot)
                    target3 = transforms.functional.rotate(target3, 90 * rand_rot)
                    target4 = transforms.functional.rotate(target4, 90 * rand_rot)
                    target5 = transforms.functional.rotate(target5, 90 * rand_rot)
                    target6 = transforms.functional.rotate(target6, 90 * rand_rot)
                    if clean is not None:
                        clean = transforms.functional.rotate(clean, 90 * rand_rot)

        data = transforms.ToTensor()(data)
        # data = tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target1 = transforms.ToTensor()(target1)
        target2 = transforms.ToTensor()(target2)
        target3 = transforms.ToTensor()(target3)
        target4 = transforms.ToTensor()(target4)
        target5 = transforms.ToTensor()(target5)
        target6 = transforms.ToTensor()(target6)
        if clean is not None:
            clean = transforms.ToTensor()(clean)
        data = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(data)
        target1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(target1)
        target2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(target2)
        target3 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(target3)
        target4 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(target4)
        target5 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(target5)
        target6 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(target6)
        if clean is not None:
            clean = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(clean)
        if clean is not None:
            return data, target1, target2, target3, target4, target5, target6, clean
        else:
            return data, target1, target2, target3, target4, target5, target6
    def __getitem__(self, index):
        if self.isTrain:
            index1 = torch.randint(0,len(self.unhazy_img_list),(1,)).item()
            index2 = torch.randint(0, len(self.hazy_img_list), (1,)).item()

            src = Image.open(self.hazy_img_list[index2]).convert('RGB')
            w, h = src.size
            gt_img = src.crop((int(w / 2), 0, w, h))
            hazy_img = src.crop((0, 0, int(w / 2), h))
            un_img = Image.open(self.unhazy_img_list[index1]).convert('RGB')
            un_img1 = Image.open(self.unhazy_img_list[index1]).convert('RGB')
            un_clean_img_NLD = Image.open(self.unclean_img_list_NLD[index1]).convert('RGB')
            un_clean_img_BCCR = Image.open(self.unclean_img_list_BCCR[index1]).convert('RGB')
            un_clean_img_DCP = Image.open(self.unclean_img_list_DCP[index1]).convert('RGB')
            un_clean_img_BCP = Image.open(self.unclean_img_list_BCP[index1]).convert('RGB')
            un_clean_img_IDE = Image.open(self.unclean_img_list_IDE[index1]).convert('RGB')
            un_clean_img_FGT = Image.open(self.unclean_img_list_FGT[index1]).convert('RGB')








            rand_hor = random.randint(0, 1)

            hazy_img = transforms.RandomHorizontalFlip(rand_hor)(hazy_img)
            gt_img = transforms.RandomHorizontalFlip(rand_hor)(gt_img)
            un_img = transforms.RandomHorizontalFlip(rand_hor)(un_img)
            un_img1 = transforms.RandomHorizontalFlip(rand_hor)(un_img1)

            un_clean_img_NLD = transforms.RandomHorizontalFlip(rand_hor)(un_clean_img_NLD)
            un_clean_img_BCCR = transforms.RandomHorizontalFlip(rand_hor)(un_clean_img_BCCR)
            # un_clean_img_CAP = un_clean_img_CAP.transpose(Image.FLIP_LEFT_RIGHT)
            un_clean_img_DCP = transforms.RandomHorizontalFlip(rand_hor)(un_clean_img_DCP)
            un_clean_img_BCP = transforms.RandomHorizontalFlip(rand_hor)(un_clean_img_BCP)
            un_clean_img_IDE = transforms.RandomHorizontalFlip(rand_hor)(un_clean_img_IDE)
            un_clean_img_FGT = transforms.RandomHorizontalFlip(rand_hor)(un_clean_img_FGT)
            # rand_rot = random.randint(0, 3)
            # hazy_img = transforms.functional.rotate(hazy_img, 90 * rand_rot)
            # gt_img = transforms.functional.rotate(gt_img, 90 * rand_rot)
            # un_img = transforms.functional.rotate(un_img, 90 * rand_rot)
            # un_img1 = transforms.functional.rotate(un_img1, 90 * rand_rot)
            # un_img_en = transforms.functional.rotate(un_img_en, 90 * rand_rot)
            # un_clean_img_NLD = transforms.functional.rotate(un_clean_img_NLD, 90 * rand_rot)
            # un_clean_img_BCCR = transforms.functional.rotate(un_clean_img_BCCR, 90 * rand_rot)
            # un_clean_img_DCP = transforms.functional.rotate(un_clean_img_DCP, 90 * rand_rot)
            # un_clean_img_BCP = transforms.functional.rotate(un_clean_img_BCP, 90 * rand_rot)
            # un_clean_img_FGT = transforms.functional.rotate(un_clean_img_FGT, 90 * rand_rot)
            # un_clean_img = un_clean_img.transpose(Image.FLIP_LEFT_RIGHT)



            # unhazy_img_e = enhance_random(unhazy_img)
            w_s, h_s = un_img.size

            hazy_img = hazy_img.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            gt_img = gt_img.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            un_img = un_img.resize((self.scale_size, self.scale_size), Image.BICUBIC)

            un_img1 = un_img1.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            un_clean_img_NLD = un_clean_img_NLD.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            un_clean_img_BCCR = un_clean_img_BCCR.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            un_clean_img_BCP = un_clean_img_BCP.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            un_clean_img_IDE = un_clean_img_IDE.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            un_clean_img_FGT = un_clean_img_FGT.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            # un_clean_img_CAP = un_clean_img_CAP.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            un_clean_img_DCP = un_clean_img_DCP.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            # un_clean_img = un_clean_img.resize((self.scale_size, self.scale_size), Image.BICUBIC)
            # un_clean_img_down = un_clean_img.resize((self.scale_size//4, self.scale_size//4), Image.BICUBIC)
            # un_clean_img_down = un_clean_img_down.resize((self.scale_size , self.scale_size ), Image.BICUBIC)

            i, j, h, w = transforms.RandomCrop.get_params(hazy_img, output_size=(self.size, self.size))

            gt_img1 = TF.crop(gt_img, i, j, h, w)
            hazy_img1 = TF.crop(hazy_img, i, j, h, w)
            un_clean_img1_NLD = TF.crop(un_clean_img_NLD, i, j, h, w)
            un_clean_img1_BCCR = TF.crop(un_clean_img_BCCR, i, j, h, w)
            un_clean_img1_DCP = TF.crop(un_clean_img_DCP, i, j, h, w)
            un_clean_img1_BCP = TF.crop(un_clean_img_BCP, i, j, h, w)
            un_clean_img1_IDE = TF.crop(un_clean_img_IDE, i, j, h, w)
            un_clean_img1_FGT = TF.crop(un_clean_img_FGT, i, j, h, w)

            un_img_1 = TF.crop(un_img, i, j, h, w)
            un_img_2 = TF.crop(un_img1, i, j, h, w)


            e_NLD = self.imgae_entropy(un_clean_img1_NLD)
            e_DCP = self.imgae_entropy(un_clean_img1_DCP)
            e_BDCP = self.imgae_entropy(un_clean_img1_BCP)
            e_BCCR = self.imgae_entropy(un_clean_img1_BCCR)
            e_IDE = self.imgae_entropy(un_clean_img1_IDE)
            e_hazy = self.imgae_entropy(un_img)

            hazy_img1 = transforms.ToTensor()(hazy_img1)

            gt_img1 = transforms.ToTensor()(gt_img1)
            un_img_1 = transforms.ToTensor()(un_img_1)
            un_img_2 = transforms.ToTensor()(un_img_2)


            un_clean_img1_NLD = transforms.ToTensor()(un_clean_img1_NLD)
            un_clean_img1_BCCR = transforms.ToTensor()(un_clean_img1_BCCR)
            # un_clean_img_CAP = transforms.ToTensor()(un_clean_img_CAP)
            un_clean_img1_DCP = transforms.ToTensor()(un_clean_img1_DCP)
            un_clean_img1_BCP = transforms.ToTensor()(un_clean_img1_BCP)
            un_clean_img1_IDE = transforms.ToTensor()(un_clean_img1_IDE)
            un_clean_img1_FGT = transforms.ToTensor()(un_clean_img1_FGT)


            # un_img_2 = transforms.ColorJitter(brightness=0.2,contrast=0.5,saturation=0.1,hue=0.1)(un_img1)
            hazy_img1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(hazy_img1)
            #
            gt_img1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(gt_img1)
            un_img_1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_img_1)
            un_img_2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_img_2)

            # un_clean_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_img)
            # un_clean_img_down = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_img_down)
            # un_clean_CLAHE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_CLAHE)
            un_clean_img1_NLD = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_img1_NLD)
            un_clean_img1_BCCR = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_img1_BCCR)
            un_clean_img1_BCP = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_img1_BCP)
            un_clean_img1_IDE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_img1_IDE)
            un_clean_img1_FGT = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_img1_FGT)
            # un_clean_img1_CAP = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_img1_CAP)
            un_clean_img1_DCP = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(un_clean_img1_DCP)



            # trans_img = trans_img*2-1
            # un_hazy_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(unhazy_img)
            # un_hazy_img_e = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(unhazy_img_e)
            return hazy_img1,gt_img1,un_img_1,un_clean_img1_NLD,un_clean_img1_BCCR,un_clean_img1_BCP,un_clean_img1_DCP,un_clean_img1_IDE,un_clean_img1_FGT,e_NLD,e_BCCR,e_BDCP,e_DCP,e_IDE,e_hazy

        else:
            hazy_img = Image.open(self.unhazy_img_list[index]).convert('RGB')
            clean_img = Image.open(self.clean_img_list[index]).convert('RGB')
            w_s, h_s = hazy_img.size
            # w_s = (w_s // 8) * 8
            # h_s = (h_s // 8) * 8
            w_s = 512
            h_s = 512
            # # clean_img = clean_img.crop((10, 10, 630, 470))
            hazy_img = hazy_img.resize((w_s, h_s), Image.BICUBIC)
            clean_img = clean_img.resize((w_s, h_s), Image.BICUBIC)

            transform_list = []
            transform_list += [transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]
            trans = transforms.Compose(transform_list)
            img_name = self.unhazy_img_list[index].split('\\')[-1]

            return trans(hazy_img), trans(clean_img)

    def __len__(self):
        return len(self.hazy_img_list)