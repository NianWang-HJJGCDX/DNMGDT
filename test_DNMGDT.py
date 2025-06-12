
import torch
use_cuda = torch.cuda.is_available()
import glob
import time
import cv2
from torch.autograd import Variable
from model import *
logdir = './checkpoints/'
which_model= 'netG.pth'
save_path_ohaze ='./ohaze/'
save_path_ihaze ='./ihaze/'
def normalize(data):
    return data / 255.

def test_ohaze():
    net = torch.load(logdir + which_model)
    if use_cuda:
        net = net.cuda()
    time_test = 0
    count = 0
    imgs_hazy = glob.glob('F:\\dataset\\NTIRE2018\\O-HAZE\\hazy\\*.jpg')

    for hazy_name in imgs_hazy:

        hazy_img = cv2.imread(hazy_name)
        t = hazy_img.shape
        h = t[0]
        w = t[1]

        hazy_img = cv2.resize(hazy_img,(512,512))
        hazy_img1 = hazy_img
        b, g, r = cv2.split(hazy_img1)
        hazy_img1 = cv2.merge([r, g, b])
        hazy_img1 = normalize(np.float32(hazy_img1)) * 2 - 1
        hazy_img1 = np.expand_dims(hazy_img1.transpose(2, 0, 1), 0)
        hazy_img1 = Variable(torch.Tensor(hazy_img1))
        img_name = hazy_name.split('\\')[-1]
        if use_cuda:
            hazy_img1 = hazy_img1.cuda()


        with torch.no_grad():
            if use_cuda:
                torch.cuda.synchronize()

            start_time = time.time()

            fake_b = net(hazy_img1)

            fake_b = torch.clamp(fake_b, -1, 1.)
            end_time = time.time()
            dur_time = end_time - start_time
            time_test += dur_time
            if use_cuda:
                save_out = np.uint8(127.5 * (fake_b.data.cpu().numpy().squeeze() + 1))
            else:
                save_out = np.uint8(255.0 * (fake_b).data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            save_out = cv2.resize(save_out,(w,h))

            cv2.imwrite(save_path_ohaze+img_name, save_out)

            count += 1
    print('Avg. time:', time_test / count)

def test_ihaze():

    net = torch.load(logdir+which_model)
    if use_cuda:
        net = net.cuda()
    net.eval()
    time_test = 0
    count = 0
    imgs_hazy = glob.glob('D:\\Data\\I-HAZE\\Data\\all\\hazy\\*.jpg')

    for hazy_name in imgs_hazy:
        hazy_img = cv2.imread(hazy_name)
        t = hazy_img.shape
        h = t[0]
        w = t[1]

        if w>h:
            hazy_img = cv2.resize(hazy_img, (512, 512))
        else:
            hazy_img = cv2.resize(hazy_img, (512, 512))

        img_name = hazy_name.split('\\')[-1]
        b, g, r = cv2.split(hazy_img)
        hazy_img = cv2.merge([r, g, b])

        hazy_img = normalize(np.float32(hazy_img))*2-1
        hazy_img = np.expand_dims(hazy_img.transpose(2, 0, 1), 0)
        hazy_img = Variable(torch.Tensor(hazy_img))
        if use_cuda:
            hazy_img = hazy_img.cuda()

        with torch.no_grad():
            if use_cuda:
                torch.cuda.synchronize()
            start_time = time.time()

            fake_b = net(hazy_img)

            fake_b = torch.clamp(fake_b, -1, 1.)
            end_time = time.time()
            dur_time = end_time - start_time
            # print("test time:", dur_time)
            time_test += dur_time

            if use_cuda:
                save_out = np.uint8(127.5 * (fake_b.data.cpu().numpy().squeeze() + 1))
            else:
                save_out = np.uint8(255.0 * (fake_b).data.numpy().squeeze())
            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])
            save_out = cv2.resize(save_out,(w,h))
            cv2.imwrite(save_path_ihaze+img_name, save_out)
            count += 1
    print('Avg. time:', time_test / count)

if __name__ =='__main__':
    test_ohaze()
    # test_ihaze()