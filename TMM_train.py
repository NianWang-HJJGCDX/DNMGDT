from __future__ import print_function
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from TMM_dataset import SemiDatasets72
from TMM_model import Model_Semi_base,DCPDehazeGenerator,Discriminator
from torch.optim.lr_scheduler import MultiStepLR
from TMM_utils import *
import random
import glob
import  time
import cv2
use_cuda = torch.cuda.is_available()

save_path = os.path.join("checkpoint", 'semi_model')
name = 'semimodel_paper'

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            # result = re.findall(".*_epoch(_.*).pth.*", file_)
            result = file_.split('_')[-1].split('.')[0]
            epochs_exist.append(int(result))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def train():

    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    np.random.seed(3407)


    train_data = SemiDatasets72('D:\Data\DAADPTION_CVPR2020\\train\\', 'D:\\UnDehaze\\hazy\\')

    test_data = SemiDatasets72('D:\Data\\O-HAZE\Data\\all\hazy\\',
                              'D:\\UnDehaze\\hazy\\','D:\Data\\O-HAZE\Data\\all\GT\\',
                              istrain=False)

    training_data_loader = DataLoader(dataset=train_data, num_workers=4, batch_size=1,
                                      shuffle=True)
    testing_data_loader = DataLoader(dataset=test_data, num_workers=4, batch_size=1,
                                     shuffle=False)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # the ratio for different image quality indexes
    ratio = 10
    # the temprature for the image quality of the dehazed image
    temp = 100

    net = Model_Semi_base().to(device)
    net_DCP = DCPDehazeGenerator(win_size=7,r=30).to(device)
    net_D1 = Discriminator(in_channel=3).to(device)

    initial_epoch = findLastCheckpoint(save_dir=save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        net = torch.load(save_path + "\\netG_model_epoch_%d.pth" % initial_epoch)
        net = net.to(device)

    criterionMSE = nn.MSELoss().to(device)

    criterion = SSIM().to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(net_D1.parameters(),
                              lr=0.0002, betas=(0.5, 0.99))
    criterionGAN = GANLoss('vanilla').to(device)
    scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25], gamma=0.75)

    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    old_psnr = 0
    for epoch in range(20):
        # lr = adjust_learning_rate(optimizer, epoch - 1)
        start_time = time.time()
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a, real_b, real_un1, real_NLD, real_BCCR, real_BCP, real_DCP,real_IDE, real_FGT,entropy_0,entropy_1,entropy_2,entropy_3,entropy_4,entropy_5 = \
                batch[0].to(device), batch[1].to(device), batch[2].to(device), \
                batch[3].to(device), batch[4].to(device), batch[5].to(device), \
                batch[6].to(device), batch[7].to(device), batch[8].to(device), batch[9].to(device) \
                    , batch[10].to(device), batch[11].to(device),batch[12].to(device),batch[13].to(device), \
                    batch[14].to(device)

            fake_b_s = net(real_a)
            fake_b1 = net(real_un1)

            _,T_S,A_S = net_DCP(real_a)
            _, T_R, A_R = net_DCP(real_un1)

            r_Tmin = torch.minimum(T_S.mean() / T_R.mean(),T_R.mean() / T_S.mean())
            r_Tmax = torch.maximum(T_S.mean() / T_R.mean(), T_R.mean() / T_S.mean())

            ratio_TS = random.uniform(r_Tmin,r_Tmax)
            ratio_TR = random.uniform(r_Tmin,r_Tmax)
            T_S1 = ratio_TS * T_S
            T_R1 = ratio_TR * T_R

            r_Amin = torch.minimum(A_S.mean() / A_R.mean(), A_R.mean() /A_S.mean())
            r_Amax = torch.maximum(A_S.mean() / A_R.mean(), A_R.mean() / A_S.mean())
            ratio_AS = random.uniform(r_Amin, r_Amax)
            ratio_AR = random.uniform(r_Amin, r_Amax)

            A_S = ratio_AS * A_S
            A_R = ratio_AR * A_R

            A_S = torch.clamp(A_S, 0, 1)
            A_R = torch.clamp(A_R, 0, 1)
            T_S1 = torch.clamp(T_S1, 0, 1)
            T_R1 = torch.clamp(T_R1, 0, 1)
            real_a1 = (real_b/2+0.5)*T_S1.repeat(1,3,1,1)+A_R*(1-T_S1.repeat(1,3,1,1))
            real_un2 = (real_FGT/2+0.5) * T_R1.repeat(1,3,1,1) + A_S *(1 - T_R1.repeat(1,3,1,1))

            # used for consistency regularization
            fake_b_s1 = net(real_a1 * 2 - 1)
            fake_b2 = net(real_un2 * 2 - 1)

            # add the GAN training
            set_requires_grad(net_D1, True)
            optimizer_D.zero_grad()
            pred_fake1 = net_D1(fake_b1.detach())

            loss_D1_fake = criterionGAN(pred_fake1, False)

            pred_real = net_D1(real_b)
            loss_D1_real = criterionGAN(pred_real, True)

            loss_D1 = (loss_D1_fake + loss_D1_real) * 0.5
            loss_D1.backward()
            optimizer_D.step()
            set_requires_grad(net_D1, False)
            optimizer.zero_grad()
            pred_fake1 = net_D1(fake_b1)

            loss_G_GAN = criterionGAN(pred_fake1, True)

            loss_d = -criterion(real_b/ 2. + 0.5, fake_b_s / 2. + 0.5)[0]

            loss_c = criterionMSE(fake_b_s1, real_b) + criterionMSE(fake_b1, fake_b2)

            # calculate the quality of the dehazed image by the priors
            w1 = haze_evaluate(real_NLD / 2. + 0.5) *ratio + entropy_0
            w2 = haze_evaluate(real_BCCR / 2. + 0.5)*ratio   + entropy_1
            w4 = haze_evaluate(real_DCP / 2. + 0.5)*ratio   + entropy_3
            w5 = haze_evaluate(real_IDE / 2. + 0.5) * ratio  + entropy_4


            w1 = w1 / temp
            w2 = w2 / temp
            w4 = w4 / temp
            w5 = w5 / temp

            sum_w = torch.exp(w1) + torch.exp(w2) + torch.exp(w4) + torch.exp(w5)

            weight_un = [torch.exp(w1) / sum_w, torch.exp(w2) / sum_w, torch.exp(w4) / sum_w, torch.exp(w5) / sum_w]
            weight_map = content_weight(real_FGT/ 2. + 0.5)

            #adaptive weight for the prior guidance
            loss_d1 = -criterion(real_NLD / 2. + 0.5, fake_b1 / 2. + 0.5)[-1]
            loss_d2 = -criterion(real_BCCR / 2. + 0.5, fake_b1 / 2. + 0.5)[-1]
            loss_d4 = -criterion(real_DCP / 2. + 0.5, fake_b1 / 2. + 0.5)[-1]
            loss_d5 = -criterion(real_IDE / 2. + 0.5, fake_b1 / 2. + 0.5)[-1]
            loss_u1 = loss_d1 * weight_un[0] + loss_d2 * weight_un[1]  + loss_d4 * weight_un[
                2] + loss_d5 * weight_un[3]

            loss_u1 =  loss_u1 * weight_map
            loss_u1 = loss_u1.mean()/(weight_map.mean())

            loss = loss_d + loss_u1 + 0.8* loss_c+0.01*loss_G_GAN
            loss.backward()

            optimizer.step()
            scheduler.step(epoch)


            if iteration % 100 == 0:
                print("train 100 iteration:", time.time() - start_time)
                start_time = time.time()
                print("===> Epoch[{}]({}/{}): Loss_T: {:.4f} Loss_U1: {:.6f} Loss_C: {:.6f} ".format(
                        epoch, iteration, len(training_data_loader), loss_d.item(), loss_u1.mean().item(),loss_c.mean().item()))
                print("===> Epoch[{}]({}/{}): Loss_T: {:.4f} Loss_U1: {:.6f} Loss_C: {:.6f} Loss_d: {:.6f} Loss_g: {:.6f}".format(
                        epoch, iteration, len(training_data_loader), loss_d.item(), loss_u1.mean().item(),
                        loss_c.mean().item(),loss_D1.item(),loss_G_GAN.item()))

            if iteration % 200 == 0:
                out_hazy = (real_a[0]).cpu().detach().permute(1, 2, 0).numpy()
                out_gt = (real_b[0]).cpu().detach().permute(1, 2, 0).numpy()
                fake_out = (fake_b_s[0]).cpu().detach().permute(1, 2, 0).numpy()
                fake_out0 = (fake_b1[0]).cpu().detach().permute(1, 2, 0).numpy()

                out_r = (real_un1[0]).cpu().detach().permute(1, 2, 0).numpy()
                out_NLD = ((real_a1*2-1)[0]).cpu().detach().permute(1, 2, 0).numpy()
                out_BCCR = ((real_un2*2-1)[0]).cpu().detach().permute(1, 2, 0).numpy()
                out_DCP = (fake_b_s1[0]).cpu().detach().permute(1, 2, 0).numpy()
                out_BDCP = (fake_b2[0]).cpu().detach().permute(1, 2, 0).numpy()

                out_r = np.clip(out_r, -1, 1)
                out_NLD = np.clip(out_NLD, -1, 1)
                out_BCCR = np.clip(out_BCCR, -1, 1)
                out_DCP = np.clip(out_DCP, -1, 1)
                out_BDCP = np.clip(out_BDCP, -1, 1)

                out_r = ((out_r + 1) / 2 * 255).astype(np.uint8)
                out_NLD = ((out_NLD + 1) / 2 * 255).astype(np.uint8)
                out_BCCR = ((out_BCCR + 1) / 2 * 255).astype(np.uint8)
                out_DCP = ((out_DCP + 1) / 2 * 255).astype(np.uint8)
                out_BDCP = ((out_BDCP + 1) / 2 * 255).astype(np.uint8)
                fake_out = np.clip(fake_out, -1, 1)
                fake_out0 = np.clip(fake_out0, -1, 1)

                out_hazy = ((out_hazy + 1) / 2 * 255).astype(np.uint8)
                out_gt = ((out_gt + 1) / 2 * 255).astype(np.uint8)
                # out_weight = ((out_weight+1)/2 * 255).astype(np.uint8)
                fake_out = ((fake_out + 1) / 2 * 255).astype(np.uint8)
                fake_out0 = ((fake_out0 + 1) / 2 * 255).astype(np.uint8)
                # fake_out1 = ((fake_out1 + 1) / 2 * 255).astype(np.uint8)
                # fake_out2 = ((fake_out2 + 1) / 2 * 255).astype(np.uint8)
                out_hazy = cv2.cvtColor(out_hazy, cv2.COLOR_RGB2BGR)
                out_gt = cv2.cvtColor(out_gt, cv2.COLOR_RGB2BGR)
                fake_out = cv2.cvtColor(fake_out, cv2.COLOR_RGB2BGR)
                fake_out0 = cv2.cvtColor(fake_out0, cv2.COLOR_RGB2BGR)

                out_r = cv2.cvtColor(out_r, cv2.COLOR_RGB2BGR)
                out_NLD = cv2.cvtColor(out_NLD, cv2.COLOR_RGB2BGR)
                out_BCCR = cv2.cvtColor(out_BCCR, cv2.COLOR_RGB2BGR)
                out_DCP = cv2.cvtColor(out_DCP, cv2.COLOR_RGB2BGR)
                out_BDCP = cv2.cvtColor(out_BDCP, cv2.COLOR_RGB2BGR)

                cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_s_hazy.png".format(iteration),
                            out_hazy)
                cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_s_gt.png".format(iteration),
                            out_gt)
                cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_s_out.png".format(iteration),
                            fake_out)
                cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_real.png".format(iteration),
                            out_r)
                cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_r_out.png".format(iteration),
                            fake_out0)
                cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_NLD.png".format(iteration),
                            out_NLD)
                cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_BCCR.png".format(iteration), out_BCCR)
                # cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_weight.png".format(iteration), out_weight)
                cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_DCP.png".format(iteration),
                            out_DCP)
                cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_BDCP.png".format(iteration),
                            out_BDCP)
            if iteration % 4500 == 0:
                    with torch.no_grad():
                        net.eval()
                        avg_psnr = 0
                        avg_ssim = 0
                        for batch in testing_data_loader:
                            input, target = batch[0].to(device), batch[1].to(device)
                            prediction = net(input)
                            psnr = batch_PSNR(prediction, target)
                            ssim_a = criterion((prediction + 1) / 2., (target + 1) / 2.)[0]
                            avg_psnr += psnr
                            avg_ssim += ssim_a
                        print("===> Avg. PSNR: {:.4f} dB,SSIM:{:4f} dB".format(avg_psnr / len(testing_data_loader),
                                                                               avg_ssim / len(testing_data_loader)))

                        if not os.path.exists("checkpoint"):
                            os.mkdir("checkpoint")
                        if not os.path.exists(os.path.join("checkpoint", name)):
                            os.mkdir(os.path.join("checkpoint", name))
                        model_out_path = "checkpoint/{}/netG_model_{}_{}.pth".format(name, epoch,
                                                                                     avg_psnr / len(
                                                                                         testing_data_loader))
                        torch.save(net, model_out_path)

                    print("Checkpoint saved to {}".format("checkpoint" + 'semi_model'))
        # if iteration % 4500 == 0:
        if epoch % 1 == 0:
            with torch.no_grad():
                net.eval()
                avg_psnr = 0
                avg_ssim =0
                for batch in testing_data_loader:
                    input, target = batch[0].to(device), batch[1].to(device)
                    prediction = net(input)
                    psnr = batch_PSNR(prediction, target)
                    ssim_a = criterion((prediction+1)/2.,(target+1)/2.)[0]
                    avg_psnr += psnr
                    avg_ssim += ssim_a
                print("===> Avg. PSNR: {:.4f} dB,SSIM:{:4f} dB".format(avg_psnr / len(testing_data_loader),avg_ssim / len(testing_data_loader)))

                if not os.path.exists("checkpoint"):
                    os.mkdir("checkpoint")
                if not os.path.exists(os.path.join("checkpoint", name)):
                    os.mkdir(os.path.join("checkpoint", name))
                model_out_path = "checkpoint/{}/netG_model_{}_{}.pth".format(name, epoch,
                                                                             avg_psnr / len(
                                                                                 testing_data_loader))
                torch.save(net, model_out_path)

            print("Checkpoint saved to {}".format("checkpoint" + 'semi_model'))
if __name__ == '__main__':
    train()