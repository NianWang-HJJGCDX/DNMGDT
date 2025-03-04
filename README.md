# DNMGDT [IEEE TMM 2025]
This is the official Python implementation of our paper "Real Scene Single Image Dehazing Network with Multi-Prior Guidance and Domain Transfer",  which is accepted by [IEEE Transactions on Multimedia] in 2025. If you are interested at this work, you can star the repository for newest update. Thanks!

# For more works
My studies focus on machine learning (deep learning) and its applications to image enhancement, data clustering (classification), object recognition (tracking) etc.  If you are interested at my works, you can get more papers and codes at my [Homepage](https://nianwang-hjjgcdx.github.io/).

Now we detail the usage of this code!
# Our Environment
Python (3.6)
PyTorch (1.10) 
numpy (1.13.3)

# Usage
## Quick testing by our pretrained model
The pretrained model "netG.pth" has been put in "checkpoints". Before running "TMM_test", you should change to your paths. For convenience, we build a  function for each testing data set. Therefore, you should first change the  function by 
```
if __name__ =='__main__':
    test_ohaze()
    # test_ihaze()
```
and then change the data path in corresponding function by
```
def test_ohaze():
    ...
    imgs_hazy = glob.glob('F:\\dataset\\NTIRE2018\\O-HAZE\\hazy\\*.jpg')
```
## Training by your own data set
Before running "TMM_train", you should change to your paths.

1 change the used traning data sets by "train_data" and change the testing data sets by "test_data". :exclamation: Note: here the testing data set is used to visualize the results during training, providing the visualization of training process. 
```
train_data = SemiDatasets72('D:\Data\DAADPTION_CVPR2020\\train\\', 'D:\\UnDehaze\\hazy\\')
test_data = SemiDatasets72('D:\Data\\O-HAZE\Data\\all\hazy\\',
                            'D:\\UnDehaze\\hazy\\','D:\Data\\O-HAZE\Data\\all\GT\\',
                            istrain=False)
```

2 change the save path of the visul results during the training
```
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
```
# Citation
```
@ARTICLE{DNMGDT,
  author={Su, Yanzhao and Wang, Nian and Cui, Zhigao and Cai, Yanping and He, Chuan and Li, Aihua},
  journal={IEEE Transactions on Multimedia}, 
  title={Real Scene Single Image Dehazing Network with Multi-Prior Guidance and Domain Transfer}, 
  year={2025},
  volume={},
  number={},
  pages={1-16},
  keywords={Training;Atmospheric modeling;Learning systems;Synthetic data;Indexes;Image color analysis;Computational modeling;Adaptation models;Translation;Training data;Single image dehazing;multi-prior guidance;domain transfer;image quality guided adaptive weighting;consistency constraint},
  doi={10.1109/TMM.2025.3543063}}
```
