# DNMGDT [IEEE TMM 2025]
This is the official Python implementation of our paper "Real Scene Single Image Dehazing Network with Multi-Prior Guidance and Domain Transfer",  which is accepted by [IEEE Transactions on Multimedia] in 2025. If you are interested at this work, you can star the repository for newest update. Thanks!

# Contributions
1 We propose DNMPDT, a novel semi-supervised dehazing model for real scene image dehazing, which simultaneously utilizes real hazy data set and synthetic hazy data set to train one encoder-decoder architecture in a parameter-shared way, and thus frees from two-stage training (pretraining and fine-tuning).

2 We propose an effective image fusion mechanism termed IQGAW to yield more reliable fake clean images for real hazy ones, which adaptively weights various prior-based dehazed images with the guidance of chosen image quality indexes, automatically attaching importance to some local regions by content variations. 

3 We propose  an image level domain transfer strategy termed PMGDT to reduce the domain gap for the collaborative training between real hazy and synthetic hazy data sets. 

For the proposed  IQGAW and PMGDT , you can find them in "train_DNMGDT".

# Our Environment
Python (3.6)

PyTorch (1.10) 

numpy (1.13.3)

Now we detail the usage of this code!
# Usage
## Quick test by our pretrained model
The pretrained model "netG.pth" has been put in "checkpoints". Before running "test_DNMGDT", you should change to your paths. For convenience, we build a  function for each testing data set e.g., ihaze and ohaze. Therefore, you should first change the data path for corresponding test function by 

```
def test_ohaze():
    ...
    imgs_hazy = glob.glob('F:\\dataset\\NTIRE2018\\O-HAZE\\hazy\\*.jpg')
```
After that, you should change to  "test_ohaze()" function as
```
if __name__ =='__main__':
    test_ohaze()
    # test_ihaze()
```
The dehazed results of all ohaze hazy images will be saved to  "ohaze" file.
## Training by your own data set
Before running "train_DNMGDT", you should change to your paths. 

1 ***change the used traning data sets by "train_data"***.  :exclamation: Note: there are two training data (synthetic haze dataset and real haze dataset) for a semi-supervised training model like our DNMGDT. The used synthetic (syn for short) haze dataset are 6000 paired images in [1]. You can directly download at syn data by  [Link](https://github.com/HUSTSYJ/DA_dahazing). Since the real   haze dataset do not have paired real clean images, we have proposed a IQGAW method obtain the fake clean image by fusing multiple prior-based dehazed images. Our real haze dataset can be downloaded by  [Link](https://pan.baidu.com/s/13J4miQv30_SHRy4l9f8RbA?pwd=1234), extract code 1234. We provide two extra prior dehazed images obtained by BDCP and BCCR. We conduct ablation experiment that trains our model by using from one to six prior-based dehazed images. We have found that using more than three prior based method can  generate a decent fake clean image, achieving competitive results. The pretrained model "netG.pth" is traind by using four priors including DCP, IDRLP, NLD and IDE, and we use it to report the experiments in our paper. You should change the path "D:\Data\DAADPTION_CVPR2020\\train\\" by your downloaded syn data set and change the path "D:\\UnDehaze\\hazy\\" by our real data set.

2 ***change the validation data sets by "test_data"***. :exclamation: Note: here the  validation data set (e.g., ohaze) is used to visualize the results during training, providing the visualization of training process.
```
train_data = SemiDatasets('D:\Data\DAADPTION_CVPR2020\\train\\', 'D:\\UnDehaze\\hazy\\')
test_data = SemiDatasets('D:\Data\\O-HAZE\Data\\all\hazy\\',
                            'D:\Data\\O-HAZE\Data\\all\\','D:\Data\\O-HAZE\Data\\all\GT\\',
                            istrain=False)
```
If you succeed to train our model, you should know the recording method of this code. 

1 After  every 200 iterations, we visualized the results before and after domain transitions. All the visualization results are saved in "samples" file. 

2 After  every 3000 iterations, we use the validation data set (e.g., ohaze) for quantitative comparison and report the PSNR. All the visualization results are saved in "samples" file. 

3 Moreover, after  every epoch, we save the traned model in "checkpoints/semimodel_paper". The naming of the trained model includes the epoch number and the PSNR value (e.g., netG_model_0_17.124600928830542.pth), providing convenience for distinguishing them. The pretrained model "netG.pth" is renamed by ourselves.
# References
[1] Shao, Y., Li, L., Ren, W., Gao, C., and Sang, N.. Domain adaptation for image dehazing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020: 2808-2817.

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
# For more works
My studies focus on machine learning (deep learning) and its applications to image enhancement, data clustering (classification), object recognition (tracking) etc.  If you are interested at my works, you can get more papers and codes at my [Homepage](https://nianwang-hjjgcdx.github.io/).
