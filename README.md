# LiMFusion: Infrared and visible image fusion via local information measurement
Yao Qian, Haojie Tang, Gang Liu∗, Mengliang Xing, Gang Xiao,  Durga Prasad Bavirisetti

Published in: Optics and Lasers in Engineering

- [paper](https://www.sciencedirect.com/science/article/abs/pii/S0143816624004135)

## Abstract
Most image fusion methods currently achieve satisfactory results under normal conditions. However, in complex scenes, the problem of information conflict between source images has not been effectively resolved. Sometimes, effective information from one source image may be masked by noise from another source image. In this paper, by organically combining the traditional decomposition strategy and the attention mechanism, we propose a novel infrared and visible image fusion network based on local information measurement, named LiMFusion. Specifically, the source image is decomposed by utilizing fourth-order partial differential equations to obtain high-frequency and low-frequency layers. Feature representation and information preservation capabilities are enhanced by well-designed spatial attention blocks as well as channel attention blocks combined with the UNet architecture. Moreover, a new localized image information measurement method based on histogram of oriented gradients and a spatial-aware loss function are proposed to make the fusion network more inclined to focus on the features of the region of interest. Extensive experimental results indicate that this method can more comprehensively reflect the brightness information and texture details of the source image, effectively addressing the challenges of fusion in complex environments.
## Framework
<img width="1421" alt="屏幕截图 2024-10-15 105140" src="https://github.com/user-attachments/assets/1af688cb-0d9c-4bd8-bcc7-ab9ca8607c25">

## Recommended Environment

 - [x] pytorch 1.12.1   
 - [x] numpy 1.11.3

## To Train
The training dataset is temporarily not publicly available. If needed, please contact the author for access.

    python train.py
## To Test
First, parameterize the structure of the trained model, and then run the testing program.

    python test_image.py
## Citation

```
@article{qian2024limfusion,
  title={LiMFusion: Infrared and visible image fusion via local information measurement},
  author={Qian, Yao and Tang, Haojie and Liu, Gang and Xing, Mengliang and Xiao, Gang and Bavirisetti, Durga Prasad},
  journal={Optics and Lasers in Engineering},
  volume={181},
  pages={108435},
  year={2024},
  publisher={Elsevier}
}
