You can find here framework to learn and test Steganography Neural Nnetwork models.

Templates:
- Encoder-Decoder model template
- GAN model template

Also, there is Tester model, but implemented testing network is of poor quality
and has to be improved.

You can find here few models, for more accurate information look into files:
- e2eCNN, model strongly inspired by [4];
- e2eGAN, previous model with added discriminator;
- SGAN, model slightly inspired by [12] (check description in file);
- e2eCGAN, model e2eGAN with discriminator additionally taking color image,
check [12];
- e2eGGAN, model e2eGAN with discriminator additionally taking data to hide,
model with best visual results;
- e2eCryptoGan, steganography model with a random key shared by
encoder and decoder.

Papers to which I refer in original work:

[1] *Hiding Images in Plain Sight: Deep Steganography*, **Shumeet Baluja**,
NIPS 2017, https://research.google/pubs/pub46526/.

[2] *Cryptography Based On Neural Network*,**Eva Volna, Martin Kotyrba, Vaclav Kocian,
Michal Janosek**, Department of Informatics and Computers, Uni-
versity of Ostrava, 05.2012, https://www.researchgate.net/publication/267960477_Cryptography_Based_On_Neural_Network.

[3] *CryptoNN: Training Neural Networks over Encrypted Data*, **Runhua Xu, James B.D.
Joshi, Chao Li**, School of Computing and Information, University of Pittsburgh, 04.2019,
https://arxiv.org/abs/1904.07303.

[4] *End-to-end Trained CNN Encode-Decoder Networks for Image Steganography*, **Atique ur
Rehman, Rafia Rahim, Shahroz Nadeem, Sibt ul Hussain**, Islamabad, Pakistan,
11.2017, https://arxiv.org/abs/1711.07201.

[5] *Generative Adversarial Nets*, **Ian J. Goodfellow, Jean Pouget-Abadie,
Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua
Bengio**, Montre’al, 06.2014, https://arxiv.org/abs/1406.2661.

[6] *Generative Reversible Data Hiding by Image-to-Image Translation via GANs*, **Zhuo
Zhang, Guangyuan Fu, Fuqiang Di, Changlong Li, Jia Liu**, China, 07.2019,
https://arxiv.org/abs/1905.02872.

[7] *Invisible Steganography via Generative Adversarial Networks*, **Ru Zhang, Shiqi Dong,
Jianyi Liu**, China, 10.2018 (org. 07.2018), https://arxiv.org/abs/1807.08571.

[8] *LR-GAN: Layered Recursive Generative Adversarial Networks for Image Generation*,
**Jianwei Yang, Anitha Kannan, Dhruv Batra, Devi Parikh**, ICLR 2017, https://arxiv.org/abs/1703.01560.

[9] *Recent Advances of Image Steganography with Generative Adversarial Networks*, **Jia Liu,
Yan Ke, Yu Lei, Zhuo Zhang, Jun Li, Peng Luo, Minqing Zhang, Xiaoyuan
Yang**, 07.2019, https://arxiv.org/abs/1907.01886.

[10] *Recursive Conditional Generative Adversarial Networks for Video Transformation*, **SAN
KIM, DOUG YOUNG SUH**, IEEE Access, 03.2019, https://ieeexplore.ieee.org/document/8673567.

[11] *Revisiting Small Batch Training for Deep Neural Networks*, **Dominic Masters, Carlo
Luschi**, UK 04.2018, https://arxiv.org/abs/1804.07612.

[12] *SteganoGAN: High Capacity Image Steganography with GANs*, **Kevin A. Zhang,
Alfredo Cuesta-Infante, Lei Xu, Kalyan Veeramachaneni**, 01.2019, https://arxiv.org/abs/1901.03892, https://github.com/DAI-Lab/SteganoGAN.

[13] *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks*,
**Wenjie Luo, Yujia Li, Raquel Urtasun, Richard Zemel**,
29th Conference on Neural Information Processing Systems (NIPS 2016), Barcelona, Spain, 01.2017, https://arxiv.org/abs/1701.04128.

[14] *Improving the Backpropagation Algorithm with Consequentialism Weight Updates
overMini-Batches*, **Naeem Paeedeha, Kamaledin Ghiasi-Shirazi**, 03.2020, https://arxiv.org/abs/2003.05164.

[15] *An introduction to artificial neural networks*, **Coryn A.L. Bailer-Jones, Ranjan
Gupta, Harinder P. Singh**, Automated Data Analysis in Astronomy, 03.2001,
https://www.researchgate.net/publication/2228238_An_introduction_to_artificial_neural_networks.

[16] *A New Backpropagation Algorithm without Gradient Descent*, **Varun Ranganathan, S.
Natarajan**, 01.2018, https://arxiv.org/abs/1802.00027.

[17] *An Introduction to Convolutional Neural Network*, **Keiron O’Shea, Ryan Nash**,
12.2015, https://arxiv.org/abs/1511.08458.

[18] *Deformable Convolutional Networks*, **Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li,
Guodong Zhang, Han Hu, Yichen We, Microsoft Research Asia**, 06.2017, https://arxiv.org/abs/1703.06211.

[19] *Dynamical Variational Autoencoders: A Comprehensive Review*, **Laurent Girin, Simon
Leglaive, Xiaoyu Bie, Julien Diard, Thomas Hueber, Xavier Alameda-Pineda**,
08.2020, https://arxiv.org/abs/2008.12595.

[20] *Checkerboard artifact free sub-pixel convolution – A note on sub-pixel convolution, resize
convolution and convolution resize*, **Andrew Aitken, Christian Ledig, Lucas Theis,
Jose Caballero, Zehan Wang, Wenzhe Shi**, 07.2017, https://arxiv.org/abs/1707.02937.

[21] *Explaining and Harnessing Adversarial Examples*, **J. Goodfellow, Jonathon Shlens,
Christian Szegedy**, ICLR 2015, 03.2015, https://arxiv.org/abs/1412.6572.

[22] *A guide to convolution arithmetic for deep learning*, **Vincent Dumoulin, Francesco Visin**,
01.2018, https://arxiv.org/abs/1603.07285, https://github.com/vdumoulin/conv_arithmetic.

[23] *Deep Learning Hierarchical Representations for Image Steganalysis*, **Jiangqun Ni,
Jian Ye, Yang YI**, IEEE Transactions on Information Forensics and Security,
06.2017,
 https://www.researchgate.net/publication/317294735_Deep_Learning_Hierarchical_Representations_for_Image_Steganalysis.

[24] *The Open Images Dataset V4: Unified image classification, object detection, and visual
relationship detection at scale*, **Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper
Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo
Malloci, Alexander Kolesnikov, Tom Duerig, Vittorio Ferrari**, IJCV, 02.2020,
https://arxiv.org/abs/1811.00982.

[25] *OpenImages: A public dataset for large-scale multi-label and multi-class image classification*,
**Krasin, Ivan and Duerig, Tom and Alldrin, Neil and Ferrari, Vittorio
and Abu-El-Haija, Sami and Kuznetsova, Alina and Rom, Hassan and Uij-
lings, Jasper and Popov, Stefan and Kamali, Shahab and Malloci, Matteo
and Pont-Tuset, Jordi and Veit, Andreas and Belongie, Serge and Gomes,
Victor and Gupta, Abhinav and Sun, Chen and Chechik, Gal and Cai, Da-
vid and Feng, Zheyun and Narayanan, Dhyanesh and Murphy, Kevin**, 2017
https://storage.googleapis.com/openimages/web/index.html.

This code is part of the University thesis (Bachelor of Science, math).
