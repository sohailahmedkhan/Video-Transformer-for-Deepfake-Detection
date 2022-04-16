# Video Transformer for Deepfake Detection with Incremental Learning
This is a PyTorch implementation of our ACM Multimedia 2021 paper titled, "Video Transformer for Deepfake Detection with Incremental Learning". Paper is available here: https://arxiv.org/abs/2108.05307 

![deepfakedetectionpipeline](https://user-images.githubusercontent.com/44908098/163416687-734bb78e-9bc6-436b-a6e7-4c5783e56a9b.png)

## Usage

We heavily rely on 3D dense face alignment (3DDFA) from here: https://github.com/cleardusk/3DDFA_V2 for UV texture map generation. We already packaged the 3DDFA_V2 repository into this repository. To build 3DDFA code within our repository, please follow instructions as below.

Tested on Linux and macOS. For Windows, please refer to this issue: https://github.com/cleardusk/3DDFA_V2#FQA.

**1: Clone this repo**

```python
git clone https://github.com/sohailahmedkhan/Video_Transformer_Deepfake_Detector.git
cd Video_Transformer_Deepfake_Detector
```

**2: Build the cython version of NMS, Sim3DR** 

```python
sh ./build.sh
```

**3: Train Image ViT**

```python
python train_Image_ViT.py
```


To train image transformer, run following command. NOTE: You need to add the location of train and validation directories. You can do this by manually editing "train_Image_ViT.py" file. 

To easily run training using Jupyter Notebook, you can use ImageTransformer.ipynb file.
NOTE: Video transformer training will be added soon. You can see the VideoTransformer's architecture in models directory, in file "videotransformer.py"

For any questions, please raise "Issues". 

## Acknowledgements

I would like to thank, (1) cleardusk for the 3DDFA implementation, (2) LukeMelas for transformer architecture and pre-trained ViT models, (3) Ross Wightman for XceptionNet.

## Cite

If you find this code useful, please cite the following:

@article{Khan2021VideoTF,
  title={Video Transformer for Deepfake Detection with Incremental Learning},
  author={Sohail Ahmed Khan and Hang Dai},
  journal={Proceedings of the 29th ACM International Conference on Multimedia},
  year={2021}
}

@inproceedings{guo2020towards,
    title =        {Towards Fast, Accurate and Stable 3D Dense Face Alignment},
    author =       {Guo, Jianzhu and Zhu, Xiangyu and Yang, Yang and Yang, Fan and Lei, Zhen and Li, Stan Z},
    booktitle =    {Proceedings of the European Conference on Computer Vision (ECCV)},
    year =         {2020}
}

@misc{3ddfa_cleardusk,
    author =       {Guo, Jianzhu and Zhu, Xiangyu and Lei, Zhen},
    title =        {3DDFA},
    howpublished = {\url{https://github.com/cleardusk/3DDFA}},
    year =         {2018}
}
