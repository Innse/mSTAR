# Report generation

This folder cloned the [source code](https://github.com/dddavid4real/HistGen) from paper "HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction". See the paper for a detailed description of **HistGen**.

**HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction**\
*Zhengrui Guo, Jiabo Ma, Yingxue Xu, Yihui Wang, Liansheng Wang, and Hao Chen*\
Paper: <https://arxiv.org/abs/2403.05396>

<!-- Link to our paper: [[arxiv]](https://arxiv.org/abs/2403.05396) -->

### Methodology
![](methodology.png)
Overview of the proposed HistGen framework: (a) local-global hierarchical encoder module, (b) cross-modal context module, (c) decoder module, (d) transfer learning strategy for cancer diagnosis and prognosis.


## HistGen WSI Report Generation Model
### Training
To try our model for training, validation, and testing, simply run the following commands:
```
cd scripts
sh train_TCGA.sh
```
Before you run the script, please set the path and other hyperparameters in `train_TCGA.sh`. Note that **--image_dir** should be the path to the **mSTAR** feature directory, and **--ann_path** should be the path to the **TCGA_mSTAR.json** file.

### Inference
To generate reports for WSIs in test set, you can run the following commands:
```
cd scripts
sh test_TCGA.sh
```
Similarly, remember to set the path and other hyperparameters in `test_TCGA.sh`.


## License and Usage
If you find this work useful in your research, please consider citing this paper at:
```
@article{guo2024histgen,
  title={HistGen: Histopathology Report Generation via Local-Global Feature Encoding and Cross-modal Context Interaction},
  author={Guo, Zhengrui and Ma, Jiabo and Xu, Yingxue and Wang, Yihui and Wang, Liansheng and Chen, Hao},
  journal={arXiv preprint arXiv:2403.05396},
  year={2024}
}
```
