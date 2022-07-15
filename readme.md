# GraphDoc
The source code for [Multimodal Pre-training Based on Graph Attention Network for Document Understanding](https://arxiv.org/abs/2203.13530).

## Requirements
* torch==1.7.1
* mmdet==2.16.0
* transformers==4.6.0

## Pretrained-Model
We provide pretrained model required for downstream tasks, the download link is https://rec.ustc.edu.cn/share/031c0580-0366-11ed-bb15-47281881a56b.
The user should unzip the pretrained model to the base folder, formed as graphdoc/pretrained_model.

## Usage
We provide a example code for extracting document representation with GraphDoc in the runner/graphdoc/encode_document.py

## Citation
If you find GraphDoc useful in your research, please consider citing:

    @article{zrzhang2022graphdoc,
        author = {Zhang, Zhenrong and Ma, Jiefeng and Du, Jun and Wang, Licheng and Zhang, Jianshu},
        title = {Multimodal Pre-training Based on Graph Attention Network for Document Understanding},
        journal = {arXiv},
        year = {2022},
        volume={abs/2203.13530}
    }

## Contact
zzr666@mail.ustc.edu.cn<br>