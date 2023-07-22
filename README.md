# MultiTQ
This is the code for the paper [Multi-granularity Temporal Question Answering over Knowledge Graphs](https://aclanthology.org/2023.acl-long.637) (Chen et al., ACL 2023).

![Architecture of MultiQA](https://img1.imgtp.com/2023/07/18/1rXQMVDG.png)

## Dataset and pretrained models

MultiTQ dataset can be found in ./data folder.

```bash
git clone https://github.com/czy1999/MultiTQ.git

cd ./MultiTQ/data
unzip Dataset.zip

cd ../MultiQA/models
unzip Models.zip
```





## Running the code

MultiQA on MultiTQ:
```
python ./train_qa_model.py --model multiqa
 ```


Please explore more argument options in train_qa_model.py.




The implementation is based on TempoQR in [TempoQR: Temporal Question Reasoning over Knowledge Graphs](https://arxiv.org/abs/2112.05785) and their code from https://github.com/cmavro/TempoQR. You can find more installation details there.
We use TComplEx KG Embeddings as implemented in https://github.com/facebookresearch/tkbc.

## Cite

If you find our method, code, or experimental setups useful, please cite our paper:



```
@inproceedings{chen-etal-2023-multi,
    title = "Multi-granularity Temporal Question Answering over Knowledge Graphs",
    author = "Chen, Ziyang  and
      Liao, Jinzhi  and
      Zhao, Xiang",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.637",
    pages = "11378--11392",
}
```

```
[1] Ziyang Chen, Jinzhi Liao, and Xiang Zhao. 2023. Multi-granularity Temporal Question Answering over Knowledge Graphs. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11378–11392, Toronto, Canada. Association for Computational Linguistics.
```