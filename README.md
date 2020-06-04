# Instance Shadow Detection (CVPR’ 20)

[Tianyu Wang](https://stevewongv.github.io)\*, [Xiaowei Hu](https://xw-hu.github.io)\*, Qiong Wang,  Pheng-Ann Heng,  and [Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/)
 (\* Joint first authors.)

[[`arXiv`]()] [[`BibTeX`](#CitingLISA)]


[-c](projects/LISA/web-shadow0573.jpg)

Instance shadow detection aims to find shadow instances paired with object instances. We present a dataset, a deep framework, and an evaluation metric to approach this new task.



## Installation 

```bash
$ cd InstanceShadowDetection
$ python setup.py install
$ cd PythonAPI
$ python setup.py install
```

## Docker
```bash
$ cd InstanceShadowDetection/docker

$ docker build --tag="instanceshadow" ./Dockerfile .
```

## Demo

```bash
$ cd projects/LISA/
$ python demo.py --input ./demo/web-shadow0573.jpg --output ./ --config ./config/LISA_101_FPN_3x_demo.yaml
```

## Train

```bash
$ python train_net.py --num-gpus 2 --config-file ./config/LISA_101_FPN_3x.yaml

```
## Evaluation

```bash
$ python train_net.py --num-gpus 2 --config-file ./config/LISA_101_FPN_3x.yaml --eval-only --resume
```

## Visualize
```bash
python visualize_json_results.py --ins_input ./output_light/inference/soba_instances_results.json --ass_input ./output_light/inference/soba_association_results.json --output ./output_light/results --dataset soba_cast_shadow_val_full
```
## <a name="CitingLISA"></a> Citation
If you use LISA, SOBA, or SOAP, please use the following BibTeX entry.

```
@InProceedings{Wang_2020_Instance,
  title = {Instance Shadow Detection},
  author = {Wang, Tianyu and Hu, Xiaowei and Wang, Qiong and Heng, Pheng-Ann and Fu, Chi-Wing},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year={2020}
}

```
