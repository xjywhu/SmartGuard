# Samsung SmartThings Internet of Things (IoT) datasets
This project is the repository of datasets used in "Accurate Action Recommendation for Smart Home via Two-Level Encoders and Commonsense Knowledge" (CIKM 2022).

We provide datasets collected from SmartThings which is a worldwide Internet of Things (IoT) platform with 62 million users. 
Our datasets consist of log datasets, routine datasets, and dictionaries.
The log datasets are collected for a month from 4 different countries: Korea, USA, Spain, and France, respectively.
The routine datasets are collected from 3 different regions: Asia-Pacific, North America, and Europe, respectively.
The dictionaries provide mappings between names and ids, for day of weeks, hours, devices, controls, and device controls.
We provide data separately by countries in repositories `kr`, `us`, `sp`, and `fr`, each of which corresponds to Korea, USA, Spain, and France, respectively.
### Log dataset
Each log dataset is a tensor of size [N x 10 x 5], where N is the number of instances.
Each instance of the log dataset contains ten consecutive actions performed by a user of Bixby service.
Each action of the instance is represented as a vector of length 5.
Elements of the vector represent day of week, hour, device, control, and device control in order.
Please note that the "control" element specifies the relative control ID assigned within each device.
We randomly split instances in 7:1:2 ratio, and provide them as: `trn_instance_10.pkl`, `vld_instance_10.pkl`, and `test_instance_10.pkl`.
### Routine dataset
A routine is a sequence of devices triggered by a pre-defined condition or a request from a user.
Each line of the routine dataset denotes a sequence of devices in a routine.
We provide the routine datasets as: `routine_device_corpus.txt`.
We have two versions of Europe routine datasets for Spain and France since the device ids for Spain and those for France are different.
### Dictionary
A Dictionary define name-to-id mappings of day of weeks, hours, devices, and device controls.
We provide the dictionaries as: `dictionary.py`.
### Citation
Please cite the following paper if you use our datasets.
```
@inproceedings{DBLP:conf/cikm/JeonKYLK22,
  author    = {Hyunsik Jeon and
               Jongjin Kim and
               Hoyoung Yoon and
               Jaeri Lee and
               U Kang},
  editor    = {Mohammad Al Hasan and
               Li Xiong},
  title     = {Accurate Action Recommendation for Smart Home via Two-Level Encoders
               and Commonsense Knowledge},
  booktitle = {Proceedings of the 31st {ACM} International Conference on Information
               {\&} Knowledge Management, Atlanta, GA, USA, October 17-21, 2022},
  pages     = {832--841},
  publisher = {{ACM}},
  year      = {2022},
  doi       = {10.1145/3511808.3557226},
}
```
