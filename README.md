
# DeepAVC


<img src="virus.png" width="40" height="40" style="vertical-align: middle;margin-right: 5px;">  DeepAVC is a deep learning framework for highly accurate broad-spectrum antiviral compound prediction.

## Abstract
Lethal viruses pose a significant threat to human life, with each pandemic causing millions of fatalities globally. Small-molecule antiviral drugs provide an efficient and convenient approach to antiviral therapy by either inhibiting viral activity or activating the host immune system. However, conventional antiviral drug discovery is often labor-intensive and time-consuming due to the vast chemical space. Although some existing computational models mitigate this problem, there remains a lack of rapid and accurate method specifically designed for antiviral drug discovery. Here, we propose DeepAVC, a universal framework based on pre-trained large language models, for highly accurate broad-spectrum antiviral compound discovery, including DeepPAVC for phenotype-based prediction and DeepTAVC for target-based prediction. We demonstrate the power of DeepAVC in antiviral compound discovery through a series of in silico and in vitro experiments, identifying MNS and NVP-BVU972 as two novel potential antiviral compounds with promising broad-spectrum antiviral activities. 


## DeepPAVC: Phenotype-based antiviral compounds prediction
DeepPAVC model only takes compound information as input, utilizes a pre-trained molecular encoder to extract compound features, and outputs the antiviral activity score of the input compound 
![Overview](DeepPAVC.png)

## DeepTAVC: Target-based antiviral compounds prediction
DeepTAVC model requires input from two modalities: compounds and proteins. It employs two distinct pre-trained encoders to extract features from each modality separately, then captures intra- and inter- modality interaction patterns through self-attention and cross-attention mechanisms, and outputs the interaction score between the compound and the protein 
![Overview](DeepTAVC.png)

## Publication
Bridging antiviral drug discovery with a large language model-powered framework

## Main requirements
* python=3.7.13
* pytorch=1.10.0
* cudatoolkit=11.3.1
* scikit-learn=1.0.2
* pandas=1.3.5
* numpy=1.21.5
* fair-esm=2.0.0
* rdkit=2021.09.2


## Quick start

**Step1: clone the repo**
```
mkdir ./DeepAVC
cd DeepAVC
git clone https://github.com/KangBoming/DeepAVC.git
```

**Step2: create and activate the environment**
```
cd DeepAVC
conda env create -f environment.yml
conda activate DeepAVC
```

**Step3: model training**
```
cd DeepAVC

Please follow DeepPAVC_train.ipynb

Please follow DeepTAVC_train.ipynb 
```

**Step4: model infernece**
```
cd DeepAVC

# Phenotype-based antivrial activity prediciton
Please follow DeepPAVC_inference.ipynb

# Target-based antivrial activity prediction
Please follow DeepTAVC_inference.ipynb

# General predition of compound-protein interaction
Please follow CADTI_inference.ipynb
```
## Web server
http://www.cuilab.cn/deepavc
![Overview](DeepPAVC.png)




## License
This project is licensed under the MIT License - see the [LICENSE.txt](https://github.com/KangBoming/DeepAVC/blob/main/LICENSE) file for details


## Contact
Please feel free to contact us for any further queations

Boming Kang <kangbm@bjmu.edu.cn>

Qinghua Cui <cuiqinghua@bjmu.edu.cn>
