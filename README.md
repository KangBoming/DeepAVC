

# DeepAVC


<img src="virus.png" width="40" height="40" style="vertical-align: middle;margin-right: 5px;">  DeepAVC is a deep learning framework for highly accurate broad-spectrum antiviral compound prediction.

# Abstract
Lethal viruses pose a significant threat to human life, with each pandemic causing millions of fatalities globally. Small-molecule antiviral drugs provide an efficient and convenient approach to antiviral therapy by either inhibiting viral activity or activating the host immune system. However, conventional antiviral drug discovery is often labor-intensive and time-consuming due to the vast chemical space. Although some existing computational models mitigate this problem, there remains a lack of rapid and accurate method specifically designed for antiviral drug discovery. Here, we propose DeepAVC, a universal framework based on pre-trained large language models, for highly accurate broad-spectrum antiviral compound discovery, including DeepPAVC for phenotype-based prediction and DeepTAVC for target-based prediction. We demonstrate the power of DeepAVC in antiviral compound discovery through a series of in silico and in vitro experiments, identifying MNS and NVP-BVU972 as two novel potential antiviral compounds with promising broad-spectrum antiviral activities. 


## DeepPAVC: Phenotype-based antiviral compounds prediction
DeepPAVC model only takes compound information as input, utilizes a pre-trained molecular encoder to extract compound features, and outputs the antiviral activity score of the input compound 
![Overview](DeepPAVC.png)

## DeepTAVC: Target-based antiviral compounds prediction
DeepTAVC model requires input from two modalities: compounds and proteins. It employs two distinct pre-trained encoders to extract features from each modality separately, then captures intra- and inter- modality interaction patterns through self-attention and cross-attention mechanisms, and outputs the interaction score between the compound and the protein 
![Overview](DeepTAVC.png)



