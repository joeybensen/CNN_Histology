# Classification of Breast Cancer Histology Using Pretrained CNNs

Breast cancer is one of the most terminal cancers among women, while analysis of biopsies remains the gold standard for diagnosis. 
Thus, integration of Convolution Neural Networks (CNNs) into the analysis and classification of histology imaging has significantly advanced the field of pathology. 
This study will focus on developing a CNN to classify histological images to assist in breast cancer diagnoses. More specifically, we will be leveraging transfer 
learning via separate pre-trained models on a dataset of Hematoxylin and Eosin (H&E) stained breast biopsy microscopy images. The approach utilizes fine-tuning of pretrained models, 
image processing and augmentation, and dynamic learning rate scheduling.

## Features

-Importing and installing the BACH image huggingface dataset<br>
-Reorganizing and converting images to tensors<br>
-Importing pretrained models (ResNet 18 and ResNet 34)<br>
-provding metrics and example labelling of images


## Running the Program

Clone the repository and run via terminal:

```bash
python3 CNN_Histology.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.
