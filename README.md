# How to Use
Using the model is very simple, do the following:
1. Download the Trained-Model from releases
2. Download Predicting notebook and put it in the same folder as Trained-Model
3. Create a Python venv, install libraries
4. change the image address  with the address of your own image(Images must be in Dicom format)

## What is Pneumonia?
Pneumonia is an inflammatory condition of the lung affecting primarily the tiny air sacs known as alveoli. It can be caused by bacteria, viruses, or fungi and often follows respiratory infections like the flu.  

![image](https://github.com/parham2013/Pneumonia-Classification-PyTorch/assets/74326920/aacedd80-d138-4f37-839f-572c9bb137e2)  
Image Source: [Wikipedia](https://simple.wikipedia.org/wiki/Pneumonia)  

## Key facts
* Pneumonia accounts for 14% of all deaths of children under 5 years old, killing 740 180 children in 2019.
* Pneumonia can be caused by viruses, bacteria or fungi.
* Pneumonia can be prevented by immunization, adequate nutrition, and by addressing environmental factors.
* Pneumonia caused by bacteria can be treated with antibiotics, but only one third of children with pneumonia receive the antibiotics they need.

Source: [World Health Organization (WHO)](https://www.who.int/news-room/fact-sheets/detail/pneumonia)  

![Chest_radiograph_in_influensa_and_H_influenzae,_posteroanterior,_annotated](https://github.com/parham2013/Pneumonia-Classification-PyTorch/assets/74326920/5798431c-74f0-45fb-9562-373ab540905a)  
Image Source: [Wikipedia](https://en.wikipedia.org/wiki/Pneumonia)


## Epidemiology of Pneumonia
Pneumonia is a leading cause of death among children and adults worldwide.
Pneumonia affects children and families everywhere, but deaths are highest in southern Asia and sub-Saharan Africa. Children can be protected from pneumonia, it can be prevented with simple interventions, and it can be treated with low-cost, low-tech medication and care.  
Source: [Wikipedia](https://en.wikipedia.org/wiki/Pneumonia)

## How Does Machine Learning Help?
Machine learning algorithms have shown promise in improving the detection, diagnosis, and treatment planning of pneumonia.  
Source: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0933365796003673)


* Detection: Machine learning can analyze X-rays and other imaging data much faster and sometimes more accurately than humans.
* Diagnosis: Algorithms can process and identify patterns in large datasets that might be difficult for a human to analyze.
* Treatment Planning: Predictive algorithms can help in determining the best treatment options based on historical data and patient-specific factors.

Sources:  
- [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning
](https://arxiv.org/abs/1711.05225)  
- [PDF Document](https://web.njit.edu/~usman/courses/cs732_spring19/CheXNet_Yanan%20Yang.pdf)



## Contributions of This Project
Dataset used for this project:
https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

This project aims to build a machine learning model for the early detection of pneumonia using chest X-ray images.

The model achieved an accuracy of 80% and a precision of 54% on the validation set.

Utilizes Class Activation Maps (CAM) for better understanding of model decisions.

tensor([True]) means positive Pneumonia  
![Pneumonia-CAM](https://github.com/parham2013/Pneumonia-Classification-PyTorch/assets/74326920/9e8539d3-57fc-4ad6-89f7-4480b088c31f)

---

Project consists of 3 sections:  
* Preprocessing
* Model-Training
* Interpretability

  
### Preprocessing
Loading data, normalizing and separating train-validation sets and saving them in another folder.

Calculating mean and std of pixel arrays for further normalization in Model-Training step.

### Model-Training

We start off with normalizing(again) and augmenting images, we use ResNet18 as a base model, Adam Optimizer and Binary Cross-Entropy with Logits Loss function,
saving top 10 models after 35 epochs and validating the model.

Model validation:
* Val Accuracy 0.8010432124137878
* Val Precision 0.5409457683563232
* Val Recall 0.7752066254615784
* Val ConfusionMatrix tensor(
  
 [[1681,  398],
  
[ 136,  469]
])

* Accuracy is not very helpful alone
* Recall is much larger than Precision, it means our model rarely misses a case of Pneumonia
* But low Precision tells us that many images without Pneumonia is also classified as Pneumonic

### Interpretability

Since we care about the reason why our model classifies an image as positive, we used Class Activation Maps to show what our model thinks.
Warmer areas in heatmaps indicate signs of Pneumonia:

tensor([True]) means positive Pneumonia  
![Pneumonia-CAM-02](https://github.com/parham2013/Pneumonia-Classification-PyTorch/assets/74326920/4401ace4-1130-49af-af51-4489a41a5e6c)

