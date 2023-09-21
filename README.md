# What is Pneumonia?
Pneumonia is an inflammatory condition of the lung affecting primarily the tiny air sacs known as alveoli. It can be caused by bacteria, viruses, or fungi and often follows respiratory infections like the flu.

# Key facts
* Pneumonia accounts for 14% of all deaths of children under 5 years old, killing 740 180 children in 2019.
* Pneumonia can be caused by viruses, bacteria or fungi.
* Pneumonia can be prevented by immunization, adequate nutrition, and by addressing environmental factors.
* Pneumonia caused by bacteria can be treated with antibiotics, but only one third of children with pneumonia receive the antibiotics they need.

Source: [World Health Organization (WHO)](https://www.who.int/news-room/fact-sheets/detail/pneumonia)

![Chest_radiograph_in_influensa_and_H_influenzae,_posteroanterior,_annotated](https://github.com/parham2013/Pneumonia-Classification-PyTorch/assets/74326920/5798431c-74f0-45fb-9562-373ab540905a)

Image Source: [Wikipedia](https://en.wikipedia.org/wiki/Pneumonia)


# Epidemiology of Pneumonia
Pneumonia is a leading cause of death among children and adults worldwide.
Pneumonia affects children and families everywhere, but deaths are highest in southern Asia and sub-Saharan Africa. Children can be protected from pneumonia, it can be prevented with simple interventions, and it can be treated with low-cost, low-tech medication and care.

Source: [Wikipedia](https://en.wikipedia.org/wiki/Pneumonia)

# How Does Machine Learning Help?
Machine learning algorithms have shown promise in improving the detection, diagnosis, and treatment planning of pneumonia.
Source: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0933365796003673)


* Detection: Machine learning can analyze X-rays and other imaging data much faster and sometimes more accurately than humans.
* Diagnosis: Algorithms can process and identify patterns in large datasets that might be difficult for a human to analyze.
* Treatment Planning: Predictive algorithms can help in determining the best treatment options based on historical data and patient-specific factors.

Sources:

- [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning
](https://arxiv.org/abs/1711.05225)

- [PDF Document](https://web.njit.edu/~usman/courses/cs732_spring19/CheXNet_Yanan%20Yang.pdf)



# Contributions of This Project
Dataset used for this project:
https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

This project aims to build a machine learning model for the early detection of pneumonia using chest X-ray images.

The model achieved an accuracy of 80% and a precision of 54% on the validation set.

Utilizes Class Activation Maps (CAM) for better understanding of model decisions.

![Pneumonia-CAM](https://github.com/parham2013/Pneumonia-Classification-PyTorch/assets/74326920/9e8539d3-57fc-4ad6-89f7-4480b088c31f)

---

Project consists of 3 sections, 
* Preprocessing
* Model-Training
* Interpretability

  ## Preprocessing
  Loading data, normalizing and separating train-validation sets and saving them in another folder.

  Calculating mean and std of pixel arrays for further normalization in Model-Training step.

![Preprocessing](https://github.com/parham2013/Pneumonia-Classification-PyTorch/assets/74326920/2d843175-cbbc-47b3-9fc6-94c6aca42e1a)

## Model-Training

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

## Interpretability

Since we care about the reason why our model classifies an image as positive, we used Class Activation Maps, as you can see, it's a heatmap:

![Pneumonia-CAM-02](https://github.com/parham2013/Pneumonia-Classification-PyTorch/assets/74326920/4401ace4-1130-49af-af51-4489a41a5e6c)

To classify an image outside of validation dataset, you can uncomment the last cell in Interpretability notebook and give it the image path, the code in the cell is the following:

```
import pydicom
import cv2
import numpy as np

# Step 1: Read the DICOM file
dcm_path = "path/to/your/image.dcm"
dcm = pydicom.read_file(dcm_path).pixel_array

# Step 2: Normalize and Resize
dcm = dcm / 255
dcm_array = cv2.resize(dcm, (224, 224)).astype(np.float32)

# Step 3: Apply Transforms
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.49, 0.248),
])
img_tensor = val_transforms(dcm_array)

# Step 4: Move to Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_tensor = img_tensor.to(device).unsqueeze(0)  # Add a batch dimension

# Step 5: Run through Model
# Assume 'model' is your trained model
activation_map, pred = cam(model, img_tensor)

# Visualizing
visualize(img.cpu().numpy(), activation_map, pred)
```
