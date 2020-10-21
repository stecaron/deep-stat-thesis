This repo contains all the work related to my master's thesis at the *Département de mathématiques et de statistique de l'Université Laval* in Quebec City.

# Reliable anomaly detection with variational autoencoders

## Context

Anomaly detection is a challenging topic that generated a lot of research in multiple fields such as statistics, machine learning, computer vision, etc. An anomaly, or what can also be called an outlier, is something intrinsic to all those fields because outliers are by nature, something interesting to extract or to remove from a given source of data. It can be interesting to extract, because it is what we are looking for in the problem or it could interesting to remove prior to another learning problem. One challenge with anomaly detection is that we often deal with unlabeled data, that means we have to tackle the problem in an unsupervised manner. Another important challenge is that those detection algorithms often need a threshold. That threshold allows us to take a decision regarding a certain anomaly. How do we set that threshold, or in others words, how do we say that observation is sufficiently different to be anormal? In some real life situations, we could have in hand a certain dataset or a historic of observations we know to be entirely normal (or at least almost). Normal means the contrary of anomaly here, not the distribution ... However, we anticipate anormal observations in the future, but we don't really know how much we will have, will they be variable in the future, what will they will look like, etc. In that sense, that project aims to be able to represent (or learn) what is normal and also detect anomalies using a intuitive filtration level rather than a threshold calculated on a distance.

## Project description

### Objectives

The main objective of this project is indeed to be able to detect anomalies, but with an unsupervised algorithm. Afterward, we want to be able to detect anomalies using an intuitive threshold approach so that the detection decision is easier to take. Finally, we want to be able to do those things on complex data structures, such as images. Our algorithm will detect anomalies at the "observation level", not part of an observation (i.e. detect anomalies within an image).

### Detailed approach

The approach chosen is to use autoencoders, more precisely varitional autoencoders, to learn a simpler representation of complex observations. An autoencoders is an unsupervised neural networks algorithm that is meant to predict the input as its output. However, it needs to do it by encoding the information in a certain representation, and then decodes it back to the original format by loosing less information as possible. Varitional autoencoders (VAE) are a specific case of autoencoders where the model is not just trying to predict back the inputs by encoding a useful representation, but also to force that representation to have a certain distribution. That known distribution, will allows us to have clear idea of what should be an inlier and what should be an outlier. In fact, if the VAE is able to understand a certain input, encodes it into a simpler representation that we known some properties, we will be able to interpret that representation for new observations and conclude if that observation is an anomaly or not by using a intuitive filtration level.

### Academic added-value

The academic added-value of that project is to put togheter the statistical properties of Kullback-Leibler distance and the power of autoencoders to learn complex data structures and encoded them in simpler representations. At the end, we will present a new approach to detect anomalies out of complex images.

## Reproduce the results

To reproduce the results you will first need to download the data :

```
# create a folder to save the data
mkdir -p ~/data
# download stanford cars
wget -c http://imagenet.stanford.edu/internal/car196/cars_train.tgz
tar zxvf cars_train.tgz -C ~/data
mv ~/data/cars_train ~/data/stanford_cars
rm cars_train.tgz
# download stanford dogs
wget -c http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
tar -xvf images.tar -C ~/data
mkdir ~/data/stanford_dogs2
find ~/data/Images/ -type f -print0 | xargs -0 mv -t ~/data/stanford_dogs2
rm -rf ~/data/Images
rm images.tar
# download indoor scene dataset
wget -c http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar
tar -xvf indoorCVPR_09.tar -C ~/data
find ~/data/Images/ -type f -print0 | xargs -0 mv -t ~/data/stanford_dogs2
rm -rf ~/data/Images
rm indoorCVPR_09.tar
```

Then, you can run different experiments by using the command examples from `results/launch_*` files.

Finally, to gather all results from multiple experiments, you can use the python script `src/utils/compile_results.py`

# Collaborators

- [Stéphane Caron](https://www.researchgate.net/profile/Stephane_Caron4) (Myself, M.Sc student)
- [Thierry Duchesne](https://www.mat.ulaval.ca/departement-et-professeurs/direction-personnel-et-etudiants/professeurs/fiche-de-professeur/show/duchesne-thierry/) (Thesis Director, Ph.D., P.Stat.)
- [François-Michel de Rainville](https://www.researchgate.net/profile/Francois-Michel_De_Rainville) (Thesis Co-director, Ph.D. Computer Engineering)
- [Samuel Perreault](https://www.researchgate.net/profile/Samuel_Perreault) (Collaborator)
- [Intact Insurance](https://www.intactlab.ca/) (Collaborating company, Intact Lab)
