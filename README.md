This repo contains all the work related to my master's thesis at the *Département de mathématiques et de statistique de l'Université Laval* in Quebec City.

# Reliable anomaly detection with variational autoencoders

## Context

Anomaly detection is a challenging topic that generated a lot of research in multiple fields such as statistics, machine learning, computer vision, etc. An anomaly, or what can also be called an outlier, is something intrinsic to all those fields because outliers are by nature, something interesting to extract or to remove from a given source of data. It can be interesting to extract, because it is what we are looking for in the problem or it could interesting to remove prior to another learning problem. One challenge with anomaly detection is that we often deal with unlabeled data, that means we have to tackle the problem in an unsupervised manner. Another important challenge is that those detection algorithms often need a threshold. That threshold allows us to take a decision regarding a certain anomaly. How do we set that threshold, or in others words, how do we say that observation is sufficiently different to be anormal? In some real life situations, we could have in hand a certain dataset or a historic of observations we know to be entirely normal (or at least almost). Normal means the contrary of anomaly here, not the distribution ... However, we anticipate anormal observations in the future, but we don't really know how much we will have, will they be variable in the future, what will they will look like, etc. In that sense, that project aims to be able to represent (or learn) what is normal and also detect anomalies using a confidence level more that a threshold based on a metric.

## Project description

### Objectives

The main objective of this project is indeed to be able to detect anomalies, but with an unsupervised algorithm. Afterward, we want to be able to detect anomalies using a confidence level so that the detection decision is based on a confidence level and not a fixed threshold. Finally, we want to be able to do those things on complex data structures, such as images. However, our algorithm will detect anomalies at the "observation level", not part of an observation (i.e. detect anomalies within an image).

### Detailed approach

The approach chosen is to use autoencoders, more precisely varitional autoencoders, to learn a simpler representation of complex observations. An autoencoders is an unsupervised neural networks algorithm that is meant to predict the input as its output. However, it needs to do it by encoding the information in a certain representation, and then decodes it back to the original format by loosing less information as possible. Varitional autoencoders (VAE) are a specific case of autoencoders where the model is not just trying to predict back the inputs by encoding a useful representation, but also to force that representation to have a certain distribution. That known distribution, will allows us to have clear idea of what should be an inlier and what should be an outlier. In fact, if the VAE is able to understand a certain input, encodes it into a simpler representation that we known some properties, we will be able to interpret that representation for new observations and conclude with hypotheses testing if that observation is an anomaly or not by using a confidence level and not a fixed threshold.

### Academic added-value

The academic added-value of that project is to put togheter the stastical theory behind hypotheses testing and the power of autoencoders to learn complex data structures and encoded them in simpler representations. That way, we are adding a statistical confidence level in components learned by complex algorithms such as neural networks. At the end, we will present another kind of anomaly detection algorithm. 

### Application

In this project, we want to apply our research to a specific imagery application associated with cars insurance. In fact, people making accidents have the chance to take photos of their car and send them to the insurer in order to accerate the claim process. To do proper evaluation, the insurer needs specific photos of the car from different angles (front, behind, left and right side, interior, etc). With our algorithm, we would like to be able to learn what each of those images look like in order to provide instant feedback to the user about a taken photo. For example, if a front picture does not look like a front car, we could ask him: "Are you sure that picture is the front of your car?". Anomalies could be caused by the user taking the wrong angle or just a false manipulation (ex: picture of the ground or the sky). To do so, we need develop a framework where the model basically learns what is normal, and then provide confidence level about what deviates from that normality.

# Collaborators

- [Stéphane Caron](https://www.researchgate.net/profile/Stephane_Caron4) (Myself, M.Sc student)
- [Thierry Duchesne](https://www.mat.ulaval.ca/departement-et-professeurs/direction-personnel-et-etudiants/professeurs/fiche-de-professeur/show/duchesne-thierry/) (Thesis Director, Ph.D., P.Stat.)
- [François-Michel de Rainville](https://www.researchgate.net/profile/Francois-Michel_De_Rainville) (Thesis Co-director, Ph.D. Computer Engineering)
- [Samuel Perreault](https://www.researchgate.net/profile/Samuel_Perreault) (Collaborator)
- [Intact Insurance](https://www.intactlab.ca/) (Collaborating company, Intact Lab)
