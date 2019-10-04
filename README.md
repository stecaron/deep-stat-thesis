This repo contains all the work related to my master's thesis at the *Département de mathématiques et de statistique de l'Université Laval* in Quebec City.

# Clustering digital behaviours

## Project context

We live in a world where the relationships between customers and companies are increasly shifting to digital interactions. Customers are expecting more user-friendly and personalized platforms and companies are generally willing to develop such technologies in order to improve efficiency on their side as well. However, automated interactions tend to be *colder* and less adaptable, which can lower the quality of the customer's experience when something *abnormal* happens. Furthermore, those automated solutions sometimes open the door to fraudulent or malicious activities, as less and less humans are involved in monitoring the process. The many financial and insurance companies facing this threat need a tradeoff procedure involving **straight-through processes** that improve the overall efficiency of most *normal* cases and **human assistance/review** for customers struggling to take full advantage of the way the process is designed.

## Project objectives

We want to tackle the problematic by learning digital behaviours to group users with respect to the way they fill the web/application-based form. We think a learning algorithm can internalize part of the knowledge traditionally acquired, either consciously or not, by experienced human assistants during their many conversations over the phone and/or in-person with clients. Our hypothesis is that some behaviours like the following can be learned: hesitation, misunderstanding, confidence, pre-craft idea, dishonesty, etc. Accordingly, our objective is to represent sets of interactions with the web-based form in a low-dimensional subspace, to then cluster these representations into a discrete set of *typical* behaviours. To achieve the first part of the objective, we intend to use a variational autoencoder (VAE), an unsupervised learning technique designed to do just that. Web analytics data provided by an insurance company will be used to train the algorithm. The data was collected during the filling of an online form by the companies' clients declaring a loss (first notice of loss or *FNOL*). Essentially, the data consist of the following collected tags :

- session's duration
- time required to answer each questions
- answers that have been edited
- answers content
- device used
- time of the day
- location

We hope that an unsupervised method such as a VAE can learn useful caracteristics that differentiate users in their digital behaviours and ultimately capture some "body langage" that was traditionnaly measured by humans.

## Academic added-value

The main academic added-value of this paper is the study and comparison of clustering methods based on neural networks (specifically variational autoencoders) with well known statistical methods used for that purposes. We think that a VAE can perform better than conventional statistical methods in the context of high-dimensionnal data, among oter things because it can capture non-linear relationships between variables.

# Collaborators

- Stéphane Caron (myself, M.Sc student)
- [Thierry Duchesne](https://www.mat.ulaval.ca/departement-et-professeurs/direction-personnel-et-etudiants/professeurs/fiche-de-professeur/show/duchesne-thierry/) (Thesis Director, Ph.D., P.Stat.)
- François-Michel de Rainville (Thesis Co-director, Ph.D. Computer Engineering)
- Intact Insurance (collaborating company)
