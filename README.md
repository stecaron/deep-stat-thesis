This repo regroups all the work related my master's thesis at the *Département de mathématiques et de statistique de l'Université Laval* in Quebec City.

# Clustering digital behaviours

## Project context

We live in a world where the relationships between customers and companies are increasly shifting to digital interactions. Customers are expecting more user-friendly and personalized platforms and companies are generally willing to develop such technologies in order to improve efficiency on their side as well. However, those kinds of interactions are often *colder* and less adaptable to certain situations, which can lead to a customer's experience of lower quality when something *abnormal* happens. Furthermore, those solutions sometimes open the door to fraudulent or malicious activities, as less and less humans are involved in monitoring the process. The many financial and insurance companies facing this threat need a tradeoff procedure involving **straight-through processes** that improve the overall efficiency of most *normal* cases and **human assistance/review** for customers struggling to take full advantage of the way the process is designed.

## Project objectives

We want to tackle the problematic by learning digital behaviours in order to group users with respect to the way they fill the web/application-based form. We think the obtained clusters can help us acquire part of the knowledge traditionally acquired, either consciously or not, by experienced human assistants over the phone or during in-person conversations. Our hypothesis is that some behaviours like the following could be learned: hesitation, misunderstanding, confidence, pre-craft idea, dishonesty, etc. Accordingly, our objective is to represent such complex interactions with the web-based form in a lower-dimensional manifold. In particular, we wish to use a variational autoencoder (VAE), which is an unsupervised learning technique designed to do just that. This should ultimately allow us to group observations into a discrete set of *typical* behaviours. We wish to train our algorithm using web analytics data provided by an insurance company. The data was collected during the filling of an online form by the companies' clients in the event of a loss, in other words the declaration of a loss (first notice of loss or *FNOL*). Essentially, the data consist of the following collected tags :

- session's duration
- time required to answer each questions
- answers that have been edited
- answers content
- device used
- time of the day
- location

We hope that an unsupervised method such as VAE can learn useful caracteristics that differentiate users in their digital behaviours and ultimately capture some "body langage" that was traditionnaly measured by humans.

## Academic added-value

The main academic added-value of this paper is to study and compare clustering methods based on neural networks (specifically variational autoencoders) with well known statistical methods used for that purposes. We think that VAE can be better in a context of high-dimensionnal data and also capture non-linear relationship between observations.

# Collaborators

- Stéphane Caron (myself, M.Sc student)
- [Thierry Duchesne](https://www.mat.ulaval.ca/departement-et-professeurs/direction-personnel-et-etudiants/professeurs/fiche-de-professeur/show/duchesne-thierry/) (Thesis Director, Ph.D., P.Stat.)
- François-Michel de Rainville (Thesis Co-director, Ph.D. Computer Engineering)
- Intact Insurance (collaborating company)
