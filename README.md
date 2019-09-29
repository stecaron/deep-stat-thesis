This repo regroup all the work related with the thesis I made with regard to my Master Degree at the *Département de mathématiques et de statistique de l'Université Laval* in Quebec City.

# Digital behaviours clustering

## Project context

We live in a world where the relationships between customers and companies are increasly shifting to digital interactions. Customers are expecting more effectiveness and personalization in the platforms provided to them and companies are generally willing to develop such technologies to improve efficiency on their side as well. However, those kind of interactions are often *colder* and less adaptable to certain situations which can sometimes decrease the customer experience if something *abnormal* happens. Furthermore, those solutions can potentially open the door to more fraudulent or malicious behaviours as less and less humans are involved in the process. A lot of financial and insurance companies are often facing this tradeoff between **straight-through** processes that improve overall efficiency of most *normal* cases and **provides human assistance/review** at the right time for customers struggling or trying to take advantage of the process.

## Project objectives

In this project, we want to tackle that problematic by learning digital behaviours and grouping users by their way of filling a web/application based form. By creating those clusters of users, we think we can acquire part of the knowledge humans were collecting before by having phone or in-person conversations. Our hypothesis (and objective in a way) is that some behaviours like the following could be learned: hesitation, misunderstanding, confidence, pre-craft idea, dishonesty, etc. To achieve that, we want to apply variational autoencoders (VAE), an unsupervised learning method, to learn complex patterns that would allow us to group observations based on a learned hidden representation. We want to apply this algorithm to web analytics data collected from a online form provided by an insurance company to insureds to declare a loss (first notice of loss or *FNOL*). The data is essentially tags collected by the website/application :

- session's duration
- the time to answer each questions
- answers that have been edited
- answers content
- device used
- time of the day
- location

We hope that an unsupervised method such as VAE could learn useful caracteristics that differentiate users in their digital behaviours and ultimately capture some "body langage" that was traditionnaly measured by humans.

## Academic added-value

The main academic added-value of this paper is to study and compare clustering methods based on neural networks (specifically variational autoencoders) with well known statistical methods used for that purposes. We think that VAE can be better in a context of high-dimensionnal data and also capture non-linear relationship between observations.

# Collaborators

- Stéphane Caron (myself, M.Sc student)
- [Thierry Duchesne](https://www.mat.ulaval.ca/departement-et-professeurs/direction-personnel-et-etudiants/professeurs/fiche-de-professeur/show/duchesne-thierry/) (Thesis Director, Ph.D., P.Stat.)
- François-Michel de Rainville (Thesis Co-director, Ph.D. Computer Engineering)
- Intact Insurance (collaborating company)
