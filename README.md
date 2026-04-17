# Projet XAI - RISE et Grad-CAM

## Présentation

Dans ce projet, on s'intéresse à l'explicabilité des modèles, c'est-à-dire essayer de comprendre pourquoi un modèle fait une prédiction.

Le but principal est de tester la méthode RISE, puis de la comparer à Grad-CAM sur des exemples simples.

On ne regarde pas seulement les heatmaps : on essaie aussi de les évaluer avec des métriques pour voir si les explications sont vraiment utiles.

## Ce qu'on fait dans le projet

Concrètement,

- on applique **RISE** sur des images ;
- on compare les résultats avec **Grad-CAM** ;
- on utilise **deux métriques** pour comparer les méthodes :
  - **insertion**
  - **deletion**

- on regarde aussi les **résultats visuellement** pour voir si les zones importantes semblent cohérentes ;
- on étudie aussi l'effet de certains **hyperparamètres** sur les résultats;
- on adapte ensuite l'idée de **RISE à des séquences d'acides aminés**.


## Idée générale

L'idée est assez simple :

- **Grad-CAM** utilise les gradients du modèle pour produire une carte d'importance ;
- **RISE** masque aléatoirement différentes parties de l'entrée et regarde l'effet sur la prédiction.

Ensuite, on compare les deux approches pour voir laquelle donne les explications les plus convaincantes.

## Comment on évalue

Pour comparer les méthodes, on utilise surtout :

- **Deletion** : on enlève progressivement les zones jugées importantes, et on regarde si la prédiction baisse ;
- **Insertion** : on ajoute progressivement les zones importantes, et on regarde si la prédiction remonte.

Ces deux métriques permettent d'avoir une comparaison un peu plus objective que juste regarder les images.
## Hyperparamètres

On regarde aussi l'effet de plusieurs hyperparamètres de RISE, par exemple :

- le nombre de masques ;
- la taille des masques ;
- la probabilité de garder une zone visible.

Le but est de voir si ces choix changent beaucoup la qualité ou la stabilité des explications.

## Partie séquences

Après la partie image, on reprend la même logique sur des séquences de protéines / acides aminés.

Cette fois, au lieu de masquer des pixels, on masque des tokens dans la séquence pour voir quelles positions influencent le plus la prédiction du modèle.



## Organisation rapide

- `scripts/run_rise.py` : lance RISE sur une image
- `scripts/run_gradcam.py` : lance Grad-CAM sur une image
- `scripts/evaluate.py` : compare les deux méthodes avec insertion / deletion
- `scripts/run_rise_seq.py` : applique RISE sur une séquence d'acides aminés
- `notebooks/` : exploration, visualisation et analyse des résultats
- `result_examples/` : exemples de sorties

## En résumé

Ce projet cherche donc à voir si RISE produit de bonnes explications, comment il se compare à Grad-CAM, ce que montrent les métriques et les visualisations, puis si cette approche peut aussi être utile pour des séquences biologiques.
