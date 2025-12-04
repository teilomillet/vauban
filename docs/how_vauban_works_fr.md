# Comment fonctionne Vauban : Une perspective bayésienne sur GEPA

Vauban est un système automatisé de "Red Teaming" conçu pour tester la robustesse des modèles d'IA. Son moteur central, **GEPA (Generalizable Evolutionary Prompt Attack)**, peut être compris à travers le prisme de l'**Optimisation Bayésienne**.

## La Philosophie : À la recherche de la "Vérité" de la Vulnérabilité

Imaginez l'"espace" de tous les prompts possibles comme un vaste paysage à haute dimension. Quelque part dans ce paysage se trouvent des "pics" de vulnérabilité — des prompts qui contournent avec succès les défenses d'un modèle.

Dans un cadre bayésien, nous commençons avec une **Croyance A Priori (Prior)** : une supposition sur ce à quoi ressemble une attaque réussie.
- **Le Prior** : Notre population initiale d'attaques, générée par diverses "Personas" (ex: un étudiant curieux, un chercheur). Nous ne savons pas encore si elles fonctionnent, mais elles représentent notre meilleure distribution initiale.

Nous rassemblons ensuite des **Preuves (Evidence)** en testant ces prompts contre le modèle cible.
- **La Vraisemblance (Likelihood)** : Le retour que nous obtenons (Refus vs Brèche, Score de Discrétion, Score de Récompense). Cela nous indique la probabilité qu'un type de prompt spécifique mène à une brèche réussie.

Nous mettons à jour nos croyances pour former une **Croyance A Posteriori (Posterior)** : une compréhension affinée de ce qui fonctionne.
- **Le Posterior** : Les "Survivants" de notre population. Nous gardons les prompts qui ont le mieux fonctionné (scores élevés, grande discrétion) et écartons les autres.

## GEPA : Le Moteur d'Optimisation

GEPA est le mécanisme qui affine itérativement ce Posterior pour converger vers l'attaque optimale. Il utilise des principes évolutionnaires pour "explorer" (élargir la recherche) et "exploiter" (se concentrer sur ce qui fonctionne).

### 1. La Boucle (Mise à jour Bayésienne)
1.  **Générer** : Créer un lot d'attaques (échantillons de notre distribution actuelle).
2.  **Évaluer** : Les tester contre la cible.
3.  **Mettre à jour** : Élagueur les faibles, garder les forts (Sélection de Pareto).
4.  **Muter** : Créer la prochaine génération basée sur les survivants.

### 2. Mécaniques d'Optimisation
-   **Sélection de Pareto (Optimisation Multi-Objectifs)** : Au lieu d'optimiser pour une seule chose (ex: "est-ce que ça a cassé ?"), Vauban optimise pour **trois** objectifs simultanément :
    -   **Score de Succès** : L'objectif a-t-il été atteint ?
    -   **Discrétion (Stealth)** : Cela semblait-il innocent ?
    -   **Récompense (Reward)** : Un score proxy pour la "qualité" ou le "danger".
    
    En conservant un ensemble diversifié de solutions "Pareto-Optimales" (solutions où l'on ne peut améliorer une métrique sans en nuire à une autre), Vauban maintient une "distribution de croyance" riche plutôt que de s'effondrer trop tôt sur une seule astuce.

-   **Réflexion (L'étape de "Raisonnement")** :
    Avant de muter, le système "réfléchit" sur *pourquoi* un prompt a échoué. "Le modèle a refusé parce que j'ai utilisé le mot 'bombe'. Je devrais essayer 'dispositif énergétique' à la place."
    
    *Note d'Optimisation* : Nous avons récemment activé l'`Agent de Réflexion` — le "cerveau" conçu pour analyser l'échec. Auparavant, le système reposait sur des scores bruts ("Score=2/10") sans compréhension sémantique. Désormais, il injecte un raisonnement qualitatif dans la boucle de rétroaction, transformant le système d'un "Devineur basé sur des Scores" en un "Optimiseur Raisonnant".

-   **Croisement (Fusion Consciente du Système)** :
    Combine deux "parents" réussis pour créer un "enfant". Si le Parent A a une super Persona et le Parent B a une Stratégie intelligente, l'enfant pourrait hériter des deux. Cela saute vers de nouvelles zones de l'espace de probabilité qui pourraient avoir une haute vraisemblance.

## Optimisations Récentes

**1. Activation du Cerveau Silencieux (Réflexion)**
L'`Agent de Réflexion` est maintenant activement appelé pour analyser les échecs. Il identifie des déclencheurs spécifiques (ton, mots-clés) et transmet ce retour sémantique à l'Agent de Mutation.

**2. Activation de l'Évolution de Persona**
Le schéma du prompt de mutation a été mis à jour pour demander explicitement les `persona_updates`. Cela permet à l'attaquant d'évoluer sa personnalité (ex: devenir plus autoritaire ou plus empathique) en fonction de ce qui fonctionne, utilisant pleinement le potentiel évolutionnaire de GEPA.
