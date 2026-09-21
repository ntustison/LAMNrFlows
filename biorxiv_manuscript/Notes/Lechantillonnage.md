
## D'abord, pourquoi les échantillons ne semblent toujours pas comme les IRM vrais ?

Les échantillons générés par les flux normalisants (comme l'architecture Glow
utilisée dans LAMNr) présentent un aspect visuel systématiquement plus lisse,
diffus et moins « photoréaliste » que les véritables IRM. Ce phénomène
s'explique par quatre contraintes physiques et mathématiques fondamentales.

### 1. La Limite de Résolution (Sous-échantillonnage)

Une acquisition IRM T1-w clinique standard possède une résolution isotrope
d'environ $1 \text{ mm}^3$ (soit un volume typique de $256 \times 256 \times
256$ voxels, représentant près de 16,7 millions de points de données). Votre
expérience « whole head » opère sur un volume compressé de $48 \times 64 \times
56$ voxels (environ 172 000 points). Cette réduction d'échelle d'un facteur 100
détruit physiquement les textures de haute fréquence. Il est mathématiquement
impossible pour le réseau de générer le grain naturel des tissus ou la netteté
absolue des interfaces (comme les méninges ou les micro-vaisseaux) avec si peu
de voxels.

### 2. Le Prix de la Log-Vraisemblance Exacte

Il existe une différence philosophique et mathématique majeure entre les flux
normalisants et d'autres modèles génératifs :

* **Réseaux Antagonistes Génératifs (GANs) / Diffusion :** Ces modèles
  optimisent la qualité perceptive. Ils peuvent inventer des textures fausses
  mais visuellement parfaites (hallucinations) pour tromper un discriminateur ou
  débruiter une image. Ils sacrifient la couverture de la distribution (mode
  collapse) au profit de l'esthétique.
* **Flux Normalisants (LAMNr) :** Votre modèle est forcé d'optimiser la
  log-vraisemblance exacte de *toutes* les données. Le modèle ne peut ignorer
  aucune variation anatomique de la cohorte d'entraînement. Pour minimiser cette
  perte globale, le réseau adopte une stratégie conservatrice : il produit des
  représentations lissées qui font la moyenne des incertitudes spatiales,
  évitant de générer des textures aiguës qui risqueraient de pénaliser
  lourdement le score BPD (Bits Per Dimension) si elles étaient légèrement mal
  placées.

### 3. La Rigidité de la Bijection (Couches Affines)

Pour que votre modèle LAMNr puisse inverser parfaitement un volume d'un espace à
l'autre (l'exigence clé pour l'anatomie computationnelle), l'architecture
utilise des couches de couplage affine. Contrairement aux convolutions standards
des réseaux profonds classiques qui peuvent éliminer l'information inutile, une
fonction bijective ne peut rien jeter. Le réseau doit encoder chaque artefact,
chaque bruit thermique du scanner original et chaque variation de contraste.
Cette conservation stricte limite la capacité du réseau à synthétiser des bords
d'un contraste parfait.

### 4. La Nature de l'Échantillonnage (Prior Gaussien)

Lorsque vous générez un échantillon, vous tirez un vecteur aléatoire depuis une
distribution de base gaussienne $\mathcal{N}(0, I)$.

* Un vrai cerveau humain possède une cohérence structurelle absolue (si un
  ventricule a une certaine forme, le sillon adjacent est contraint par la
  pression physique).
* Le tirage aléatoire dans l'espace latent $\mathcal{Z}$ combine des
  caractéristiques statistiques de la cohorte. Le résultat est souvent un
  cerveau « chimère » ou une moyenne statistique interpolée qui est
  anatomiquement valide sur le plan macroscopique, mais qui manque de la netteté
  et de la cohésion microscopique d'un patient individuel réel.

Si l'objectif du manuscrit est la synthèse d'images pures (Deepfakes médicaux),
les flux normalisants sont le mauvais outil. Cependant, si l'objectif est
l'analyse anatomique rigoureuse, l'imputation de données manquantes et
l'alignement topologique garanti, ce lissage visuel est le compromis
mathématique nécessaire et acceptable.


## Ça va mais pourquoi pouvait-on générer les échantillons plus réalistes en 2D (3 vues--T1, T2, FA) ?

C'est une excellente observation. Il est tout à fait vrai que vos expériences 2D
intégrant trois modalités (T1, T2, FA) produisaient des échantillons
visuellement plus nets et réalistes.

Ce n'est pas une régression de votre code 3D ou une erreur d'optimisation. La
différence de réalisme s'explique par trois phénomènes physiques et
architecturaux majeurs qui séparent la 2D de la 3D.

### 1. La synergie multimodale (L'effet "Corset")

Dans votre expérience 2D, vous forciez le réseau à apprendre une représentation
conjointe de trois vues très différentes :

* **T1 :** Fournit la structure anatomique globale.
* **T2 :** Isole strictement les fluides (LCR très brillant) avec des bords
  francs.
* **FA (Anisotropie Fractionnelle) :** C'est une carte dérivée de tenseurs de
  diffusion qui représente les faisceaux de matière blanche. Par nature, c'est
  un signal de **très haute fréquence** et hautement directionnel.

Pour qu'une seule représentation latente (ou des latents alignés) puisse générer
simultanément ces trois contrastes, les modalités agissent comme des contraintes
mutuelles. L'obligation de devoir reconstruire les fibres directionnelles du FA
forçait le modèle à conserver ses filtres de haute fréquence. Votre expérience
3D actuelle ("whole head") n'a qu'une seule vue (T1). Sans la pression des
autres modalités pour "ancrer" les textures, le modèle choisit la solution de
facilité : une moyenne lisse.

### 2. La contrainte de continuité topologique (L'axe Z)

Générer une image réaliste en 2D est un problème mathématique beaucoup plus
simple que d'en générer une en 3D.

* **Le "luxe" de la 2D :** Sur une coupe 2D, le modèle peut générer un pixel
  très net pour simuler un vaisseau sanguin. Il n'a aucune obligation de
  s'assurer que ce vaisseau continue parfaitement sur la coupe suivante. Il peut
  optimiser l'esthétique locale pure.
* **La "prison" de la 3D :** Un volume 3D impose une cohérence géométrique
  stricte. Le vaisseau sanguin doit être continu à travers l'axe Z (la
  profondeur). S'il y a un décalage d'un seul voxel d'une coupe à l'autre, le
  Jacobien se fracture et la perte (NLL) explose. Face à cette difficulté
  géométrique, le modèle Glow 3D "triche" : il crée un tube flou (une
  distribution de probabilité plus large) pour s'assurer que les voxels se
  connectent sans erreur, ce qui détruit l'illusion de photoréalisme.

### 3. La malédiction de la dimensionnalité (Budget de paramètres)

Il y a une différence massive de densité d'information par rapport à la capacité
de votre réseau de neurones.

* Une coupe 2D haute résolution (ex: $256 \times 256$) contient **65 536
  pixels**.
* Votre volume 3D compressé ($48 \times 64 \times 56$) contient **172 032
  voxels**.

Bien que le volume contienne près de trois fois plus de points, la complexité
des relations spatiales (haut/bas, gauche/droite, avant/arrière) croît au cube.
Pour un budget de paramètres constant (ex: `HIDDEN="64 128 192"`), le modèle 2D
peut allouer une grande partie de ses poids à mémoriser les micro-textures (le
grain de l'image). Le modèle 3D, en revanche, épuise l'intégralité de sa
capacité d'apprentissage simplement pour réussir à positionner correctement les
ventricules et le cortex dans l'espace.

En résumé, vos modèles 2D multimodaux excellaient dans la **synthèse de
texture** grâce à la richesse des contrastes (FA, T2). Votre modèle 3D actuel
excelle dans la **cohérence spatiale globale**, ce qui nécessite un lissage
mathématique inévitable.

