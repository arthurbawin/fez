# Vérification du modèle quasi incompressible

## Préparation du partage : 28 octobre 2026

La compilation Release de `incompressible_chns` a été relancée après
formatage des portions de C++ modifiées. Les liens locaux de la documentation
et la syntaxe des scripts Python ont également été vérifiés.

La sélection de contrôles utilisée pour préparer le partage est :

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ctest --test-dir buildRelease --output-on-failure -j 2 \
  -R 'incompressible_chns/(quasi_|chns_|stepien_postprocessing|mms_stepien|adaptive_timestep_cfl|jacobian_matrix|mms_2d_bdf2_stepien_stabilized|mms_chns_2d_space_degenerate_mobility.mpirun=1)|unit_tests/(chns_density_mobility|quasi_incompressible|stepien_article|zero_mean_pressure|restart)'
```

Cette sélection couvre les tests du modèle et du post-traitement, les
comparaisons MPI, les reprises et des régressions Abels/Stepien.
Les **28 tests sélectionnés et leurs quatre dépendances de compilation ont
tous été validés**. Le premier passage donnait 30/32 succès en 666,69 s ;
les deux écarts restants concernaient les références de sortie décrites
ci-dessous. Après correction, les trois comparaisons de sorties concernées
et leurs deux dépendances ont réussi (5/5, en 2,57 s), avec une nouvelle
exécution du test de Jacobien Abels. Les autres contrôles déjà réussis
n'ont pas été rejoués.

Open MPI doit pouvoir ouvrir ses sockets locaux : la première tentative
dans l'environnement isolé échouait à `MPI_Init`, avant tout calcul ; les
contrôles ont ensuite été relancés avec ces communications autorisées.

Le test existant `jacobian_matrix` présentait un écart de référence sur
l'erreur maximale du dernier pas : `9.23e-6` au lieu de `8.18e-6`, soit
`1.05e-6` d'écart pour une comparaison de sorties à `1e-6`. Les cinq
erreurs de Jacobien restent sous le seuil configuré de `1e-5`, sans entrée
signalée hors tolérance. Une référence alternative
`jacobian_matrix.output.gcc14` conserve les résultats obtenus avec GCC 14,
en complément de la référence initiale. Les seuils de vérification du
Jacobien et de comparaison des sorties sont conservés.

La référence de reprise à six processus conserve la ligne vide finale
produite par les journaux MPI : elle fait partie de la comparaison exacte
du fichier. Les blancs de fin de ligne inutiles de la nouvelle référence
`mms_stepien_density_mobility` ont été supprimés.

Le rejeu ciblé s'effectue avec :

```sh
ctest --test-dir buildRelease --output-on-failure -j 2 \
  -R '^(incompressible_chns/(jacobian_matrix.release|mms_stepien_density_mobility.mpirun=1.release)|unit_tests/restart_different_timestep.mpirun=6.release)$'
```

Aucune campagne longue de bulle montante ou de séparation de phases n'est
relancée pour ce partage. Les limites de validation physique et les limites
énergétiques de BDF2 décrites ci-dessous restent applicables.

## Jauge de pression quadratique : correction du 24 octobre 2026

Le contrôle de phase pure en mouvement avec vitesse cubique, pression
quadratique, phase et potentiel quadratiques échouait avant le premier pas.
La jauge de pression moyenne nulle utilisait le premier degré de liberté de
pression comme pivot. Sur les triangles affines, l'intégrale d'une fonction
quadratique associée à un sommet est nulle, ce qui rend ce choix invalide.

La jauge conserve le premier pivot lorsqu'il est suffisamment grand. Sinon,
elle sélectionne le poids d'intégration de plus grande valeur absolue. Les
mises à jour de géométrie conservent ensuite le même degré de liberté pour
préserver la structure des contraintes. Une vérification explicite empêche
la division par un coefficient nul ou presque nul. Aucune stabilisation
n'a été ajoutée.

Le nouveau test unitaire `zero_mean_pressure` vérifie sur des triangles les
pressions linéaire et quadratique, la moyenne nulle après distribution des
contraintes, et la conservation du pivot après déplacement des sommets.
Le contrôle `quasi_pressure_gauge` utilise le petit cas
`pure_moving_quasi_incompressible.prm`, sans interface ni gravité, sur
128 triangles et cinq pas acceptés jusqu'à `t = 0.02`.

Le calcul avec pression quadratique se termine maintenant sur un et deux
processus MPI. L'erreur maximale de phase est respectivement `1.33e-15` et
`1.44e-15`, la densité minimale reste à 1 à l'arrondi près, et le flux
diffusif reste inférieur à `2.51e-19`. Les deux calculs concordent et
l'énergie totale décroît dans ce contrôle.

**6/6 contrôles ciblés ont réussi en 42.41 secondes**, dépendance de
compilation comprise : jauge unitaire, phase pure en mouvement sur un et
deux processus, invariants existants, physique et extension BDF2. Les
résultats et les journaux du nouveau contrôle se trouvent dans
`buildRelease/tests/incompressible_chns/quasi_pressure_gauge/`.

```sh
ctest --test-dir buildRelease --output-on-failure \
  -R '(^unit_tests/zero_mean_pressure.release$|^incompressible_chns/quasi_pressure_gauge$|^incompressible_chns/quasi_incompressible_(invariants|physics|bdf2)$)'
```

Ce correctif rétablit le contrôle de pression quadratique. Il ne modifie
pas le transport du cas avec pression linéaire et phase quadratique :
celui-ci a montré une variation de phase pure de l'ordre de `1.8e-3` dans
le diagnostic lancé par l'utilisateur. La préservation d'une phase pure
par le contrôle compatible ne garantit pas les bornes de phase en présence
d'une interface, qui restent à vérifier sur une bulle.

## Post-traitement standard : vérifications du 23 octobre 2026

Les sous-sections `Postprocessing/chns conservation` et
`Postprocessing/chns energy` utilisent les mêmes paramètres et le même
gestionnaire de tableaux que les autres traitements de FEZ. Elles produisent
par défaut `chns_conservation.txt` et `chns_energy.txt`. Elles fonctionnent
indépendamment des sorties VTU, des diagnostics détaillés et de la qualité
du maillage. Les trois exemples de ce dossier les activent explicitement.

Les intégrales ont été extraites dans `PostProcessingTools` et sont partagées
avec le CSV détaillé existant : un seul parcours de quadrature et une somme
MPI groupée sont nécessaires. La densité affine définit la masse conservée ;
la densité positive reste utilisée pour l'énergie cinétique. Les énergies
de puits double et de gradient sont disponibles séparément.

La compilation Release a réussi. **14/14 contrôles CTest ciblés ont réussi**,
dont deux dépendances de compilation, en 175,06 s après les recompilations
nécessaires. La suite séquentielle propre au post-traitement a pris 7,96 s.
Aucun cas de bulle n'est lancé par ces suites.

Les vérifications ajoutées couvrent :

- Les intégrales analytiques de phase et de densité, et les quatre énergies
  sur des champs affines dans le carré unité : cinétique, puits double,
  gradient et gravité.
- Un état homogène de phase nulle : masse 500,5 pour les densités 1000 et 1,
  énergie de puits double analytique et erreur de phase normalisée finie.
- La distinction entre masse affine et masse prolongée : pour la phase
  homogène -1,02, les intégrales valent respectivement -8,99 et 1.
- Les options d'activation et d'écriture indépendantes, les préfixes,
  la fréquence de réécriture et le fonctionnement sans VTU.
- Les indicateurs géométriques de phases sans option de qualité du maillage.
- L'accord des nouveaux tableaux avec le CSV détaillé en BDF1 et BDF2.
  Les diagnostics non applicables gardent la valeur `nan`.
- La reprise avec référence originale et historique complet, même lorsque
  les fichiers de tableaux n'avaient pas été écrits avant le checkpoint.
- La reprise depuis un ancien checkpoint sans métadonnées : un nouvel
  instant de référence est alors explicitement enregistré dans `reference_time`.
- L'accord entre un et deux processus MPI, et la reprise sur deux processus
  sans doublon temporel ni perte des références.

Les références et les tableaux sont enregistrés dans un fichier binaire
supplémentaire `<checkpoint>.postprocessing`. Il préserve aussi les valeurs
IEEE non finies des diagnostics non applicables, sans changer le format
des archives historiques de solution. Ce fichier doit accompagner les autres
fichiers de checkpoint pour garder la référence originale.

La suite complète BDF1/BDF2, les diagnostics physiques, les jacobiens,
la non-régression du solveur CHNS existant et les reprises du solveur commun
sur un et six processus restent validés.

Pour vérifier seulement ce post-traitement :

```sh
ctest --test-dir buildRelease --output-on-failure -R chns_postprocessing
```

Pour relancer tous les contrôles ciblés :

```sh
ctest --test-dir buildRelease --output-on-failure \
  -R 'chns_postprocessing|quasi_incompressible|^incompressible_chns/mms_2d_bdf2.mpirun=1.release$|^unit_tests/(chns_density_mobility|stepien_article).release$|restart_different_timestep'
```

Les entrées et journaux des nouveaux tests sont conservés dans
`buildRelease/tests/incompressible_chns/chns_postprocessing/` et
`chns_postprocessing_mpi/`. Le [README](README.md) fournit le bloc de paramètres.

## Extension BDF2 : vérifications du 22 octobre 2026

Le modèle offre maintenant les choix BDF1 et BDF2 dans la même structure
d'assemblage. [bdf2.md](bdf2.md) donne les équations fortes et faibles,
les coefficients à pas variables, le démarrage et le jacobien analytique.
La mobilité reste indépendante des densités.

La compilation Release du solveur et du test unitaire a réussi.
Les douze contrôles ciblés sont validés, dont deux dépendances de compilation :
les huit contrôles précédents, la suite BDF2 et les trois contrôles de
compilation/reprise sur un et six processus. La suite BDF2 seule a pris
17,53 s sur cette machine, après compilation. Les cas de bulles sont laissés
au lancement manuel ; ils sont exclus de la suite automatique.

La première exécution complète après extension a validé dix contrôles sur
douze. Les deux écarts restants concernaient les références numériques du
test commun de reprise après changement de pas. Après contrôle des
coefficients, comparaison des erreurs globales sur les six processus et
mise à jour des références, ces deux tests et leur dépendance ont réussi
en 2,23 s. Aucun seuil de comparaison n'a été relâché.

### Contrôles spécifiques

- Jacobien ponctuel en deux et trois dimensions, pour les deux schémas,
  les coefficients à pas constants ou variables, les trois mobilités,
  les densités égales ou de contraste 1000, dans les deux orientations,
  ainsi que les prolongements hors des bornes.
- Jacobien assemblé BDF2 en deux et trois dimensions et en boîte périodique.
- Conservation de la phase et de la masse affine en domaine fermé,
  démarrage compris, pour les rapports 1:1, 20:1 et 1000:1.
- Quatre phases exactement pures avec gravité et pression hydrostatique,
  aux rapports 1000:1 et 1:1000 : vitesse et flux diffusif nuls à la
  précision de résolution.
- Relaxations d'interface avec les deux mobilités dégénérées ; contrôle
  du bilan obtenu en testant les équations et des identités cinétique
  et de gradient à pas constant.
- Reprise identique au calcul continu à pas inchangé ; reprise avec
  changement de pas et vérification des coefficients réellement utilisés.
  Un contrôle unitaire couvre aussi la reprise juste après le démarrage BDF1.
- Comparaison séquentiel/deux processus en boîte périodique et relaxation
  d'interface BDF2 sur deux processus.
- Solutions fabriquées affines en espace, à vitesse, pression, phase et
  potentiel variables en temps, avec trois pas 0.02, 0.01 et 0.005.
  Le flux normal exact de \(\nabla\mu+\alpha\nabla p\) est nul, y compris
  aux densités inégales.
- Rejet des configurations stationnaires, d'un démarrage BDF2 sans BDF1
  ou d'un rapport de démarrage hors de \(0<r<1\), en plus des contraintes
  physiques et spatiales précédentes.

### Ordres mesurés au temps final

Les deux valeurs de chaque case correspondent aux deux raffinements
successifs. L'erreur spatiale n'impose pas de plancher dans cette solution
fabriquée affine. Les valeurs tendent vers deux.

| Densités | Vitesse | Pression | Phase | Potentiel chimique |
|---|---|---|---|---|
| 1:1 | 1.881, 1.945 | 1.971, 1.986 | 2.011, 2.001 | 2.011, 2.001 |
| 1000:1 | 2.009, 2.004 | 1.963, 1.983 | 1.944, 1.975 | 1.944, 1.975 |
| 1:1000 | 2.009, 2.004 | 1.970, 1.982 | 1.785, 1.900 | 1.785, 1.900 |

Ces mesures sont faites après le démarrage. La différence divisée du
premier pas BDF1 reste en général d'ordre un pour le potentiel à cet instant ;
un maximum temporel qui l'inclut ne mesure pas l'ordre deux général du
potentiel après démarrage. La solution doit aussi être suffisamment régulière
pour une mesure d'ordre deux.

### Autres résultats BDF2

| Vérification | Résultat maximal |
|---|---|
| Écart absolu du jacobien assemblé, deux dimensions | \(2.18714\times10^{-8}\) |
| Même écart, trois dimensions | \(3.31636\times10^{-8}\) |
| Même écart, boîte périodique | \(1.64585\times10^{-7}\) |
| Dérive de masse affine sur les cas fermés de la suite BDF2 | \(5.69\times10^{-13}\) |
| Résidu du bilan temporel testé, cas fermés | \(7.10\times10^{-12}\) |
| Résidu de l'identité cinétique à pas constant | \(1.18\times10^{-16}\) |
| Résidu de l'identité de gradient à pas constant | \(5.51\times10^{-17}\) |
| Résidu temporel de l'interface sur deux processus | \(2.09\times10^{-14}\) |

Ces bilans vérifient les équations assemblées. Ils ne constituent pas une
preuve de décroissance inconditionnelle de l'énergie totale en BDF2.
La propriété démontrée de BDF1 est conservée.

### Correction commune de la reprise

Lors d'une reprise avec un nouveau pas, le gestionnaire conservait
correctement les solutions mais remplaçait la longueur du dernier pas
accepté par le nouveau pas. Le premier calcul BDF2 après reprise utilisait
alors des coefficients correspondant à un faux historique de temps.
La longueur acceptée est maintenant conservée ; seul le prochain pas
demandé est modifié.

Dans le test existant, le passage de 0.1 à 0.05 donne donc les coefficients
\(c_0=26.6666667\), \(c_1=-30\), \(c_2=3.3333333\), au lieu des coefficients
à pas constant. Au premier instant après reprise, l'erreur de vitesse passe
de \(3.24056\times10^{-5}\) à \(2.22438\times10^{-6}\), et celle de pression
de \(0.0498894\) à \(0.00113616\). Les historiques d'erreurs globales sont
identiques sur un processus et sur chacun des six processus.
La référence à six processus couvre désormais tous les journaux de rangs
produits par le dispositif de test.

### Relancer

~~~sh
cmake --build buildRelease --target incompressible_chns -j 4
ctest --test-dir buildRelease --output-on-failure \
  -R 'quasi_incompressible|^incompressible_chns/mms_2d_bdf2.mpirun=1.release$|^unit_tests/(chns_density_mobility|stepien_article).release$|restart_different_timestep'
~~~

Le [README](README.md) donne les commandes séparées des suites et les
paramètres pour les bulles. Les entrées générées, les journaux et les CSV de
la nouvelle suite sont conservés dans
buildRelease/tests/incompressible_chns/quasi_incompressible_bdf2/.

## Vérifications du 21 octobre 2026

La formulation retenue est celle documentée dans [equations.md](equations.md) :
vitesse moyenne massique, différence de fractions volumiques, pression
transformée, flux commun aux équations de phase et de divergence. Le schéma
est monolithique, sur maillage fixe, du premier ordre en temps. La sélection
reste `CHNS model = quasi_incompressible`.

La compilation Release du solveur et des trois tests unitaires concernés a
réussi. Après le renommage et l'intégration des formes et de l'assembleur
dans `include/assembly/incompressible_chns_assemblers.h`, la suite ciblée a
été relancée : **8/8 contrôles CTest réussis**, incluant la dépendance de
compilation, en 27,86 s. Les sept tests sont :

- `unit_tests/quasi_incompressible.release` ;
- `unit_tests/chns_density_mobility.release` ;
- `unit_tests/stepien_article.release` ;
- `incompressible_chns/mms_2d_bdf2.mpirun=1.release` (non-régression) ;
- `incompressible_chns/quasi_incompressible_invariants` ;
- `incompressible_chns/quasi_incompressible_physics` ;
- `incompressible_chns/quasi_incompressible_mpi`.

Les exécutions du solveur nécessitent les sockets locales MPI. Dans
l'environnement de développement, elles ont été exécutées hors du bac à
sable après un échec d'initialisation MPI dans celui-ci. Aucune simulation
longue de bulle ou de séparation de phases n'a été lancée.

### Contrôles supplémentaires

- Préfacteur de mobilité réglable, défaut `0.1`, sans facteur de densité
  dans les variables de discrétisation ; contrôle de la conversion constante vers les
  variables Stepien, de la mobilité nulle et des deux orientations de densité.
- Jacobien local avec préfacteur non standard `0.037`, en plus des contrôles
  assemblés 2D/3D, des prolongements et de la différence divisée exacte.
- Quatre états hydrostatiques exactement purs, `phi = +1/-1`, aux rapports
  `1000:1` et `1:1000`, avec gravité : flux nul, vitesse/divergence au niveau
  des erreurs d'arrondi, conservation de phase, de masse affine et d'énergie.
- Relaxation non uniforme sur un maillage de 8x8 carrés subdivisés en
  triangles, sans gravité, avec mobilité dégénérée absolue puis quadratique. La
  dissipation CH est strictement positive ; la masse affine et le bilan
  énergétique sont contrôlés, y compris sur deux processus pour la loi dégénérée absolue.
- État homogène volontairement non physique `phi = -1.02`, densités
  `1000:1`, sans gravité : la densité affine `-8.99` reste visible, tandis que
  la densité cinétique prolongée vaut `1`. Le test vérifie le volume négatif,
  le volume hors bornes, l'écart L1 des densités et les deux champs VTU
  `density`/`density_affine` depuis un dossier de sortie neuf.
- Séparation des pertes numériques cinétique/interfaciale et contrôle de
  leur somme dans le bilan énergétique.
- Rejet de sept configurations incompatibles : BDF2, absence de jauge,
  deux jauges, bord fluide ouvert, phase imposée hors MMS, gravité en boîte
  périodique, préfacteur négatif.

### Résultats actuels

| Vérification | Résultat |
|---|---|
| Écart absolu maximal du Jacobien assemblé 2D | $1.006\times10^{-8}$ |
| Même écart en 3D | $8.035\times10^{-9}$ |
| Même écart périodique, force volumique nulle | $5.515\times10^{-8}$ |
| Ordres temporels sur la phase manufacturée | 1.023, 1.012 |
| Ordres spatiaux sur la phase manufacturée | 1.788, 1.946 |
| Flux diffusif des quatre états purs | 0 |
| Norme L2 maximale de divergence, états purs | $7.54\times10^{-17}$ |
| Résidu énergétique maximal, interface à mobilité dégénérée absolue | $6.49\times10^{-15}$ |
| Même résidu, interface quadratique | $5.66\times10^{-15}$ |
| Même résidu, interface à mobilité dégénérée absolue sur deux processus | $2.63\times10^{-15}$ |
| Dérive maximale de masse affine, interfaces séquentielles | $4.09\times10^{-14}$ |
| Perte CH intégrée, interface à mobilité dégénérée absolue | $3.581\times10^{-5}$ |
| Perte CH intégrée, interface quadratique | $3.307\times10^{-5}$ |

Les deux dernières valeurs correspondent aux lois de même amplitude au
centre de l'interface ; elles ne constituent pas une comparaison de
précision physique à conductance interfaciale identique. Le cas périodique
manufacturé contient des sources : son CSV sert à comparer les partitions
MPI, pas à prouver la décroissance d'une énergie sans travail extérieur.

Pour relancer l'ensemble ciblé depuis la racine du dépôt :

```sh
cmake --build buildRelease --target incompressible_chns -j 4
ctest --test-dir buildRelease --output-on-failure \
  -R 'quasi_incompressible|^incompressible_chns/mms_2d_bdf2.mpirun=1.release$|^unit_tests/(chns_density_mobility|stepien_article).release$'
```

Les fichiers d'entrée générés, journaux et CSV restent dans
`buildRelease/tests/incompressible_chns/quasi_incompressible_invariants/`,
`quasi_incompressible_physics/` et `quasi_incompressible_mpi/`. Le
[README](README.md) donne les commandes des suites `core`, `physics` et MPI
ainsi que celles des exemples physiques. Ces tests ne démontrent pas un
principe du maximum ni une validation quantitative des bulles Hysing.

## Historique : vérifications du 20 octobre 2026

Les résultats ci-dessous sont conservés comme historique de la première
implémentation ; les vérifications actuelles sont celles ci-dessus. Le
fixture périodique emploie maintenant une force volumique nulle.
Les noms de tests présentés dans ce document suivent la nomenclature
actuelle ; les journaux des anciennes campagnes gardent leurs identifiants
originaux.

Résultats historiques de la branche de développement CHNS.
Compilation Release avec deal.II/PETSc, solveur linéaire MUMPS.

Les équations de référence et leurs variations sont dans
[equations.md](equations.md). Les fichiers physiques et les commandes sont
dans [README.md](README.md).

## Vérifications effectuées

- Tous les blocs du Jacobien des formes faibles : différences finies à cinq
  points en 2D et 3D, trois mobilités, densités égales, rapports 1000:1 et
  1:1000, phases à l'intérieur et à l'extérieur de l'intervalle physique.
- Différence divisée du potentiel : identité exacte et limite lorsque les
  deux valeurs de phase coïncident.
- Identité énergétique des formes faibles : annulation de la convection
  antisymétrique et des couplages pression/capillarité ; incréments cinétique
  et interfacial, dissipation physique et dissipation numérique.
- Matrices réellement assemblées : comparaison avec les différences finies
  du résidu complet sur des solutions manufacturées.
- Conservation et bilan énergétique : petit test fermé de relaxation de
  vitesse avec phase initiale uniforme, gravité et quatre pas, sans sources.
- Raffinements manufacturés : trois niveaux spatiaux et trois pas temporels.
- Boîte périodique : comparaison séquentiel/deux processus, y compris les
  diagnostics des termes diffusifs et visqueux.
- Conservation et bilan énergétique du test fermé sur deux processus.
- Rejet explicite de BDF2 pour ce modèle.

## Résultats

| Vérification | Résultat |
|---|---|
| Écart absolu maximal du Jacobien assemblé, 2D | $1.006\times10^{-8}$ |
| Écart absolu maximal du Jacobien assemblé, 3D | $8.035\times10^{-9}$ |
| Écart absolu maximal du Jacobien assemblé, boîte périodique | $4.130\times10^{-8}$ |
| Résidu maximal du bilan énergétique fermé, densités égales | $1.976\times10^{-16}$ |
| Même résidu, rapport 20:1 | $5.759\times10^{-16}$ |
| Même résidu, rapport 1000:1 | $8.736\times10^{-15}$ |
| Dérive maximale de la masse affine, ces tests | $3.411\times10^{-13}$ |
| Résidu énergétique fermé, deux processus MPI | $2.221\times10^{-16}$ |
| Ordres temporels de l'erreur $L^2$ sur $\phi$ | 1.023, 1.012 |
| Ordres spatiaux de l'erreur $L^2$ sur $\phi$ | 1.788, 1.946 |

Les ordres sont calculés par $\log_2(e_{\Delta}/e_{\Delta/2})$. La vérification
temporelle utilise une solution spatialement uniforme pour isoler l'erreur
temporelle. La vérification spatiale utilise une solution stationnaire en
temps. Les tests de convergence ci-dessus sont en 2D ; la vérification 3D
porte sur les formes et le Jacobien assemblé, sans étude de convergence 3D.

Les seuils automatiques vérifient aussi que l'énergie décroît d'une quantité
mesurable, que les dissipations sont positives et que les intégrales de phase
et de masse restent constantes. Le bilan inclut l'énergie gravitationnelle.

Commande complète exécutée :

```sh
ctest --test-dir buildRelease --output-on-failure \
  -R 'quasi_incompressible|^incompressible_chns/mms_2d_bdf2.mpirun=1.release$|^unit_tests/(chns_density_mobility|stepien_article).release$'
```

Résultat : **7/7 tests réussis**, incluant une étape de compilation dépendante,
les trois groupes de vérification du nouveau modèle et les tests existants de
non-régression. Les journaux et CSV sont conservés sous
`buildRelease/tests/incompressible_chns/quasi_incompressible_invariants/` et
`buildRelease/tests/incompressible_chns/quasi_incompressible_mpi/`.

Les deux cas de bulle montante et le cas de séparation de phases de l'article
sont fournis comme fichiers de paramètres et laissés au lancement par
l'utilisateur. Les résultats ci-dessus concernent les tests de vérification
du code, pas une validation de ces trois simulations physiques.

## Énergie par phase

Les suites `incompressible_chns/chns_phase_energy` et
`incompressible_chns/chns_phase_energy_mpi` vérifient la sortie indépendante
`chns_phase_energy.txt` sur de petits problèmes, sans lancer de bulle :

- Intégrales analytiques cinétique, de puits double, de gradient et de gravité
  dans chaque région du champ `phi = 2*x-1`, en 2D et 3D, pour les modèles
  quasi incompressible et Stepien.
- Somme des deux phases égale à chaque composante globale ; sommes des
  composantes interfaciales et totales dans chaque phase.
- Phases uniformes de signes opposés, phase absente et convention `phi = 0`.
- Activation indépendante, écriture désactivée, préfixe, verbosité, fréquence
  et écriture finale entre deux multiples de la fréquence.
- Interface résolue en BDF2, accord entre un et deux processus MPI et reprise
  avec historique en mémoire, même sans fichier texte avant le checkpoint.
- Activation du suivi à une reprise avec des tableaux globaux existants,
  et reprise depuis un checkpoint sans fichier de post-traitement.

```sh
ctest --test-dir buildRelease --output-on-failure \
  -R '^incompressible_chns/chns_phase_energy(_mpi)?$'
```

Le script accepte aussi `--legacy-solver /chemin/ancien/incompressible_chns`
pour créer un véritable checkpoint V1 avec l'ancien exécutable, puis vérifier
sa lecture et l'activation du suivi par phase avec le nouvel exécutable.
Les journaux de chaque lancement sont conservés dans un sous-dossier `run_*`
du répertoire de validation.
