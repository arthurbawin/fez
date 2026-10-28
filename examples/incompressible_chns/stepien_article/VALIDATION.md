# Vérifications de la branche chns_stepien_article_bdf

Travail démarré depuis `0f59b57`, en conservant les modifications déjà présentes
dans l'arbre de travail. Une copie de cet état est conservée localement dans
`tmp/incompressible_chns/stepien_article_baseline_2026-10-19/`.

## Compilation et tests ciblés

Compilation Release du solveur `incompressible_chns` et du nouveau test
unitaire réussie, avec la configuration deal.II existante.

Les sept cibles suivantes ont passé les contrôles ciblés :

- dépendance de compilation `incompressible_chns.executable` ;
- `mms_2d_bdf2_stepien_stabilized`, 1 processus MPI ;
- `mms_chns_2d_space_degenerate_mobility`, 1 processus MPI (régression Abels) ;
- `mms_stepien_density_mobility`, 1 processus MPI ;
- `mms_stepien_article`, 1 processus MPI ;
- test unitaire existant `chns_density_mobility` ;
- nouveau test unitaire `stepien_article`, en dimensions 2 et 3.

Le test manufacturé de la nouvelle variante utilise simultanément la mobilité
de l'article, le transport conservatif, l'inertie antisymétrique, BDF2 et des
pressions de référence différentes (`3` et `1`). Les écarts maximaux entre le
Jacobien analytique et sa vérification par différences finies sont
`8.84546e-8`, `2.84377e-8` et `1.30503e-8` sur les trois pas du démarrage BDF.

Un essai indépendant sur trois maillages, sans la comparaison coûteuse des
Jacobiennes, donne les erreurs L2 suivantes :

| Cellules triangulaires | Degrés de liberté | u | p | phi | mu |
|---:|---:|---:|---:|---:|---:|
| 8 | 77 | 2.7031e-3 | 2.0503e-1 | 2.0827e-2 | 2.2081e-1 |
| 32 | 237 | 9.4222e-4 | 6.6770e-2 | 6.0690e-3 | 7.1436e-2 |
| 128 | 821 | 2.4128e-4 | 1.8499e-2 | 1.5790e-3 | 1.9047e-2 |

Les erreurs diminuent pour les quatre champs. Ce contrôle ne constitue pas une
preuve de stabilité énergétique.

## Essais courts de bulle montante

Paramètres communs : `h=1/16`, `epsilon=0.04`, `dt=0.008`, BDF2 avec démarrage
BDF1, 1024 triangles et 5973 degrés de liberté, rapport de densité 1000.
La mobilité constante initiale est `6.1e-6`. Les résultats complets sont dans
`tmp/incompressible_chns/stepien_article_smoke_v2/`.

Les quatre variantes atteignent `t=0.04`. La variation finale de l'intégrale de
phase et la variation d'énergie mesurée sont :

| Variante | Variation intégrale de phase | Variation énergie |
|---|---:|---:|
| baseline | -1.46525e-7 | -0.0925419 |
| article_mobility | -3.11570e-8 | -0.0913935 |
| conservative | -5.28466e-14 | -0.0917420 |
| conservative_skew | -5.68434e-14 | -0.0917418 |

L'énergie décroît à chaque pas de ces essais courts. La masse calculée avec
la densité limitée ne se confond pas avec celle calculée avec la densité
affine ; leurs diagnostics sont séparés dans les CSV.

Les résultats jusqu'à t=3 sont enregistrés dans
`tmp/incompressible_chns/stepien_article_h16_t3/`. Ce maillage grossier sert à
explorer le comportement des variantes ; il ne valide pas le benchmark 2.

## Comparaison demandée jusqu'à t=3

Les quatre essais ont été exécutés avec les mêmes paramètres communs que les
simulations courtes. Le code de retour du lanceur est 1 parce que deux variantes
échouent ; les deux variantes conservatives ont effectivement atteint t=3.

| Variante | Dernier temps atteint | Résultat | Dérive maximale de l'intégrale de phase | Pas avec hausse d'énergie mesurée |
|---|---:|---|---:|---:|
| baseline | 2.192 | Divergence non linéaire | 3.98306e-02 | 16 |
| article_mobility | 1.160 | Non-convergence non linéaire | 2.08211e-03 | 0 |
| conservative | 3.000 | Terminé | 2.18048e-13 | 0 |
| conservative_skew | 3.000 | Terminé | 2.20934e-13 | 0 |

Les valeurs aux derniers temps de deux simulations interrompues ne doivent pas
être comparées directement aux valeurs finales à t=3 des autres variantes.

Pour les deux simulations qui atteignent t=3 :

| Variante | Centre vertical (phi<0) | Vitesse de montée (phi<0) | Variation relative de la masse limitée | Min/max de phi sur tout le calcul |
|---|---:|---:|---:|---|
| conservative | 1.028301 | 0.172509 | -3.656 % | -1.254 / 1.651 |
| conservative_skew | 1.030395 | 0.173311 | -3.632 % | -1.308 / 1.650 |

Sur cet essai, le passage au transport conservatif apporte le changement
principal : les simulations atteignent t=3 et préservent l'intégrale de phase.
L'effet additionnel de l'inertie antisymétrique sur les indicateurs finaux est
plus modéré. Le changement de mobilité seul n'améliore pas la robustesse de
cet essai.

Ces résultats ne valident pas le benchmark 2 : le maillage est grossier, phi
sort sensiblement de [-1,1], et la masse calculée avec la densité limitée dérive
d'environ 3.6 % même lorsque l'intégrale de phase et la masse affine sont
conservées. L'absence de hausse d'énergie mesurée sur ces deux trajectoires ne
constitue pas une garantie inconditionnelle pour BDF2.

Les paramètres complets, journaux et séries temporelles sont disponibles dans
les sous-dossiers des quatre variantes. `summary.csv`, `summary.json` et
`comparison.png` rassemblent les mesures, y compris celles des essais
interrompus. Le lanceur et le modèle de paramètres restent disponibles pour
une étude de convergence avec une résolution plus fine.
