# CHNS quasi incompressible : lancement et vérifications

Le modèle se sélectionne avec `set CHNS model = quasi_incompressible` dans
`Cahn Hilliard`, via l'exécutable `incompressible_chns`. Les équations fortes,
faibles, le schéma temporel et les variations du Jacobien sont détaillés dans
[equations.md](equations.md).
L'extension temporelle, ses équations faibles et son jacobien sont décrits
dans [bdf2.md](bdf2.md).

## Parcours de relecture

Pour récupérer cette branche depuis un clone du dépôt :

```sh
git fetch origin
git switch --track origin/chns_brunk_ten_eikelder_energy_stable
```

| Sujet | Point d'entrée |
|---|---|
| Modèle, changements de pression et de mobilité, formes faibles et Jacobien BDF1 | [equations.md](equations.md) |
| Démarrage, coefficients BDF2 et limites du bilan énergétique | [bdf2.md](bdf2.md) |
| Résidus et Jacobien du schéma énergétique | [`Assembly::IncompressibleCHNS::EnergyStable`](../../../include/assembly/incompressible_chns_assemblers.h) |
| Lois de densité, viscosité et mobilité | [`cahn_hilliard.h`](../../../include/cahn_hilliard.h) |
| Sélection du modèle, contraintes et export des diagnostics | [`incompressible_chns_solver.cpp`](../../../src/incompressible_chns_solver.cpp) |
| Intégrales de masse, phase, énergie et aire de bulle | [`post_processing_tools.cpp`](../../../src/post_processing_tools.cpp) |
| Tableaux de suivi et historique de reprise | [`post_processing_handler.cpp`](../../../src/post_processing_handler.cpp) |
| Vérifications reproductibles et limites de validation | [VALIDATION.md](VALIDATION.md) et [scripts de test](../../../tests/incompressible_chns/) |

La sélection `quasi_incompressible` est distincte de `stepien`, qui conserve
son chemin d'assemblage et ses variantes expérimentales décrites dans
[`stepien_article`](../stepien_article/README.md).

## Sélection et lancement

La nomenclature de configuration est désormais descriptive :

```text
subsection Cahn Hilliard
  set CHNS model = quasi_incompressible
  set mobility model = degenerate_absolute
  set degenerate absolute mobility prefactor = 0.1
  set write quasi incompressible diagnostics = true
  set quasi incompressible periodic box = false
end
```

Les paramètres et scripts des exemples ont été migrés. Les anciens fichiers
de campagnes conservés ailleurs doivent employer ces noms pour être relancés.
Les résultats archivés ne sont pas déplacés. Le fichier de diagnostics
s'appelle maintenant `quasi_incompressible_diagnostics.csv`.

Cette sélection représente la reformulation quasi incompressible du système
analytique fourni, avec $p=P-\phi\mu$ et $M_{\rm doc}=k^2M$ ; le document
précise ces conventions. Il n'y a pas de second solveur ou de modèle
compressible ajouté. Les trois simulations physiques ci-dessous restent
des calculs à lancer et à comparer aux références ; les résultats de tests
effectivement exécutés sont consignés dans [VALIDATION.md](VALIDATION.md).

Depuis la racine du dépôt, avec les dépendances décrites dans le
[README principal](../../../README.md) :

```sh
cmake -S . -B buildRelease -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTING=ON
cmake --build buildRelease --target incompressible_chns -j 4
./buildRelease/incompressible_chns examples/incompressible_chns/quasi_incompressible/rising_bubble_case1.prm
./buildRelease/incompressible_chns examples/incompressible_chns/quasi_incompressible/rising_bubble_case2.prm
./buildRelease/incompressible_chns examples/incompressible_chns/quasi_incompressible/phase_separation.prm
```

Chaque commande lance un cas distinct. On peut aussi utiliser
`mpirun -np 4 ./buildRelease/incompressible_chns chemin/du/cas.prm`.
Ces trois simulations sont préparées pour être lancées par l'utilisateur.

## Paramètres des exemples

Le choix temporel se fait dans la section existante :

```text
subsection Time integration
  set scheme = BDF2
  set bdf start method = BDF1
  set bdf start step ratio = 0.5
  set dt = 0.004
  set t_initial = 0
  set t_end = 0.04
end
```

`scheme = BDF1` conserve le schéma énergétique initial. BDF2 utilise le
gestionnaire de coefficients et les deux historiques de FEZ, puis un
potentiel cubique implicite après le démarrage BDF1. Le rapport proposé
donne deux premiers pas de `dt/2`, puis le pas nominal. BDF2 exige ce
démarrage et un rapport strictement compris entre zéro et un.
Les exemples conservent BDF1 par défaut pour les comparaisons ; changer
`scheme` suffit lorsque les deux lignes de démarrage sont présentes.

Pour une première montée de bulle courte, garder `dt = 0.004` et limiter
`t_end` à `0.04`, puis à `1.2` si le calcul est satisfaisant. Comparer ensuite
avec `dt = 0.002` au même temps final, sur le même maillage et avec la même
épaisseur d'interface. Le passage à BDF2 ne garantit pas qu'un pas plus
grand soit rentable : comparer aussi les itérations et la précision.

Les bulles utilisent la discrétisation de la section 4.3 de l'article : domaine
$[0,1]\times[0,2]$, bulle centrée en $(0.5,0.5)$ de rayon $0.25$, gravité
$0.98$, glissement sur les côtés et adhérence en haut et en bas. Les valeurs
de référence sont $h=1/32$, $\epsilon=0.64h=0.02$ et $\Delta t=0.128h=0.004$.

| Cas | $\rho_0,\rho_1$ | $\eta_0,\eta_1$ | $\sigma$ |
|---|---|---|---|
| Bulle 1 | 1000, 100 | 10, 1 | 24.5 |
| Bulle 2 | 1000, 1 | 10, 0.1 | 1.96 |

Le cas 2 prend la viscosité dynamique lourde $10$ de Hysing, avec
`kinematic viscosity = 0.01` à densité 1000. Le tableau 3 de l'article
imprime $1$ ; ce fichier vise le benchmark Hysing et les paramètres du code
des auteurs. Le cas 1 a les viscosités cinématiques $0.01,0.01$ ; le cas 2,
$0.01,0.1$.

La mobilité `degenerate_absolute` vaut
$m_0\epsilon^2|1-\phi^2|$, où `degenerate absolute mobility prefactor = 0.1`
règle $m_0$ indépendamment des densités. `mobility` ne règle pas cette loi.
La compensation en densité et le limiteur de phase de la mobilité sont
désactivés explicitement. Pour une loi quadratique, remplacer ce bloc par

```text
set mobility model = degenerate
set enable mobility tracer limiter = false
set enable density mobility compensation = false
subsection degenerate mobility
  set Function expression = 0.00004*(1-x*x)^2
end
```

Ici `x` représente $\phi$ et $0.00004=0.1(0.02)^2$ donne la même mobilité
au centre de l'interface que la loi précédente à $\epsilon=0.02$.
Cette égalité au centre n'égalise pas l'intégrale de mobilité dans l'interface.
L'expression est l'amplitude complète, sans multiplication automatique par
`mobility`, le préfacteur de mobilité absolue ou $\epsilon^2$.

Pour distinguer les sources d'erreur, raffiner dans cet ordre, avec une
copie du `.prm` et un répertoire de sortie distinct pour chaque calcul :

1. Garder maillage et $\epsilon=0.02$ fixes et diminuer `dt`. Pour le cas 1,
   comparer $0.001$ puis $0.0005$ ; pour le cas 2, $0.005$, $0.001$, $0.0005$.
2. Garder $\epsilon$ et un petit `dt` fixes, puis passer par exemple de
   32×64 à 64×128 cellules.
3. Réduire ensuite $\epsilon$ en gardant suffisamment de cellules dans
   l'interface ; mettre à jour aussi `eps` de la condition initiale.

Une suite reproduisant les rapports de l'article peut aussi coupler
$h=1/64$, $\epsilon=0.01$ et $\Delta t=0.002$. Elle change simultanément le
modèle diffus et la discrétisation. La mobilité prédéfinie change alors
d'amplitude avec $\epsilon^2$ ; ce n'est pas une compensation en densité.

La séparation de phases reprend la section 4.1 : boîte périodique $[0,1]^2$,
$\phi^0=0.2\sin(4\pi x)\sin(4\pi y)$, $v^0=0$, $\beta=\gamma=10^{-3/2}$,
$\eta_0=\eta_1=0.01$ et $m=0.01(1-\phi^2)^2$. Le fichier fixe le rapport
$1:1000$ ; pour un autre rapport, ajuster les **viscosités cinématiques** pour
conserver $\eta_i=\rho_i\nu_i=0.01$.

La boîte utilise Q2/Q1 pour la vitesse/pression et Q1 pour phase/potentiel,
conformément à la remarque 5. Les voisins périodiques sont enregistrés avant
la partition MPI. La pression est fixée au point intérieur $(0.5,0.5)$ ; garder
ce point comme sommet intérieur lors d'un changement de maillage. Les sections
`Fluid boundary conditions` et `CahnHilliard boundary conditions` ont toutes
deux `number = 0` : l'option périodique impose les contraintes sur tous les champs.
La boîte périodique de cette première version requiert `use quads = true` et
un rectangle colorisé avec les identifiants opposés 0/1, 2/3, et 4/5 en 3D.

## Diagnostics

Le suivi de conservation et d'énergie utilise la section standard de FEZ :

```text
subsection Postprocessing
  subsection chns conservation
    set enable = true
    set write results = true
    set output prefix = chns_conservation
    set output frequency = 1
    set precision = 16
    set verbosity = quiet
  end
  subsection chns energy
    set enable = true
    set write results = true
    set output prefix = chns_energy
    set output frequency = 1
    set precision = 16
    set verbosity = quiet
  end
  subsection chns phase energy
    set enable = true
    set write results = true
    set output prefix = chns_phase_energy
    set output frequency = 1
    set precision = 16
    set verbosity = quiet
  end
end
```

Les fichiers sont `chns_conservation.txt`, `chns_energy.txt` et
`chns_phase_energy.txt` dans le répertoire `Output/output directory`.
Ils utilisent les tableaux de FEZ,
avec une ligne initiale puis une ligne par pas accepté. Les trois traitements
sont désactivés par défaut dans les paramètres généraux et activés dans les
trois exemples de ce dossier. Ils concernent le modèle `quasi_incompressible`.
Ils fonctionnent sans sortie VTU et sans option de qualité du maillage.

Comme pour les autres tableaux, `output frequency` règle la fréquence de
réécriture du fichier contenant l'historique calculé. Avec `1`, il est
actualisé à chaque pas ; le dernier pas est toujours écrit. `write results =
false` conserve le calcul et l'historique en mémoire. `verbosity = verbose`
annonce les dernières valeurs ; `quiet` évite ces annonces.

Le tableau de conservation contient les intégrales de phase et de densité
affine, ainsi que la masse prolongée positive à titre de diagnostic.
`mass_change_from_reference` et `phase_change_from_reference` comparent à
l'état initial suivi ; `mass_relative_change` divise par sa masse, tandis que
`phase_volume_relative_change` divise par son volume. Cette normalisation
reste définie lorsque l'intégrale initiale de phase est nulle. Une masse
initiale nulle donne `nan` pour son erreur relative, avec conservation de
l'écart absolu. `mass_step_change` et `phase_step_change` comparent au pas précédent.
La conservation est attendue en domaine fermé sans sources. Avec des sources
fabriquées ou des flux imposés, ces colonnes mesurent des variations qui ne
sont pas automatiquement des erreurs de conservation.

Le tableau d'énergie sépare `kinetic_energy`, `bulk_energy`, `gradient_energy`
et `gravitational_energy`. `interfacial_energy` est la somme des deux
contributions interfaciales, et `total_energy` leur somme avec les énergies
cinétique et gravitationnelle. `energy_change_from_reference` mesure l'écart
depuis l'état initial suivi. Les dissipations, le bilan temporel testé et
les identités BDF2 à pas constant sont aussi disponibles ; leurs limites
théoriques sont celles décrites dans [bdf2.md](bdf2.md).

Le traitement indépendant `chns phase energy` exporte `time`, `iteration` et
les six composantes `kinetic_energy`, `bulk_energy`, `gradient_energy`,
`interfacial_energy`, `gravitational_energy` et `total_energy`, chacune avec
les suffixes `_phase_0` et `_phase_1`. Comme pour les volumes géométriques,
la phase 0 correspond aux points de quadrature où `phi >= 0`, la phase 1 à
`phi < 0`. Il s'agit de l'intégrale de chaque densité d'énergie sur cette
région, avec les mêmes définitions que pour l'énergie globale : densité
positive pour l'énergie cinétique et densité affine pour la gravitation.
La somme des deux phases restitue donc chaque composante globale, y compris
l'énergie interfaciale répartie entre les deux régions. Une phase absente
contribue zéro. Ce traitement fonctionne seul et réutilise le calcul des
intégrales globales ; il est aussi disponible pour le modèle `stepien`.

Une reprise charge les références et les tableaux depuis le fichier
supplémentaire `<nom du checkpoint>.postprocessing`. Conserver ce fichier
avec les autres fichiers de checkpoint pour maintenir la référence originale,
même sans les anciens fichiers de sortie. L'instant repris n'est pas ajouté
deux fois. Avec un ancien checkpoint sans ce fichier, le suivi commence à
l'instant repris : la colonne `reference_time` le signale explicitement.

Le format de checkpoint de post-traitement V2 conserve aussi le tableau
d'énergie par phase et reste capable de lire les fichiers V1. Si le suivi
par phase est activé pour la première fois à une reprise, son historique
commence à l'instant repris, sans modifier ni dupliquer les lignes des
tableaux globaux déjà suivis. Aucune énergie par phase antérieure n'est
reconstruite à partir des bilans globaux.

L'option détaillée historique est indépendante de ces trois traitements :

`write quasi incompressible diagnostics = true` produit
`quasi_incompressible_diagnostics.csv` dans le répertoire de sortie, dès l'état
initial et à chaque pas accepté, indépendamment de la sortie VTU.
Elle n'est pas nécessaire pour obtenir les deux tableaux standard.
Si les deux modes sont activés, ils partagent le même calcul des intégrales.

- `phase_integral`, `mass_affine` : quantités conservées en domaine fermé,
  sans sources ; `mass_positive` concerne l'extension et n'est pas une masse
  dont la conservation est démontrée.
- `total_energy` : énergie cinétique avec densité positive, interfaciale et
  gravitationnelle avec densité affine.
- `energy_inequality_residual` en BDF1 : $E^{n+1}-E^n+\Delta tD^{n+1}$, attendu négatif
  ou nul dans les conditions du théorème.
- `energy_balance_residual` en BDF1 : même quantité plus la dissipation numérique,
  attendue proche de zéro à la précision de Newton et de l'arithmétique.
- `phi_min_q`, `phi_max_q` : extrema aux points de quadrature ; ce ne sont pas
  nécessairement les extrema globaux du champ éléments finis.
- `density_affine_min_q`, `density_affine_max_q`,
  `negative_density_volume_q` : bornes observées de densité affine et mesure
  de la région négative, sans la masquer avec le prolongement positif.
- `phase_overshoot_volume_q`, `density_extension_l1`, `domain_volume` :
  région hors de $[-1,1]$, intégrale de l'écart entre densités prolongée et
  affine, et volume du domaine pour normaliser ces mesures.
- `numerical_kinetic_energy_loss`, `numerical_interfacial_energy_loss` :
  pertes numériques par pas BDF1. Leur somme est `numerical_energy_loss` ;
  comparer les sommes cumulées aux dissipations physiques intégrées en temps.
- `velocity_l2`, `velocity_divergence_l2`, `diffusive_flux_l2` : normes
  permettant de vérifier le repos, la divergence et l'absence de diffusion.
- `bubble_area`, `center_y`, `rise_velocity` : intégrales de l'indicatrice
  $\phi<0$, et non une reconstruction géométrique précise du contour.

En 2D, `bubble_area` est l'aire de la phase 1 ; en 3D, c'est son volume,
malgré le nom de colonne conservé. Ce diagnostic géométrique peut varier
alors que `phase_integral` et `mass_affine` restent conservées : il mesure
la région $\phi<0$, tandis que les deux autres intègrent le champ diffus.
Les tableaux FEZ existants `chns phase volume`, `chns phase center of mass`
et `chns phase average velocity` peuvent aussi être activés dans
`Postprocessing` pour suivre séparément les deux régions. Fixer un
`output prefix` distinct pour chaque tableau.

`effective_bdf_order` et `bdf_c0`, `bdf_c1`, `bdf_c2` indiquent le schéma
effectif, y compris au démarrage. Pour BDF2, contrôler
`temporal_energy_balance_residual`, les identités cinétique et de gradient
aux pas constants, puis les variations d'énergie physique. Les anciens
résidus énergétiques et pertes numériques propres à BDF1 valent `NaN` aux
pas BDF2. Le nouveau bilan testé ne constitue pas une preuve de décroissance
inconditionnelle de l'énergie totale ; voir [bdf2.md](bdf2.md).

Les résidus énergétiques des calculs manufacturés avec sources ou des
domaines ouverts incluent du travail extérieur non soustrait dans ce CSV.
Les bords ouverts ne font pas partie du périmètre physique accepté par ce
modèle. L'état initial n'est pas automatiquement équilibré pour $\mu$ et
$p$ ; les diagnostics au temps initial décrivent les champs prescrits.
Dans les VTU, `density` est la densité prolongée d'inertie et
`density_affine` la densité de masse et de gravité. Ce sont des moyennes DG0
par cellule ; contrôler aussi les extrema aux quadratures, car une moyenne
positive peut masquer une négativité dans une partie de la cellule.

## Vérifications du code

Après compilation, lancer d'abord les vérifications courtes :

```sh
ctest --test-dir buildRelease --output-on-failure -R 'quasi_incompressible|chns_postprocessing'
```

Pour exécuter directement les vérifications physiques ciblées, sans attendre
les solutions manufacturées et les études de convergence :

```sh
python3 tests/incompressible_chns/validate_quasi_incompressible.py \
  --solver ./buildRelease/incompressible_chns \
  --workdir ./buildRelease/verification_quasi_physics \
  --suite physics
```

Cette suite vérifie les phases exactement pures avec pression hydrostatique,
les rapports de densité et leur inversion, puis une relaxation à phase
non uniforme sans gravité : conservation, fermeture du bilan énergétique,
flux, divergence et diagnostics des dépassements. Une dérive de phase ou
un flux non nul dans une phase pure doit faire échouer la vérification.
Elle vérifie aussi un état homogène volontairement hors des bornes, afin
de contrôler les diagnostics de densité négative et les deux champs VTU.
Pour les Jacobiennes assemblées et les convergences, utiliser `--suite core`
dans un autre `--workdir`. La suite BDF2 est également disponible seule :

```sh
python3 tests/incompressible_chns/validate_quasi_incompressible.py \
  --solver ./buildRelease/incompressible_chns \
  --workdir ./buildRelease/verification_quasi_bdf2 \
  --suite bdf2
```

Elle mesure l'ordre au temps final sur les quatre champs, et contrôle aussi
les jacobiens, les phases pures, les interfaces et la reprise avec changement
de pas. `--suite all` est le défaut et regroupe les trois suites.

Une comparaison séquentiel/deux processus peut être exécutée séparément :

```sh
python3 tests/incompressible_chns/validate_quasi_incompressible.py \
  --solver ./buildRelease/incompressible_chns \
  --workdir ./buildRelease/verification_quasi_mpi \
  --mpiexec /usr/bin/mpirun
```

Adapter le chemin `--mpiexec` si MPI est installé ailleurs. Les fichiers
`.prm`, journaux `solver.log` et CSV de chaque cas restent dans le répertoire
fourni. Les vérifications portent sur les Jacobiennes par différences finies
en 2D/3D, les extensions, les mobilités, la différence divisée du potentiel,
l'identité énergétique, les solutions manufacturées et la conservation sur
petits cas fermés. Les études de convergence et la comparaison MPI font
partie du contrôle du code, sans validation des trois longues simulations
physiques. [VALIDATION.md](VALIDATION.md) distingue les tests exécutés des
calculs laissés au lancement par l'utilisateur.

Les futurs tests de goutte statique (Laplace, courants parasites) et les
comparaisons quantitatives complètes Hysing demandent encore des campagnes
séparées. Pour les bulles, examiner les contours, le centre et la vitesse
de montée avec les courbes de référence ; le CSV de base n'effectue pas une
reconstruction géométrique de circularité. Vérifier aussi les bornes de
densité, la dérive d'aire et les deux pertes numériques. Terminer à $t=3$
et fermer le bilan énergétique ne suffit pas à établir la convergence du
benchmark.

Les deux choix temporels sont `BDF1` et `BDF2`. Seul BDF1 possède ici la
preuve de décroissance inconditionnelle de l'énergie totale. L'ALE,
l'adaptation de maillage et les stabilisations SUPG/PSPG restent indisponibles.
Une seule jauge de pression doit être activée. En domaine non périodique,
chaque bord fluide doit être explicitement `no_slip` ou `slip` ; les
conditions Cahn–Hilliard sont `no_flux` ou naturelles homogènes si omises.
La boîte périodique exige une force massique nulle. Les conditions MMS de
Dirichlet restent disponibles pour les vérifications avec sources.
