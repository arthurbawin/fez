# État de la stratégie FSI au 27 novembre 2026

Branche locale : `origin/iterative_solveur/iterative_FSI`, issue de
`b5eeefcf0be668b823071d8db334c0c31e807f3f`. Aucun push effectué.

## Portée et validation bloquante

Le document fourni impose P0 avant la validation de P1, puis P2/P3. Cette branche
prépare P0/P1 : exécutable `iterative_fsi`, FGMRES droit sur la matrice historique,
partition d'indices par rôle, résidu vrai, comparaison des corrections à MUMPS,
diagnostics mécaniques et différences finies globales. **Le code C++ n'a pas été
compilé ou exécuté dans l'environnement de développement Windows : CMake et
deal.II/PETSc ne sont pas disponibles, WSL n'est pas activé.** Les tests ci-dessous
sont préparés, pas déclarés réussis. Aucun gain de performance n'est revendiqué.
Vérifications locales effectuées : `git diff --check`, syntaxe Python du script,
génération des trois entrées de test et extraction des observables depuis les
sorties versionnées (12 valeurs force/position, 36 valeurs MMS). clang-format 16
n'est pas disponible ; son installation dans un répertoire temporaire n'a pas
abouti. Le formatage CI reste donc également à exécuter.

P2 (condensation ALE et cache des réponses) et P3 (Schur pression, AMG de volume,
patches d'interface et correction globale) restent non implémentés, conformément
aux contrôles numériques préalables bloquants du document. GMG, matrix-free et
Newton–Euler 3D ne sont pas ajoutés. Les IS sont préparés mais ne définissent pas
encore un PCFIELDSPLIT : pression, trace et modes rigides ne doivent pas être
factorisés isolément en supposant leurs blocs inversibles.

## Utilisation

La configuration CMake existante découvre automatiquement les trois nouveaux
fichiers C++/en-tête et le nouvel exécutable. Le solveur dérive de `FSISolver` :
aucune copie de son assemblage, de Newton, des CL ou des équations mécaniques.
Le paramètre commun accepte désormais `fgmres`. MUMPS reste la valeur par défaut.
Dans un fichier de paramètres historique :

```
subsection Linear solver
  subsection main physics
    set method = fgmres
    set preconditioner = block_jacobi
    set right preconditioning = true
    set relative tolerance = 1e-11
    set absolute tolerance = 1e-13
    set max iterations = 1000
    set max krylov vectors = 100
    set reuse = true
  end
end
```

Commandes depuis un environnement FEZ configuré (adapter les chemins des maillages) :

```
cmake -S . -B build -DENABLE_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target iterative_fsi monolithic_fsi
ctest --test-dir build -R '^iterative_fsi/' --output-on-failure
mpiexec -n 4 build/iterative_fsi case.prm -fsi_sub_pc_type ilu -fsi_sub_pc_factor_shift_type nonzero -fsi_mechanical_diagnostics -log_view
```

Options propres : `-fsi_compare_mumps`, `-fsi_comparison_tolerance` (défaut 1e-7,
erreur relative de correction), `-fsi_check_jacobian`, `-fsi_fd_step` (1e-6),
`-fsi_mechanical_diagnostics`. Les autres options KSP/PC portent le préfixe `fsi_`.
Le `ilu fill level` du .prm concerne l'ILU sériel ; pour les blocs MPI employer
`-fsi_sub_pc_factor_levels`. Le décalage de pivots modifie uniquement le PC.
Il n'est pas une correction de l'opérateur physique, ni une garantie de convergence.

La comparaison MUMPS supplémentaire coûte une factorisation à chaque correction :
à réserver aux petits problèmes. Pour vérifier FGMRES indépendamment de la qualité
de l'ILU, employer `-fsi_pc_type lu -fsi_pc_factor_mat_solver_type mumps`.
Les six tests CTest (trois cas, MPI 1/4) utilisent cette référence LU et ne constituent
pas un benchmark itératif scalable. `compare.py --pc bjacobi` permet d'exercer la
baseline ILU séparément et conserve entrées, commandes et logs dans le build.

Le résidu vrai est mesuré avant distribution des contraintes, sur le système
linéaire effectivement résolu. Les corrections MUMPS/FGMRES sont comparées après
distribution. Un échec KSP ou de résidu vrai n'est pas accepté silencieusement.
Si un PC réutilisé échoue, une seule nouvelle tentative le reconstruit. Le PC est
invalidé à chaque nouveau pattern, changement de coefficient BDF et croissance
des itérations ; son opérateur reste toujours la Jacobienne courante.

## Anomalies historiques, laissées intactes

1. Ressort sans masse et accumulateurs : le RHS historique de ces lignes vaut
   zéro. Le contrôle Jv complète uniquement un **résidu diagnostique** avec les
   relations affines correspondant aux lignes manuelles, offsets de référence
   compris. Le RHS de production est restauré après contrôle et reste inchangé.
   Les défauts diagnostiques sont imprimés hors norme Newton ; aucune projection
   mécanique nouvelle n'est appliquée. Les normes sont celles des lignes mises
   à l'échelle (p. ex. déplacement-force/k), pas une force dimensionnée universelle.
2. Masse non nulle avec couplage 4 : pattern autorisé mais lignes dynamiques
   absentes dans la branche matricielle historique. Le nouvel exécutable refuse
   ce cas et exige le couplage 3 pour la référence dynamique ; l'ancien est inchangé.
3. Rotation 2D : bras de levier intrinsèque fixe et diagonale angulaire remplacée
   par 1 si presque nulle. Le test global peut révéler cette régularisation et
   échouer ; il ne la masque pas. Rotation 3D libre/inertie non validées.
4. Sources non uniformes, traction MMS et rotation prescrite : certaines dérivées
   géométriques historiques sont incomplètes. Utiliser d'abord `jacobian_matrix.prm`
   sans source non uniforme pour le contrôle strict Jv. Les MMS à Lamé variables
   sont comparés à MUMPS sans prétendre que leur Jacobienne est exacte.
5. AMR/restart : rôles algébriques stockés dans des DoF promus. Le nouveau solver
   reconstruit ses IS/PC lors du changement de pattern, mais n'apporte pas un
   transfert mécanique persistant. Cela reste une condition préalable de P2.
6. La diagonale pression/trace est nulle ; ILU/block Jacobi peut échouer ou devenir
   inefficace, surtout avec plusieurs rangs. Aucun fallback direct automatique
   ne transforme un échec itératif en faux succès.

## Suite requise avant acceptation

Compiler Debug/Release, exécuter P0/P1 MPI 1/4, contrôler les sorties mécaniques
et les sources ci-dessus. Élargir aux rotations, 3D, autres stratégies sans masse,
pas variable, AMR et redémarrage. Mesurer forces, couples, divergence, défaut de
trace, trajectoire, erreurs MMS, rejets de pas, mémoire et temps séparés. L'actuel
script compare forces/positions et tables MMS lorsqu'elles sont imprimées ; il
ne calcule pas encore une norme volumique indépendante de divergence ni un couple
physique indépendant. Le test Jv utilise une direction déterministe globale :
multiplier directions et tailles de perturbation avant certification exhaustive.

La condensation devra garder le terme `K_II^{-1} b_I` et les dérivées H, vérifier
coercivité, temps des Lamé et signature du cache ; ne pas former de Schur dense.
Les corrections d'interface devront garder une correction globale des modes
rigides et de la compliance fluide. Ne pas désigner les prérequis préparés ici
comme la réalisation complète de ces deux étapes.

API consultées : [KSPFGMRES](https://petsc.org/release/manualpages/KSP/KSPFGMRES/),
[réutilisation PC](https://petsc.org/release/manualpages/KSP/KSPSetReusePreconditioner/),
[index sets explicites](https://petsc.org/release/manualpages/PC/PCFieldSplitSetIS/).
Vérifier ces API avec la version réellement utilisée pour compiler.
