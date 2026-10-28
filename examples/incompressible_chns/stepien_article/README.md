# Essais Stępień / Brunk–ten Eikelder avec BDF

Ces essais conservent la pression inconnue de FEZ, les pressions de référence
`pressure reference` des deux fluides, le potentiel chimique et BDF2 avec
démarrage BDF1. Les nouvelles options sont désactivées par défaut.

L'article de référence est Brunk & ten Eikelder, *A simple, fully-discrete,
unconditionally energy-stable method for the two-phase Navier-Stokes
Cahn-Hilliard model with arbitrary density ratios*, JCP 548 (2026), 114558.
La variante proposée ici ne reprend pas son schéma temporel et ne bénéficie
donc pas de son théorème de stabilité énergétique inconditionnelle.

## Variantes comparées

| Variante | Mobilité | Transport | Inertie |
|---|---|---|---|
| `baseline` | constante, `6.1e-6` par défaut | Stępień actuel | advective |
| `article_mobility` | loi du benchmark de l'article | Stępień actuel | advective |
| `conservative` | loi du benchmark de l'article | conservatif | advective |
| `conservative_skew` | loi du benchmark de l'article | conservatif | antisymétrique et variation de densité |

À résolution donnée, les quatre essais ont les mêmes données physiques,
conditions aux limites, maillage, épaisseur d'interface et pas de temps.
Le passage à la mobilité de l'article change à la fois la forme de la loi et
son amplitude. Pour comparer uniquement sa forme, fixer la mobilité constante
à son amplitude en `phi=0` via `--baseline-mobility`.

Les données physiques du fichier FEZ benchmark 2 sont conservées : densités
1000 et 1, viscosités dynamiques 10 et 0.1, tension 1.96, gravité 0.981.
La table 3 du PDF imprime une viscosité lourde de 1 : cette divergence doit
être clarifiée pour reproduire exactement les calculs des auteurs.

Le lanceur prend `h=1/cells`, un maillage carré de `cells` par `2*cells`,
`epsilon=0.64*h` et `dt=0.128*h`, selon la section 4.3 de l'article. Les deux
derniers paramètres peuvent être imposés séparément. Ces valeurs ne sont pas
celles de l'ancien fichier FEZ benchmark2.prm.

## Construction et lancement

Depuis la racine du dépôt :

```bash
cmake --build buildRelease --target incompressible_chns -j 2

# Vérification rapide des quatre chemins (ne valide pas le benchmark).
python3 examples/incompressible_chns/stepien_article/run_comparison.py \
  --cells 16 --t-end 0.04 --output tmp/incompressible_chns/article_smoke

# Comparaison jusqu'à t=3 ; augmenter cells pour une étude de convergence.
python3 examples/incompressible_chns/stepien_article/run_comparison.py \
  --cells 32 --t-end 3 --output tmp/incompressible_chns/article_h32
```

Le répertoire de sortie doit être vide ou inexistant. Chaque variante produit
son fichier `.prm`, son journal et son CSV de diagnostics. `summary.csv` et
`summary.json` contiennent les résultats et le statut effectif de chaque essai.
Un échec n'est pas considéré comme une simulation terminée. Si Matplotlib est
disponible, `comparison.png` est également produit. Le lanceur n'utilise sinon
que la bibliothèque standard Python.

Options utiles : `--generate-only`, `--variants article_mobility conservative`,
`--dt`, `--epsilon`, `--ranks`, `--timeout` (par variante),
`--pressure-ref0 3 --pressure-ref1 1`. Une simulation déjà exécutée n'est jamais
écrasée. Le benchmark utilise par défaut les mêmes pressions de référence,
égales à 1 ; le test manufacturé vérifie le cas différent, 3 et 1.

## Équations et options

Dans `Cahn Hilliard` :

```text
set CHNS model = stepien
set mobility model = degenerate_absolute
set enable density mobility compensation = false
set enable mobility tracer limiter = false
set stepien conservative transport = true
set stepien skew symmetric inertia = true
set write stepien diagnostics = true
```

La mobilité de bulle montante de l'article (p. 12) est

```
m_A(phi) = 0.1 epsilon^2 abs(1-phi^2).
```

FEZ diffuse `xi=(rho*mu-a*(p+p_r))/(rho0*rho1)`. En posant
`b=(rho0+rho1)/2`, `a=(rho0-rho1)/2`, `k=rho0*rho1/b`, on a
`w_A=mu-(a/b)*(p+p_r-phi*mu)=k*xi`. La nouvelle loi calcule donc
automatiquement `M_F=k^2*m_A`. Les paramètres `mobility` et
`degenerate mobility` sont ignorés pour cette loi. La compensation par densité
est rejetée pour éviter de changer cette définition. La valeur absolue est
conservée hors de [-1,1] ; aux deux coins, la dérivée choisie pour Newton est
zéro. Activer le limiteur de mobilité change ce prolongement et ne reproduit
plus exactement la loi du PDF.

Le transport conservatif utilise `J_A=(M_F/k)*grad(xi)` et les résidus faibles

```
(BDF(phi), psi) - (phi*u, grad(psi)) + (J_A, grad(psi)),
-(div(u), q) - alpha*(J_A, grad(q)),  alpha=-a/b.
```

Ces formes supposent une paroi imperméable avec `J_A.n=0`. Elles ne constituent
pas une implémentation des flux de bord ouverts. Les tests manufacturés ont
des dérivées normales de phi, mu et p nulles sur les parois.

L'inertie antisymétrique remplace la forme advective par

```
(rho*BDF(u) + 0.5*BDF(rho)*u, v)
+ 0.5*(rho*(u.grad(u)), v) - 0.5*(rho*(u.grad(v)), u).
```

`BDF(rho)` utilise les densités des niveaux temporels précédents, avec le même
limiteur de propriétés que la densité courante. Le BDF de la vitesse n'est pas
modifié. Au continu, cette réécriture ajoute la moitié de la vitesse multipliée
par le résidu de conservation de masse ; elle est équivalente sans sources
pour une densité affine non limitée. Avec BDF2, aucun bilan discret d'énergie
inconditionnel n'est affirmé. Le cubique implicite de FEZ reste inchangé.

La seconde viscosité de FEZ reste `lambda=-2*eta/3`. L'article choisit
`lambda=-2*eta/d` ; les conventions diffèrent donc en 2D. Ce choix est conservé
pour isoler les effets de la mobilité, du transport et de l'inertie.

Les nouvelles formes sont limitées à Stępień sur maillage fixe. Le chemin
`Stabilization` SUPG/PSPG n'est pas pris en charge pour ces variantes.

## Mesures et portée des résultats

### Post-traitement standard de l'énergie

Pour le modèle Stępień sur maillage fixe, activer la table standard FEZ avec :

```text
subsection Postprocessing
  subsection chns energy
    set enable = true
    set write results = true
    set output prefix = chns_energy
    set output frequency = 1
    set precision = 16
    set verbosity = quiet
  end
end
```

Le fichier `chns_energy.txt` est créé dans `Output/output directory`.
Il contient l'état initial puis chaque pas accepté, indépendamment de la
sortie VTU. `output frequency` règle la fréquence de réécriture du fichier ;
la table conserve tous les pas et est aussi écrite à la fin. Les tables et
leur référence initiale sont restaurées lors d'une reprise avec checkpoint.
Si le suivi d'énergie est activé seulement à la reprise, sa référence est
l'état chargé, au temps indiqué par `reference_time`.
`write stepien diagnostics` peut être désactivé : la table standard fonctionne
sans le CSV de diagnostic.

La table donne les énergies cinétique, de double puits, de gradient,
interfaciale, gravitationnelle et totale, les dissipations visqueuse et
diffusive et la variation d'énergie depuis le temps `reference_time`.
Avec `sigma_tilde = 3*sigma/(2*sqrt(2))` et l'épaisseur `epsilon`, la densité
d'énergie intégrée vaut :

```text
e = rho_used*|u|^2/2
  + sigma_tilde*(phi^2-1)^2/(4*epsilon)
  + sigma_tilde*epsilon*|grad(phi)|^2/2
  - rho_affine*(body_force . x).
```

`rho_affine=((rho0+rho1)+(rho0-rho1)*phi)/2` est la densité affine.
`rho_used` utilise le même limiteur de propriétés que le solveur : lorsque
`enable tracer limiter=true`, seule cette densité utilise `clip(phi,-1,1)`.
Le potentiel de gravité est pris nul à l'origine. La gravité et l'énergie
libre utilisent le champ `phi` brut, comme le CSV Stępień.

Chaque composante possède aussi deux colonnes avec les suffixes `_phase_0`
et `_phase_1`. Elles intègrent la même densité d'énergie, respectivement
sur les points de quadrature où `phi>=0` et `phi<0`. Cette convention suit
les volumes de phases de FEZ : la phase 0 correspond au fluide `phi=+1`,
la phase 1 au fluide `phi=-1`. Leur somme retrouve l'énergie globale.
L'énergie interfaciale est ainsi répartie selon ce découpage géométrique ;
il s'agit d'une convention de répartition, et non d'une énergie propre à
chacun des fluides. Une phase absente a une énergie nulle.

Le calcul utilise la quadrature du solveur. Une interface qui traverse les
points de quadrature peut faire varier la répartition entre phases.
Les colonnes de résidus temporels et d'énergies BDF modifiées propres au
modèle quasi-incompressible ne sont pas produites pour Stępień.
Ces énergies instantanées ne constituent pas une garantie de décroissance
pour le schéma BDF utilisé ici.

`stepien_diagnostics.csv` est écrit au temps initial et à chaque pas. Il donne :

- l'intégrale du champ de phase, la masse avec densité affine et celle avec la
  densité utilisée par le solveur ;
- l'aire, le centre vertical et la vitesse moyenne de la région `phi<0`,
  calculés par quadrature comme les indicateurs de l'article ;
- des indicateurs complémentaires pondérés par `(1-clip(phi))/2` ;
- l'énergie cinétique avec la densité utilisée, l'énergie interfaciale et
  l'énergie gravitationnelle avec la densité affine ;
- les dissipations visqueuse et diffusive et les extrêmes de phi aux points
  de quadrature.

Les extrêmes de quadrature ne prouvent pas une borne ponctuelle de phi.
L'aire définie par `phi<0` peut fluctuer lorsque l'interface traverse les points
de quadrature. La masse affine et la masse limitée peuvent différer en cas de
dépassement de phi ; les deux sont donc rapportées séparément. L'énergie est
un diagnostic et ne constitue pas une preuve pour le schéma hybride.

La validation du benchmark demande une convergence en maillage, épaisseur et
pas de temps et une comparaison aux données de référence. Le PDF signale
encore des écarts de vitesse de montée après t=1.5 avec les références à
interface nette du cas 2. Une simulation terminée et une énergie décroissante
ne suffisent donc pas à déclarer le benchmark validé.
