# CHNS quasi incompressible : équations et discrétisation dans FEZ

Ce document décrit la reformulation CHNS quasi incompressible du modèle de
Stępień et sa discrétisation dans FEZ. La construction énergétique
du schéma s'appuie sur la référence : A. Brunk et M. F. P. ten Eikelder,
*A simple, fully-discrete,
unconditionally energy-stable method for the two-phase Navier-Stokes
Cahn-Hilliard model with arbitrary density ratios*, JCP 548 (2026), 114558.
Les identités discrètes de référence sont celles du problème 1,
équations (38)–(41).
Voir aussi le [DOI](https://doi.org/10.1016/j.jcp.2025.114558).

Ce document décrit le système effectivement assemblé avec
`Cahn Hilliard/CHNS model = quasi_incompressible`. Les densités propres des
constituants sont constantes ; la vitesse moyenne massique n'est pas
nécessairement à divergence nulle. Le cadre est exclusivement **quasi
incompressible**, sans loi d'état compressible.

Les formes locales du résidu, leur Jacobien et l'assembleur volumique sont
regroupés dans le fichier existant
[`incompressible_chns_assemblers.h`](../../../include/assembly/incompressible_chns_assemblers.h),
dans l'espace `Assembly::IncompressibleCHNS::EnergyStable`. La sélection de
l'assembleur passe par la fabrique CHNS commune du même fichier.
Les commandes sont dans [README.md](README.md) et les vérifications
effectivement exécutées dans [VALIDATION.md](VALIDATION.md).

## Variables et lois constitutives

Les inconnues sont la vitesse moyenne massique $v$, la pression $p$, la
différence des fractions volumiques $\phi$ et le potentiel chimique $\mu$.
Dans FEZ, `Fluid 0` correspond à $\phi=+1$ (constituant 1 de l'article),
`Fluid 1` à $\phi=-1$ (constituant 2).
Les indices 0/1 de FEZ correspondent aux indices 1/2 du développement
analytique fourni par l'utilisateur.

$$
\rho(\phi)=\bar\rho+a\phi,\quad
\bar\rho=(\rho_0+\rho_1)/2,\quad a=(\rho_0-\rho_1)/2,\quad
\alpha=-a/\bar\rho.
$$

### Changement de variables depuis le développement analytique

Posons $r=\bar\rho$, $s=a$ et $k=\rho_0\rho_1/r$. Avec $P=p^*+p_r$ et
$\widehat\mu$ du système quasi incompressible de la page 3 du développement
analytique fourni, on définit

$$
p=P-\phi\widehat\mu,\qquad \mu=\widehat\mu,\qquad
w=\mu+\alpha p=\frac{\rho\widehat\mu-sP}{r}.
$$

Alors $-\nabla P+\widehat\mu\nabla\phi=-\nabla p-\phi\nabla\mu$.
Le potentiel diffusif du développement vaut $q_{\rm doc}=w/k$, et sa
mobilité est reliée à celle utilisée ci-dessous par

$$ M_{\rm doc}=k^2M. $$

Pour vérifier les facteurs, noter
$B=\nabla\cdot(M_{\rm doc}\nabla q_{\rm doc})$. Le bloc phase–divergence
du développement s'écrit

$$
D_t\phi=\frac{\rho}{\rho_0\rho_1}B,\qquad
\nabla\cdot v=-\frac{s}{\rho_0\rho_1}B.
$$

En ajoutant $\phi\nabla\cdot v$ à la première équation, on obtient
$\partial_t\phi+\nabla\cdot(\phi v)=B/k=\nabla\cdot(M\nabla w)$.
La seconde devient $\nabla\cdot v=\alpha\nabla\cdot(M\nabla w)$.
Le facteur constant $k^2$ convertit les conventions de potentiel et de flux ;
il ne constitue pas une compensation locale en densité.

L'équivalence suppose les mêmes lois d'énergie et de contrainte visqueuse.
Au niveau EF, $\phi_h\mu_h$ n'appartient pas en général à l'espace de pression :
on résout directement pour $p_h$, puis on peut reconstruire $P=p+\phi\mu$.
Ces deux champs ne doivent pas être identifiés dans les sorties.

### Prolongements et contrainte visqueuse

La viscosité dynamique affine est
$\eta(\phi)=(\eta_0+\eta_1)/2+(\eta_0-\eta_1)\phi/2$.
$\widetilde\rho(\phi)=\rho(\operatorname{clip}(\phi,-1,1))$ et
$\widetilde\eta(\phi)=\eta(\operatorname{clip}(\phi,-1,1))$ sont les extensions
positives (non négative pour la viscosité si une viscosité propre est nulle).
Les valeurs extérieures correspondent aux valeurs aux extrémités
de (2) : $\rho_1$ pour $\phi<-1$, $\rho_0$ pour $\phi>1$ avec les indices FEZ.
La phrase de la section 3.1 inverse les extrémités par rapport à (2) ; le code
conserve la continuité de (2). Il ne tronque pas l'inconnue $\phi$.

$$
S=\widetilde\eta\left(2D(v)-\frac2d(\nabla\cdot v)I\right),\qquad
D(v)=\tfrac12(\nabla v+\nabla v^T),\qquad w=\mu+\alpha p.
$$

$f(\phi)=C_w(\phi^2-1)^2/4$, $C_w=\widetilde\sigma/\epsilon$,
$\gamma=\widetilde\sigma\epsilon$, $\widetilde\sigma=3\sigma/(2\sqrt2)$.
Le paramètre de viscosité de FEZ est **cinématique** : $\eta_i=\rho_i\nu_i$.

Le coefficient volumique de la contrainte vaut $-2\widetilde\eta/d$ :
$-\widetilde\eta$ en 2D et $-2\widetilde\eta/3$ en 3D. Il n'y a pas de
viscosité volumique supplémentaire. C'est la contrainte déviatorique dans
la dimension de calcul, donnant
$S:\nabla v=2\widetilde\eta|\operatorname{dev}D(v)|^2\ge0$.
Un modèle 2D conservant la loi volumique 3D demanderait un autre choix
constitutif explicite.

### Mobilité dégénérée indépendante des densités

La mobilité est scalaire isotrope. On note $J=M\nabla w$ ; le flux diffusif
orienté selon la loi de Fick est $-J$. Les bulles utilisent

$$ M(\phi)=m_0\epsilon^2|1-\phi^2|,\qquad m_0=0.1, $$

avec `mobility model = degenerate_absolute` et
`degenerate absolute mobility prefactor = 0.1`. Ce préfacteur est non négatif,
réglable et indépendant des densités. Le paramètre `mobility` sert à la
loi `constant`, pas à régler cette loi.

Une loi quadratique se configure par `mobility model = degenerate` et
l'expression `a*(1-x*x)^2`, avec `x` représentant $\phi$ et `a` l'amplitude
complète. L'expression n'est pas automatiquement multipliée par le préfacteur
de mobilité absolue, $\epsilon^2$ ou le paramètre `mobility`. La séparation de phases prend
$a=0.01$.

Les exemples désactivent `enable mobility tracer limiter` et
`enable density mobility compensation`. La compensation locale des variantes
Stepien n'est pas admise dans le modèle quasi incompressible. La dégénérescence
$M(\pm1)=0$ annule le flux en phase exactement pure, même avec un gradient
de pression. Le flux peut rester non nul dans une phase presque pure.
Les deux lois ci-dessus redeviennent positives hors de $[-1,1]$ et ne
garantissent pas les bornes de phase.

## Forme forte continue

Dans l'intervalle physique, les deux premières équations scalaires impliquent
la conservation $\partial_t\rho+\nabla\cdot(\rho v)=0$ puisque
$r\alpha+s=0$. Le système conservatif équivalent est

$$
\begin{aligned}
\nabla\cdot v-\alpha\nabla\cdot J&=0,\\
\partial_t(\rho v)+\nabla\cdot(\rho v\otimes v)
-\nabla\cdot S+\nabla p+\phi\nabla\mu&=\rho b,\\
\partial_t\phi+\nabla\cdot(\phi v)-\nabla\cdot J&=0,\\
\mu+\gamma\Delta\phi-f'(\phi)&=0,\qquad J=M\nabla(\mu+\alpha p).
\end{aligned}
$$

La forme antisymétrique continue correspondante, système (10), est

$$
\begin{aligned}
\nabla\cdot v-\alpha\nabla\cdot(M\nabla w)&=0,\\
\rho\partial_t v+\tfrac12v\partial_t\rho
+\rho(v\cdot\nabla)v+\tfrac12v\nabla\cdot(\rho v)
-\nabla\cdot S+\nabla p+\phi\nabla\mu&=\rho b,\\
\partial_t\phi+\nabla\cdot(\phi v)-\nabla\cdot(M\nabla w)&=0,\\
\mu+\gamma\Delta\phi-f'(\phi)&=0.
\end{aligned}
$$

Pour les extensions hors de $[-1,1]$, le terme cinétique utilise
$\widetilde\rho\partial_t v+v\partial_t\widetilde\rho/2$.
La convection et la gravité utilisent toujours la densité **affine** $\rho$.
L'écart entre l'inertie antisymétrique non prolongée et $\rho D_tv$ vaut
$v[\partial_t\rho+\nabla\cdot(\rho v)]/2$, nul par conservation de masse.
Hors de $[-1,1]$, la formulation prolongée est un choix mathématique pour
le bilan cinétique ; elle n'est plus la quantité de mouvement conservative
du mélange physique de densité affine.
La source manufacturée implémente cette forme continue, avec
$\nabla\cdot(M\nabla w)=M(\Delta\mu+\alpha\Delta p)
+M'(\phi)\nabla\phi\cdot\nabla w$.

## Espaces, conditions aux limites et forme faible continue

On considère des parois fermées, soit en adhérence $v=0$, soit en glissement
$v\cdot n=0$ avec $(Sn)_\tau=0$, et les conditions naturelles
$\nabla\phi\cdot n=0$ et $J\cdot n=0$. Elles annulent le travail des parois
et les flux de masse/phase à travers le bord. Les bulles utilisent le
glissement sur les côtés et l'adhérence en haut et en bas.
L'autre option est une boîte périodique sur tous les champs ; FEZ y impose
$b=0$, car $-b\cdot x$ ne serait pas un potentiel périodique.

Formellement, prendre $V\subset H^1(\Omega)^d$ satisfaisant les conditions
homogènes de vitesse, $Q=H^1(\Omega)/\mathbb R$ et $W=H^1(\Omega)$,
ou leurs versions périodiques. La pression ne dépend que de sa classe à une
constante près. Les bulles imposent $\int p=0$ ; la boîte périodique fixe un
degré de liberté de pression intérieur. Il faut choisir exactement une jauge.
Le potentiel $\mu$ est fixé par son équation, sans jauge supplémentaire.
Pour les conditions MMS non homogènes, les espaces de tests portent les
contraintes homogènes correspondantes.

Le périmètre physique implémenté est fermé : chaque bord non périodique du
maillage doit avoir une condition fluide explicite `no_slip` ou `slip`.
Les conditions Cahn–Hilliard sont `no_flux`, ou les conditions naturelles
homogènes quand aucune entrée n'est fournie. Les conditions fluides
`velocity_mms` et Cahn–Hilliard `dirichlet_mms` sont réservées aux
vérifications avec sources manufacturées ; les bords ouverts ou les parois
mobiles ne font pas partie du modèle énergétique implémenté.

Avec $(\cdot,\cdot)$ le produit $L^2(\Omega)$ et

$$
c_s(a,v,z)=\tfrac12((\nabla v)a,z)-\tfrac12((\nabla z)a,v),
$$

chercher $(v,p,\phi,\mu)\in V\times Q\times W\times W$ annulant, pour tous
les tests $(z,q,\psi,\xi)$ dans ces espaces,

$$
\begin{aligned}
R_v^{\rm cont}={}&(\widetilde\rho\partial_t v+
\tfrac12v\partial_t\widetilde\rho,z)+c_s(\rho v,v,z)
+(S,\nabla z)-(p,\nabla\cdot z)+(\phi\nabla\mu-\rho b,z),\\
R_p^{\rm cont}={}&-(\nabla\cdot v,q)-\alpha(J,\nabla q),\\
R_\phi^{\rm cont}={}&(\partial_t\phi,\psi)-(\phi v,\nabla\psi)
+(J,\nabla\psi),\\
R_\mu^{\rm cont}={}&(\mu-f'(\phi),\xi)-\gamma(\nabla\phi,\nabla\xi).
\end{aligned}
$$

Pour un flux convectif sans composante normale, $c_s(a,v,z)$ correspond à
$((a\cdot\nabla)v+v\nabla\cdot a/2,z)$ et vérifie $c_s(a,v,v)=0$.

## Forme forte discrète en temps : BDF1

Posons $\delta_t z=(z^{n+1}-z^n)/\Delta t$. Toutes les quantités étoilées de
l'article sont prises en $n+1$, comme dans sa section 4. Les équations sont

$$
\begin{aligned}
\nabla\cdot v-\alpha\nabla\cdot(M\nabla w)&=0,\\
\tfrac12v\delta_t\widetilde\rho+\widetilde\rho^n\delta_t v
+\rho(v\cdot\nabla)v+\tfrac12v\nabla\cdot(\rho v)
-\nabla\cdot S+\nabla p+\phi\nabla\mu&=\rho b,\\
\delta_t\phi+\nabla\cdot(\phi v)-\nabla\cdot(M\nabla w)&=0,\\
\mu+\gamma\Delta\phi-\overline{f'}(\phi,\phi^n)&=0.
\end{aligned}
$$

Ici les inconnues sans exposant sont au nouveau pas. Le coefficient de
$\delta_t v$ est celui de (40), **$\widetilde\rho^n$**. Le choix `scheme=BDF1`
fournit l'historique et le pas de temps ; l'assembleur calcule cette inertie
et le potentiel moyen explicitement. Il ne s'agit pas d'appliquer BDF1
directement à toutes les équations continues.
Ce schéma est du premier ordre en temps. Le choix `BDF2` utilise une
discrétisation distincte de l'inertie et du potentiel, documentée avec ses
équations fortes, faibles et son jacobien dans [bdf2.md](bdf2.md).
Le démarrage d'un calcul BDF2 réutilise le schéma BDF1 écrit ici.

La différence divisée est évaluée sans division ni annulation numérique :

$$
\overline{f'}(A,B)=\frac{C_w}4(A+B)(A^2+B^2-2),\quad
\partial_A\overline{f'}=\frac{C_w}4(3A^2+2AB+B^2-2).
$$

Elle vérifie $\overline{f'}(A,B)(A-B)=f(A)-f(B)$, y compris par continuité
lorsque $A=B$.

## Résidus faibles assemblés : BDF1

Le maillage est fixe et conforme. Les bulles utilisent Taylor–Hood
$P_2/P_1$ pour vitesse/pression, et $P_1/P_1$ pour phase/potentiel sur
triangles. La boîte périodique utilise $Q_2/Q_1$ et $Q_1/Q_1$.
On cherche les quatre champs dans ces espaces EF, avec les conditions et
la jauge précédentes, en annulant les résidus ci-dessous pour tous les tests
EF correspondants. Le solveur est monolithique ; l'historique est fixe
pendant la résolution d'un pas.

Sur domaine périodique, ou avec $v\cdot n=0$, glissement sans traction
tangentielle ou adhérence, et $\nabla\phi\cdot n=M\nabla w\cdot n=0$,
les tests sont $(z,q,\psi,\xi)$. Les conditions de Dirichlet manufacturées
utilisent les espaces de tests homogènes correspondants.

$$
c_s(k,v,z)=\tfrac12((\nabla v)k,z)-\tfrac12((\nabla z)k,v).
$$

La convention deal.II est $(\nabla v)_{ij}=\partial_jv_i$.

$$
\begin{aligned}
R_v={}&(\tfrac12v\delta_t\widetilde\rho+
\widetilde\rho^n\delta_t v,z)+c_s(\rho v,v,z)
+(S,\nabla z)-(p,\nabla\cdot z)+(\phi\nabla\mu-\rho b,z),\\
R_p={}&-(\nabla\cdot v,q)-\alpha(M\nabla w,\nabla q),\\
R_\phi={}&(\delta_t\phi,\psi)-(\phi v,\nabla\psi)
+(M\nabla w,\nabla\psi),\\
R_\mu={}&(\mu-\overline{f'},\xi)-\gamma(\nabla\phi,\nabla\xi).
\end{aligned}
$$

$R_p$ est l'opposé de (41), conformément au signe Stokes de FEZ. Cela ne
change pas l'équation. Le second membre Newton est **$-R$**, la matrice est
**$DR$**. Les sources prescrites s'ajoutent à chaque résidu avec le signe FEZ ;
les sources MMS sont les opposés des résidus forts, sauf la source pression
qui vaut $\nabla\cdot v-\alpha\nabla\cdot(M\nabla w)$.

Pour tout test $a_h$ appartenant à la fois aux espaces de phase et de
pression, la combinaison $sR_\phi[a_h]-rR_p[a_h]$ donne la conservation
de masse faible

$$ (\delta_t\rho,a_h)-(\rho v,\nabla a_h)=0. $$

Une constante peut être retirée du test de pression pour respecter sa jauge,
sans changer l'identité en domaine fermé. Le test de phase constant conserve
ainsi $\int\phi$ et donc $\int\rho$. La même évaluation de $J$ et la même
quadrature servent aux lignes phase et pression et aux couplages mécaniques.

## Jacobien

À chaque itération de Newton, on résout $DR(U)[\delta U]=-R(U)$, puis on
met à jour les quatre champs, avec recherche linéaire si activée. Pour une
direction $(h,\pi,\chi,\zeta)$, historique et sources restent fixes :

$$
\begin{aligned}
\delta\rho&=a\chi,&
\delta\widetilde\rho&=a\operatorname{clip}'(\phi)\chi,\\
\delta J&=M'\chi\nabla w+M\nabla(\zeta+\alpha\pi),&
\delta I&=\frac{\widetilde\rho+\widetilde\rho^n}{2\Delta t}h
+\frac{\delta\widetilde\rho}{2\Delta t}v.
\end{aligned}
$$

$$
\begin{aligned}
DR_v={}&(\delta I-\delta\rho b+\chi\nabla\mu+\phi\nabla\zeta,z)
+c_s(\delta\rho v+\rho h,v,z)+c_s(\rho v,h,z)\\
&+\left(\delta\widetilde\eta(2D(v)-\tfrac2d\operatorname{div}v I)
+\widetilde\eta(2D(h)-\tfrac2d\operatorname{div}h I),\nabla z\right)
-(\pi,\operatorname{div}z),\\
DR_p={}&-(\operatorname{div}h,q)-\alpha(\delta J,\nabla q),\\
DR_\phi={}&(\chi/\Delta t,\psi)-(\chi v+\phi h,\nabla\psi)
+(\delta J,\nabla\psi),\\
DR_\mu={}&(\zeta-\partial_A\overline{f'}\chi,\xi)
-\gamma(\nabla\chi,\nabla\xi).
\end{aligned}
$$

Les dérivées des extensions sont nulles hors de l'intervalle et prises du
côté intérieur aux extrémités. Pour $|1-\phi^2|$, le code choisit la dérivée
généralisée nulle en $\phi=\pm1$. Les différences finies classiques sont
testées à distance de ces points non différentiables.
Pour une expression utilisateur de mobilité, FEZ appelle les dérivées
numériques de `ParsedFunction` ; les formes directionnelles ci-dessus sont
exactes, mais cette évaluation constitutive reste soumise à la précision
des différences finies. La loi `degenerate_absolute` a une dérivée analytique.

La variation de viscosité est
$\delta\widetilde\eta=(\eta_0-\eta_1)\operatorname{clip}'(\phi)\chi/2$.
Ces expressions incluent les deux occurrences de vitesse dans la convection,
la variation de densité de convection et de gravité, ainsi que $M'$ dans
les deux lignes diffusives. Les sources MMS correspondent à la forme
continue : elles permettent de mesurer l'erreur du schéma temporel et
n'imposent pas une solution artificiellement exacte du schéma discret.

## Bilan énergétique et limites de validation

Avec $V=-b\cdot x$,
$\widetilde E=\int\widetilde\rho|v|^2/2+f(\phi)+\gamma|\nabla\phi|^2/2+\rho V$.
Au continu, les tests $z=v$, $\psi=\mu$, $q=-p$ et
$\xi=-\partial_t\phi$ dans la somme des résidus donnent d'abord
$d(K+F)/dt+D=(\rho b,v)$, où $K$ et $F$ sont les énergies cinétique et
interfaciale. Le test de masse $V$ ajoute l'énergie gravitationnelle pour
obtenir $d\widetilde E/dt+D=0$. La convection s'annule par antisymétrie, les
couplages capillaires se compensent et les deux flux diffusifs se combinent
en $M|\nabla(\mu+\alpha p)|^2$.

Au discret, prendre $z_h=v$, $\psi_h=\mu$, $q_h=-p$ et
$\xi_h=-\delta_t\phi$. L'identité de différence divisée traite exactement
le potentiel ; les identités algébriques temporelles sont

$$
\begin{aligned}
I\cdot v&=\delta_t\!\left(\tfrac12\widetilde\rho|v|^2\right)
+\frac{\widetilde\rho^n}{2\Delta t}|v-v^n|^2,\\
\gamma\nabla\phi\cdot\nabla\delta_t\phi
&=\delta_t\!\left(\tfrac\gamma2|\nabla\phi|^2\right)
+\frac\gamma{2\Delta t}|\nabla\phi-\nabla\phi^n|^2.
\end{aligned}
$$

Ici $I=\widetilde\rho^n\delta_t v+v\delta_t\widetilde\rho/2$.
Pour la gravité, la conservation de masse faible avec $a_h=V$ donne
$(\delta_t\rho,V)=-(\rho b,v)$. Cette étape exige que $V$ appartienne aux
espaces scalaires communs, à une constante de jauge près ; c'est le cas
pour la gravité constante sur les maillages affines des exemples.
La boîte entièrement périodique est utilisée sans gravité.

Sans sources, avec les conditions fermées/périodiques compatibles,

$$
\begin{aligned}
\widetilde E^{n+1}-\widetilde E^n+\Delta t D^{n+1}+N^{n+1}&=0,\\
D&=\int M|\nabla w|^2+2\widetilde\eta|\operatorname{dev}D(v)|^2,\\
N&=\int\tfrac12\widetilde\rho^n|v^{n+1}-v^n|^2
+\tfrac\gamma2|\nabla\phi^{n+1}-\nabla\phi^n|^2.
\end{aligned}
$$

Notons séparément $N_{\rm kin}$ et $N_{\rm int}$ les deux termes de $N$.
Ce sont des pertes **par pas**, alors que $D$ est un taux. Leur positivité
donne l'inégalité énergétique sans restriction sur $\Delta t$, à condition
de résoudre les équations non linéaires. Elle ne donne pas une garantie
de convergence de Newton ni de précision pour un grand pas.

La masse conservée est $\int\rho(\phi)$, et non $\int\widetilde\rho(\phi)$.
Les diagnostics utilisent la même quadrature que l'assemblage. Ils donnent
donc le bilan de l'énergie évaluée par cette quadrature, à la précision de
la résolution non linéaire. Les lois non polynomiales et les extensions
peuvent nécessiter une quadrature plus fine pour approcher l'intégrale exacte.
Les sources MMS et les bords ouverts peuvent apporter de l'énergie ; leur
bilan n'est pas une vérification de décroissance de l'énergie sans travail.

La première version utilise un maillage fixe, un couple Taylor–Hood et les
mêmes espaces pour $\phi$ et $\mu$, sans stabilisation supplémentaire.
L'inégalité démontrée ne fournit ni principe du maximum pour $\phi$, ni
garantie de convergence de Newton pour tout pas de temps.

### Diagnostics de précision et de validité physique

Pour $\rho_0=1000$ et $\rho_1=1$, la densité affine devient négative dès que
$\phi<-1001/999\simeq-1.002002$. L'énergie cinétique prolongée peut rester
positive et son bilan exact malgré ce défaut physique. On suit donc les
bornes et la région négative en plus de l'énergie.

| Colonnes du CSV | Interprétation |
|---|---|
| `phase_integral`, `mass_affine` | Quantités conservées en domaine fermé sans sources |
| `phase_change`, `mass_change` | Variations d'un pas accepté au suivant |
| `mass_positive`, `density_extension_l1` | Masse prolongée et $\int|\widetilde\rho-\rho|$ |
| `density_affine_min_q`, `density_affine_max_q` | Extrema de la densité affine aux quadratures |
| `negative_density_volume_q` | Mesure par quadrature de la région $\rho<0$ |
| `phi_min_q`, `phi_max_q`, `phase_overshoot_volume_q` | Extrema et mesure de la région $|\phi|>1$ aux quadratures |
| `domain_volume` | Volume du domaine utilisé pour normaliser ces mesures |
| `viscous_dissipation`, `diffusive_dissipation` | Les deux taux constitutifs de dissipation |
| `numerical_kinetic_energy_loss`, `numerical_interfacial_energy_loss` | $N_{\rm kin}$ et $N_{\rm int}$ en BDF1 |
| `numerical_energy_loss` | $N_{\rm kin}+N_{\rm int}$ |
| `energy_inequality_residual` | $\widetilde E^{n+1}-\widetilde E^n+\Delta tD^{n+1}$, attendu $\le0$ |
| `energy_balance_residual` | Même quantité plus $N$, attendue proche de zéro |
| `velocity_l2`, `velocity_divergence_l2`, `diffusive_flux_l2` | Normes utiles aux vérifications de phase pure, divergence et flux |

Les volumes et extrema sont calculés aux quadratures ; ils ne sont pas des
bornes rigoureuses sur tout l'élément. Une faible erreur de bilan ne suffit
pas à valider un résultat avec densité affine négative. Une forte part de
perte numérique peut aussi masquer une erreur de vitesse : il faut raffiner
le pas et examiner séparément $N_{\rm kin}$ et $N_{\rm int}$.

Dans les sorties VTU, `density` reste le champ prolongé utilisé par
l'inertie ; `density_affine` permet de visualiser la densité de masse et de
gravité. Ces deux sorties DG0 sont des moyennes par cellule ; une moyenne
positive peut encore contenir une région localement négative. Les extrema
et volumes aux quadratures restent nécessaires pour contrôler ce défaut.

L'état initial ne résout pas automatiquement un problème séparé d'équilibre
pour $\mu$ et $p$. Le potentiel volumique n'est pas initialisé par une
résolution de son équation ; les éventuelles traces MMS sont imposées aux
bords. Les diagnostics de flux au temps initial décrivent l'état fourni ;
l'identité énergétique porte sur les pas résolus. Les vérifications
physiques et les études Hysing se lancent séparément des tests courts,
suivant le README.
