# Extension temporelle BDF2 du modèle quasi incompressible

Ce document décrit la branche BDF2 implémentée dans le même assembleur que
le schéma BDF1. Le système continu, les conventions de pression, les espaces
d'éléments finis et les conditions aux limites sont ceux de
[equations.md](equations.md). Le choix temporel ne change pas les lois
physiques. La preuve de décroissance inconditionnelle de l'énergie totale
du schéma BDF1 ne s'étend pas automatiquement à BDF2.

## Champs, coefficients et conditions aux limites

La vitesse moyenne massique est \(\boldsymbol v\), la différence des fractions
volumiques est \(\phi\), le potentiel chimique est \(\mu\), et la pression
résolue est \(p=P-\phi\mu\). Les indices suivent FEZ :
\(\phi=1\) correspond au fluide 0 et \(\phi=-1\) au fluide 1.

Les lois utilisées sont

$$
\begin{aligned}
\rho(\phi)&=\frac{\rho_0+\rho_1}{2}+\frac{\rho_0-\rho_1}{2}\phi,\\
\widetilde\rho(\phi)&=\frac{\rho_0+\rho_1}{2}
 +\frac{\rho_0-\rho_1}{2}\max(-1,\min(1,\phi)),\\
\widetilde\eta(\phi)&=\frac{\eta_0+\eta_1}{2}
 +\frac{\eta_0-\eta_1}{2}\max(-1,\min(1,\phi)),\\
\alpha&=\frac{\rho_1-\rho_0}{\rho_0+\rho_1},\\
m(\phi)&=m_0\epsilon^2|1-\phi^2|,\\
f(\phi)&=\frac{3\sigma}{8\sqrt{2}\epsilon}(\phi^2-1)^2.
\end{aligned}
$$

La loi quadratique configurée par expression est également supportée.
La mobilité ne reçoit aucun facteur de densité. Les viscosités dynamiques
sont \(\eta_i=\rho_i\nu_i\), où FEZ demande les viscosités cinématiques
\(\nu_i\). Seuls les coefficients d'inertie et de viscosité sont prolongés :
l'inconnue \(\phi\) reste libre et la densité de masse reste affine.

Sur un domaine fermé, les conditions physiques sont
\(\boldsymbol v=0\), ou \(\boldsymbol v\cdot\boldsymbol n=0\) avec traction
tangentielle nulle, ainsi que

$$
\nabla\phi\cdot\boldsymbol n=0,\qquad
m(\phi)(\nabla\mu+\alpha\nabla p)\cdot\boldsymbol n=0.
$$

La boîte périodique impose la périodicité de tous les champs et requiert
une force massique nulle. Une seule jauge de pression est imposée.
Les conditions fabriquées ont leurs traces propres. L'équation faible de
pression conserve une condition naturelle de flux normal nul ; une solution
fabriquée doit aussi satisfaire cette condition.

## Système fort continu

Pour une force massique \(\boldsymbol b\), sans sources fabriquées :

$$
\begin{aligned}
&\widetilde\rho(\phi)\partial_t\boldsymbol v
 +\frac12\boldsymbol v\,\partial_t\widetilde\rho(\phi)
 +\rho(\phi)(\boldsymbol v\cdot\nabla)\boldsymbol v
 +\frac12\boldsymbol v\,\nabla\cdot(\rho(\phi)\boldsymbol v)\\
&\quad-\nabla\cdot\left[
 \widetilde\eta(\phi)
 \left(\nabla\boldsymbol v+\nabla\boldsymbol v^T
 -\frac2d(\nabla\cdot\boldsymbol v)\boldsymbol I\right)\right]
 +\nabla p+\phi\nabla\mu=\rho(\phi)\boldsymbol b,\\
&\nabla\cdot\boldsymbol v
 -\alpha\nabla\cdot\left[m(\phi)(\nabla\mu+\alpha\nabla p)\right]=0,\\
&\partial_t\phi+\nabla\cdot(\phi\boldsymbol v)
 -\nabla\cdot\left[m(\phi)(\nabla\mu+\alpha\nabla p)\right]=0,\\
&\mu-\frac{3\sigma}{2\sqrt2\epsilon}(\phi^3-\phi)
 +\frac{3\sigma\epsilon}{2\sqrt2}\Delta\phi=0.
\end{aligned}
$$

Le prolongement cinétique est celui de la version BDF1. Pour
\(-1<\phi<1\), il coïncide avec la densité affine. Les lois de conservation
portent sur cette dernière.

## Coefficients temporels et démarrage

Avec \(\Delta t_{n+1}=t^{n+1}-t^n\) et \(\Delta t_n=t^n-t^{n-1}\),
les coefficients calculés par le gestionnaire commun sont

$$
\begin{aligned}
c_0&=\frac1{\Delta t_{n+1}}+
       \frac1{\Delta t_{n+1}+\Delta t_n},\\
c_1&=-\frac1{\Delta t_{n+1}}-\frac1{\Delta t_n},\\
c_2&=\frac{\Delta t_{n+1}}
           {\Delta t_n(\Delta t_{n+1}+\Delta t_n)}.
\end{aligned}
$$

À pas constant, ils valent \(3/(2\Delta t),-2/\Delta t,1/(2\Delta t)\).
Ils ne sont pas recopiés comme constantes dans l'assembleur.

La configuration est

~~~text
subsection Time integration
  set scheme = BDF2
  set bdf start method = BDF1
  set bdf start step ratio = 0.5
  set dt = 0.004
  set t_initial = 0
  set t_end = 0.04
end
~~~

Le premier pas réutilise exactement le schéma BDF1 de
[equations.md](equations.md), avec \(c_0=1/\Delta t_1\),
\(c_1=-1/\Delta t_1\), \(c_2=0\). Son inertie et sa différence divisée du
potentiel restent celles de BDF1. La sélection effective se fait par le
troisième coefficient, qui est nul au démarrage.

Avec le rapport \(r\), FEZ prend d'abord \(r\Delta t\), puis
\((1-r)\Delta t\), puis le pas nominal. Le rapport doit vérifier
\(0<r<1\). Avec \(r=0.5\), les deux premiers pas valent donc
\(\Delta t/2\). Le défaut commun de FEZ reste \(r=0.1\) si rien n'est indiqué.
La reprise conserve les deux états et les longueurs des pas acceptés,
y compris si le nouveau pas demandé diffère de l'ancien.

Un démarrage BDF1 donne des erreurs d'ordre deux sur les variables
évolutives après ce premier pas. En revanche, sa différence divisée donne
en général une erreur d'ordre un sur le potentiel chimique au premier
instant. L'ordre deux est vérifié au temps final après passage à BDF2 ;
il ne faut pas interpréter un maximum temporel incluant le potentiel de
ce premier pas comme une mesure générale d'ordre deux.
L'ordre suppose aussi des champs suffisamment réguliers : traverser un
point anguleux du prolongement ou de la mobilité peut le dégrader.

## Système fort discrétisé en temps

Aux pas BDF2, l'inertie remplace la dérivée dans l'identité continue

$$
\widetilde\rho\,\partial_t\boldsymbol v
 +\frac12\boldsymbol v\,\partial_t\widetilde\rho
=\sqrt{\widetilde\rho}\,
  \partial_t(\sqrt{\widetilde\rho}\boldsymbol v).
$$

Les équations résolues sont

$$
\begin{aligned}
&\sqrt{\widetilde\rho^{n+1}}\left[
 c_0\sqrt{\widetilde\rho^{n+1}}\boldsymbol v^{n+1}
 +c_1\sqrt{\widetilde\rho^n}\boldsymbol v^n
 +c_2\sqrt{\widetilde\rho^{n-1}}\boldsymbol v^{n-1}\right]\\
&\quad+\rho(\phi^{n+1})(\boldsymbol v^{n+1}\cdot\nabla)\boldsymbol v^{n+1}
 +\frac12\boldsymbol v^{n+1}
       \nabla\cdot(\rho(\phi^{n+1})\boldsymbol v^{n+1})\\
&\quad-\nabla\cdot\left[\widetilde\eta(\phi^{n+1})
 \left(\nabla\boldsymbol v^{n+1}+\nabla\boldsymbol v^{n+1,T}
       -\frac2d(\nabla\cdot\boldsymbol v^{n+1})\boldsymbol I\right)\right]
 +\nabla p^{n+1}+\phi^{n+1}\nabla\mu^{n+1}
 =\rho(\phi^{n+1})\boldsymbol b^{n+1},\\
&\nabla\cdot\boldsymbol v^{n+1}
 -\alpha\nabla\cdot\left[m(\phi^{n+1})
      (\nabla\mu^{n+1}+\alpha\nabla p^{n+1})\right]=0,\\
&c_0\phi^{n+1}+c_1\phi^n+c_2\phi^{n-1}
 +\nabla\cdot(\phi^{n+1}\boldsymbol v^{n+1})\\
&\quad-\nabla\cdot\left[m(\phi^{n+1})
      (\nabla\mu^{n+1}+\alpha\nabla p^{n+1})\right]=0,\\
&\mu^{n+1}
 -\frac{3\sigma}{2\sqrt2\epsilon}
       \left((\phi^{n+1})^3-\phi^{n+1}\right)
 +\frac{3\sigma\epsilon}{2\sqrt2}\Delta\phi^{n+1}=0.
\end{aligned}
$$

Ici \(\widetilde\rho^j=\widetilde\rho(\phi^j)\). Aucun champ
\(\sqrt{\widetilde\rho}\boldsymbol v\) supplémentaire n'est résolu :
ces produits sont évalués aux points de quadrature.

## Formulation faible discrète

Les espaces sont ceux de Taylor–Hood pour vitesse/pression et des degrés
égaux pour phase/potentiel : par exemple \(P_2/P_1/P_1/P_1\) sur triangles
ou \(Q_2/Q_1/Q_1/Q_1\) sur quadrilatères. Les tests
\(\boldsymbol z,q,\psi,\chi\) satisfont les contraintes homogènes.
Toutes les intégrales portent sur \(\Omega\). La résolution est monolithique.

Pour tous ces tests, chercher les quatre champs au nouveau pas tels que

$$
\begin{aligned}
0={}&\int_\Omega\sqrt{\widetilde\rho^{n+1}}\left[
 c_0\sqrt{\widetilde\rho^{n+1}}\boldsymbol v^{n+1}
 +c_1\sqrt{\widetilde\rho^n}\boldsymbol v^n
 +c_2\sqrt{\widetilde\rho^{n-1}}\boldsymbol v^{n-1}\right]\cdot\boldsymbol z\\
&+\frac12\int_\Omega\rho(\phi^{n+1})\left[
 ((\nabla\boldsymbol v^{n+1})\boldsymbol v^{n+1})\cdot\boldsymbol z
 -((\nabla\boldsymbol z)\boldsymbol v^{n+1})\cdot\boldsymbol v^{n+1}\right]\\
&+\int_\Omega\widetilde\eta(\phi^{n+1})
 \left(\nabla\boldsymbol v^{n+1}+\nabla\boldsymbol v^{n+1,T}
 -\frac2d(\nabla\cdot\boldsymbol v^{n+1})\boldsymbol I\right):\nabla\boldsymbol z\\
&-\int_\Omega p^{n+1}\nabla\cdot\boldsymbol z
 +\int_\Omega\phi^{n+1}\nabla\mu^{n+1}\cdot\boldsymbol z
 -\int_\Omega\rho(\phi^{n+1})\boldsymbol b^{n+1}\cdot\boldsymbol z ,
\end{aligned}
$$

$$
0=-\int_\Omega q\,\nabla\cdot\boldsymbol v^{n+1}
  -\alpha\int_\Omega m(\phi^{n+1})
       (\nabla\mu^{n+1}+\alpha\nabla p^{n+1})\cdot\nabla q ,
$$

$$
\begin{aligned}
0={}&\int_\Omega(c_0\phi^{n+1}+c_1\phi^n+c_2\phi^{n-1})\psi
 -\int_\Omega\phi^{n+1}\boldsymbol v^{n+1}\cdot\nabla\psi\\
 &+\int_\Omega m(\phi^{n+1})
     (\nabla\mu^{n+1}+\alpha\nabla p^{n+1})\cdot\nabla\psi ,
\end{aligned}
$$

$$
0=\int_\Omega\left[\mu^{n+1}
 -\frac{3\sigma}{2\sqrt2\epsilon}
       ((\phi^{n+1})^3-\phi^{n+1})\right]\chi
 -\frac{3\sigma\epsilon}{2\sqrt2}
       \int_\Omega\nabla\phi^{n+1}\cdot\nabla\chi .
$$

Le signe négatif du résidu de pression suit la convention de Stokes du
solveur. Les sources fabriquées sont ajoutées dans les mêmes résidus
que pour BDF1 ; elles sont dérivées du système continu.

## Jacobien analytique

Les variations des inconnues sont
\(\delta\boldsymbol v,\delta p,\delta\phi,\delta\mu\).
Les historiques, les tests, la force et les coefficients temporels sont fixes.
Pour alléger les exposants dans cette section seulement, les champs sans
exposant désignent les valeurs \(n+1\).

Les dérivées constitutives sont

$$
\rho'=\frac{\rho_0-\rho_1}{2},\quad
\widetilde\rho'=\frac{\rho_0-\rho_1}{2}
          \frac{d}{d\phi}\max(-1,\min(1,\phi)),\quad
\widetilde\eta'=\frac{\eta_0-\eta_1}{2}
          \frac{d}{d\phi}\max(-1,\min(1,\phi)).
$$

La dérivée du flux dans chacun des deux résidus scalaires est écrite
explicitement dans les formules ci-dessous. Pour la loi absolue,

$$
m'(\phi)=-2m_0\epsilon^2\phi\,\operatorname{sign}(1-\phi^2)
$$

hors des points anguleux. À ceux-ci et aux extrémités du prolongement,
le code prend les dérivées généralisées des lois constitutives existantes.
Les contrôles classiques par différences finies sont faits hors de ces points.

La variation du résidu de quantité de mouvement est

$$
\begin{aligned}
\delta R_{\boldsymbol v}={}&
\int_\Omega c_0\widetilde\rho\,\delta\boldsymbol v\cdot\boldsymbol z\\
&+\int_\Omega\widetilde\rho'\delta\phi\left[
 c_0\boldsymbol v+
 \frac{c_1\sqrt{\widetilde\rho^n}\boldsymbol v^n+
       c_2\sqrt{\widetilde\rho^{n-1}}\boldsymbol v^{n-1}}
      {2\sqrt{\widetilde\rho}}\right]\cdot\boldsymbol z\\
&+\frac12\int_\Omega\rho'\delta\phi\left[
 ((\nabla\boldsymbol v)\boldsymbol v)\cdot\boldsymbol z
 -((\nabla\boldsymbol z)\boldsymbol v)\cdot\boldsymbol v\right]\\
&+\frac12\int_\Omega\rho\left[
 ((\nabla\delta\boldsymbol v)\boldsymbol v+
      (\nabla\boldsymbol v)\delta\boldsymbol v)\cdot\boldsymbol z
 -((\nabla\boldsymbol z)\delta\boldsymbol v)\cdot\boldsymbol v
 -((\nabla\boldsymbol z)\boldsymbol v)\cdot\delta\boldsymbol v\right]\\
&+\int_\Omega\widetilde\eta'\delta\phi
 \left(\nabla\boldsymbol v+\nabla\boldsymbol v^T
       -\frac2d(\nabla\cdot\boldsymbol v)\boldsymbol I\right):\nabla\boldsymbol z\\
&+\int_\Omega\widetilde\eta
 \left(\nabla\delta\boldsymbol v+\nabla\delta\boldsymbol v^T
       -\frac2d(\nabla\cdot\delta\boldsymbol v)\boldsymbol I\right):\nabla\boldsymbol z\\
&-\int_\Omega\delta p\,\nabla\cdot\boldsymbol z
 +\int_\Omega(\delta\phi\nabla\mu+\phi\nabla\delta\mu)\cdot\boldsymbol z
 -\int_\Omega\rho'\delta\phi\,\boldsymbol b\cdot\boldsymbol z .
\end{aligned}
$$

Les trois autres variations sont

$$
\begin{aligned}
\delta R_p={}&-\int_\Omega q\,\nabla\cdot\delta\boldsymbol v\\
 &-\alpha\int_\Omega\left[
 m'(\phi)\delta\phi(\nabla\mu+\alpha\nabla p)
 +m(\phi)(\nabla\delta\mu+\alpha\nabla\delta p)\right]\cdot\nabla q ,
\end{aligned}
$$

$$
\begin{aligned}
\delta R_\phi={}&\int_\Omega c_0\delta\phi\,\psi
 -\int_\Omega(\delta\phi\boldsymbol v+\phi\delta\boldsymbol v)\cdot\nabla\psi\\
 &+\int_\Omega\left[
 m'(\phi)\delta\phi(\nabla\mu+\alpha\nabla p)
 +m(\phi)(\nabla\delta\mu+\alpha\nabla\delta p)\right]\cdot\nabla\psi ,
\end{aligned}
$$

$$
\delta R_\mu=\int_\Omega\left[
 \delta\mu-\frac{3\sigma}{2\sqrt2\epsilon}(3\phi^2-1)\delta\phi\right]\chi
 -\frac{3\sigma\epsilon}{2\sqrt2}
       \int_\Omega\nabla\delta\phi\cdot\nabla\chi .
$$

Le résidu est assemblé avec le signe opposé au second membre de Newton,
comme dans les autres assembleurs de FEZ.

## Conservation et diagnostics énergétiques

Sans sources, sur domaine fermé, les deux équations scalaires impliquent
au sens faible la conservation de la densité affine :

$$
\begin{aligned}
&\int_\Omega[c_0\rho(\phi^{n+1})+c_1\rho(\phi^n)+c_2\rho(\phi^{n-1})]a\\
&\qquad-\int_\Omega\rho(\phi^{n+1})\boldsymbol v^{n+1}\cdot\nabla a=0.
\end{aligned}
$$

Le test constant conserve la masse lorsque les historiques sont cohérents ;
le démarrage BDF1 établit cette cohérence. La masse prolongée n'a pas de
garantie de conservation.

L'énergie physique diagnostiquée reste

$$
\int_\Omega\left[
\frac12\widetilde\rho|\boldsymbol v|^2+
\frac{3\sigma}{8\sqrt2\epsilon}(\phi^2-1)^2+
\frac{3\sigma\epsilon}{4\sqrt2}|\nabla\phi|^2+
\rho(-\boldsymbol b\cdot\boldsymbol x)\right].
$$

Le produit de l'inertie par la vitesse et les tests des équations scalaires
donnent le bilan temporel assemblé

$$
\begin{aligned}
0={}&\int_\Omega\sqrt{\widetilde\rho^{n+1}}\boldsymbol v^{n+1}\cdot
 \left[c_0\sqrt{\widetilde\rho^{n+1}}\boldsymbol v^{n+1}
      +c_1\sqrt{\widetilde\rho^n}\boldsymbol v^n
      +c_2\sqrt{\widetilde\rho^{n-1}}\boldsymbol v^{n-1}\right]\\
&+\frac{3\sigma}{2\sqrt2\epsilon}\int_\Omega
 ((\phi^{n+1})^3-\phi^{n+1})
 (c_0\phi^{n+1}+c_1\phi^n+c_2\phi^{n-1})\\
&+\frac{3\sigma\epsilon}{2\sqrt2}\int_\Omega
 \nabla\phi^{n+1}\cdot
 (c_0\nabla\phi^{n+1}+c_1\nabla\phi^n+c_2\nabla\phi^{n-1})\\
&+\int_\Omega(-\boldsymbol b\cdot\boldsymbol x)
       [c_0\rho(\phi^{n+1})+c_1\rho(\phi^n)+c_2\rho(\phi^{n-1})]\\
&+\int_\Omega 2\widetilde\eta(\phi^{n+1})
 \left|\frac{\nabla\boldsymbol v^{n+1}+\nabla\boldsymbol v^{n+1,T}}2
       -\frac{\nabla\cdot\boldsymbol v^{n+1}}d\boldsymbol I\right|^2\\
&+\int_\Omega m(\phi^{n+1})
       |\nabla\mu^{n+1}+\alpha\nabla p^{n+1}|^2 .
\end{aligned}
$$

Ce bilan est celui des équations testées, à la précision de quadrature et
de résolution. La force doit être constante et conservative et son potentiel
doit appartenir à l'espace de test pour la contribution gravitationnelle
écrite ici. Les sources et les traces imposées ajoutent leur travail.

À pas constant, l'énergie cinétique modifiée est

$$
\frac14\int_\Omega\left[
 \widetilde\rho^{n+1}|\boldsymbol v^{n+1}|^2+
 |2\sqrt{\widetilde\rho^{n+1}}\boldsymbol v^{n+1}
       -\sqrt{\widetilde\rho^n}\boldsymbol v^n|^2\right].
$$

Sa variation, augmentée du reste positif

$$
\frac14\int_\Omega
 |\sqrt{\widetilde\rho^{n+1}}\boldsymbol v^{n+1}
 -2\sqrt{\widetilde\rho^n}\boldsymbol v^n
 +\sqrt{\widetilde\rho^{n-1}}\boldsymbol v^{n-1}|^2,
$$

égale le travail temporel cinétique multiplié par \(\Delta t\).
L'identité de gradient suit avec les mêmes coefficients, en remplaçant les
vitesses pondérées par les gradients de phase et en multipliant par
\(3\sigma\epsilon/(2\sqrt2)\).
La contribution cubique n'a pas ici d'inégalité complète établie qui donne
la décroissance inconditionnelle d'une énergie totale modifiée.

Le fichier de diagnostics distingue donc :

- effective_bdf_order et bdf_c0/bdf_c1/bdf_c2 : schéma réellement utilisé ;
- temporal_energy_balance_residual : somme des quatre travaux temporels
  et des dissipations, valable pour les équations testées des deux schémas ;
- physical_energy_increment_plus_dissipation : variation de l'énergie
  physique augmentée de \(\Delta t\) fois la dissipation au nouvel instant,
  sans assertion de signe ou de nullité en BDF2 ;
- bdf2_modified_kinetic_energy et bdf2_modified_gradient_energy :
  énergies des deux niveaux temporels ;
- bdf2_constant_step : indique si l'identité classique BDF2 s'applique ;
- bdf2_kinetic_remainder, bdf2_gradient_remainder et leurs identity_residual :
  restes positifs et fermeture des identités à pas constant seulement.

Les pertes numériques et les deux anciens résidus énergétiques propres à
BDF1 valent NaN aux pas BDF2. Les identités classiques à pas constant valent
également NaN aux pas variables et au démarrage. Ces valeurs signalent
une formule non applicable, et non une défaillance numérique.

## Vérifications courtes

Depuis la racine du dépôt :

~~~sh
cmake --build buildRelease --target incompressible_chns -j 4
ctest --test-dir buildRelease --output-on-failure -R quasi_incompressible
~~~

Pour ne lancer que la suite BDF2 :

~~~sh
python3 tests/incompressible_chns/validate_quasi_incompressible.py \
  --solver ./buildRelease/incompressible_chns \
  --workdir ./buildRelease/verification_quasi_bdf2 \
  --suite bdf2
~~~

Elle conserve les paramètres générés, les journaux et les diagnostics.
Elle contrôle les jacobiens assemblés, la conservation, les phases pures,
les interfaces, le démarrage et la reprise avec pas identique ou modifié.
La solution fabriquée affine en espace vérifie l'ordre temporel au temps
final sur les quatre champs, aux rapports 1:1, 1000:1 et 1:1000.
La suite MPI séparée couvre également BDF2. Les cas de bulles sont laissés
au lancement manuel par l'utilisateur ; aucune suite automatique ne les lance.
