# Mobilité compensée en densité

L'option `enable quasi incompressible density mobility compensation` applique
la loi archivée au modèle `quasi_incompressible` :

\[
M_c(\phi)=M_b(\phi)\min\left[
 \left(\frac{\bar\rho}{\widetilde\rho(\phi)}\right)^2,L\right],\qquad
\bar\rho=\frac{\rho_0+\rho_1}{2},\qquad
\widetilde\rho(\phi)=\rho(\min(1,\max(-1,\phi))).
\]

Pour la base absolue, le bloc suivant conserve la phase brute dans
\(M_b=m_0\varepsilon^2|1-\phi^2|\). Seule la densité du multiplicateur est
écrêtée. Le plafond borne le multiplicateur, pas la mobilité entière.

```text
subsection Cahn Hilliard
  set CHNS model = quasi_incompressible
  set mobility model = degenerate_absolute
  set degenerate absolute mobility prefactor = 0.1
  set interface thickness = 0.00781
  set enable mobility tracer limiter = false
  set enable density mobility compensation = false
  set enable quasi incompressible density mobility compensation = true
  set quasi incompressible density mobility compensation maximum = 1000
  set enable profile correction = true
  set profile correction strength = 0.6
end
```

Avec `Fluid 0/density = 1000` et `Fluid 1/density = 1`, la phase +1 désigne
le liquide et la phase −1 le gaz. Le préfacteur de mobilité est
\(0.1(0.00781)^2=6.09961\times10^{-6}\). Utiliser la même épaisseur dans
le profil initial. Les autres paramètres (tension superficielle, maillage,
pas de temps et conditions aux limites) restent à définir dans le cas.

La compensation est désactivée par défaut. Son plafond vaut 1000 par défaut
et doit être fini et strictement positif. L'ancienne option
`enable density mobility compensation` est réservée à Stepien et doit rester
désactivée ici.

## Correction de profil et bilans

Les deux options peuvent être activées ensemble ou séparément. La correction
reconstruite reste additive : le flux partagé par les équations de phase et
de continuité est

\[
K=M_c(\phi_h)(\nabla\mu_h+\alpha\nabla p_h)
 + A_{\mathrm{corr}}F_\beta[\phi_{r,h}].
\]

Pour la base absolue, l'amplitude constante reste
\(A_{\mathrm{corr}}=2\lambda(\widetilde\sigma/\varepsilon)
m_0\varepsilon^2\). Le multiplicateur compensé et son plafond ne multiplient
pas le flux correctif. La compensation est aussi utilisée dans les sources
manufacturées et les diagnostics, via le même évaluateur de mobilité.

Les dérivées première et seconde tiennent compte du multiplicateur. Dans la
région non plafonnée, avec \(C=(\bar\rho/\widetilde\rho)^2\) et
\(\ell=\widetilde\rho'/\widetilde\rho\), elles sont

\[
M_c'=C(M_b'-2\ell M_b),\qquad
M_c''=C(M_b''-4\ell M_b'+6\ell^2M_b).
\]

Dans la région plafonnée et hors de [−1,1], le multiplicateur est constant.
Au seuil du plafond, le code choisit la dérivée du côté plafonné. Les
conventions de la base absolue aux phases pures sont conservées.

La densité affine reste celle des équations de conservation. La compensation
ne garantit pas sa positivité. Avec correction, le bilan énergétique inclut
`profile_correction_work` ; sa monotonie n'est pas garantie. Les vérifications
utilisent `energy_balance_residual_with_correction` en BDF1 et
`temporal_energy_balance_residual_with_correction` en BDF2.

## Vérification

Depuis la racine du dépôt :

```sh
cmake --build buildRelease --target incompressible_chns -j 4
ctest --test-dir buildRelease --output-on-failure \
  -R '^unit_tests/(chns_density_mobility|quasi_incompressible)\.release$|^incompressible_chns/quasi_density_compensation$'
```

Le test unitaire couvre les valeurs, les dérivées, les dépassements, les
phases pures, les densités inversées et la compatibilité avec la correction.
La suite couplée vérifie les Jacobiennes assemblées et les bilans de phase,
masse et énergie en BDF1/BDF2, avec et sans correction. Chaque exécution
conserve ses entrées et journaux dans un sous-dossier `run_*` distinct.
Ces tests courts ne constituent pas une validation de bulle jusqu'à t=3.

Validation locale du 12 novembre 2026 : les 20 cas de la suite couplée passent
(8 vérifications de Jacobien assemblé et 12 relaxations fermées). L'erreur
absolue maximale du Jacobien est de 4.47e-8, pour un seuil de 2e-5. Avec
correction et plafond 1000, la dérive absolue de masse reste sous 1.5e-12 et
le défaut du bilan corrigé par pas, en unités d'énergie, sous 6.5e-15.
Les tests unitaires `chns_density_mobility` et `quasi_incompressible` passent
en 2D et 3D, ainsi que l'exécutable de vérification de la correction reconstruite.
