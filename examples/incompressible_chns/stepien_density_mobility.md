# Mobilité Stepien compensée par la densité

Dans la sous-section `Cahn Hilliard` du fichier de simulation, activer :

```text
subsection Cahn Hilliard
  set CHNS model = stepien
  set mobility model = constant
  set mobility = 1e-6
  set enable density mobility compensation = true
end
```

Le paramètre `mobility` donne alors l'amplitude M0 de la loi

    M(phi) = M0 * (rho0*rho1)^2 / rho(phi)^2.

Les densités sont lues automatiquement dans `Physical properties / Fluid 0`
et `Fluid 1`. La densité de mélange est celle de l'équation de phase :

    rho(phi) = ((1+phi)*rho0 + (1-phi)*rho1)/2.

L'option est désactivée par défaut. Elle multiplie la mobilité de base sans
modifier les facteurs de densité présents dans l'équation de Stepien. Le
coefficient du terme principal en Laplacien du potentiel chimique, après
normalisation par le coefficient de la dérivée matérielle de phi, vaut M0.
Les autres contributions du flux et les gradients de densité restent présents.
Avec la compensation activée, M0 désigne cette amplitude normalisée ; ses
unités sont celles de M divisé par une densité au carré.

Pour combiner compensation et mobilité dégénérée :

```text
subsection Cahn Hilliard
  set CHNS model = stepien
  set mobility model = degenerate
  set enable density mobility compensation = true
  set enable mobility tracer limiter = false
  subsection degenerate mobility
    set Function expression = 1e-6*(1-x*x)*(1-x*x)
  end
end
```

Ici `x` représente phi et l'expression donne toute la mobilité de base, y
compris M0. Le paramètre `mobility` ne multiplie pas cette expression.
La mobilité obtenue est

    M(phi) = M0*(1-phi^2)^2 * (rho0*rho1)^2 / rho(phi)^2.

Les dérivées première et seconde de la mobilité compensée sont calculées.
Le Jacobien de phase utilise la dérivée première.
La compensation utilise le même limiteur de propriétés que rho si
`enable tracer limiter = true`. Le limiteur de mobilité agit seulement sur
l'argument de la mobilité de base. Les dérivées tiennent compte de ces deux
limiteurs ; aux coins phi = +/-1, la dérivée du limiteur est choisie depuis
l'intérieur de l'intervalle.

Cette option est disponible pour Stepien sur maillage fixe. Les densités des
fluides et du mélange doivent être strictement positives ; aucune borne
artificielle n'est ajoutée au dénominateur. Les chemins d'assemblage actuels
de la sous-section `Stabilization` supposent une mobilité constante :
`enable supg` et `enable tracer supg` doivent donc rester `false` lorsque
la compensation est activée.
