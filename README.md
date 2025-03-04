# Projet King and Courtesan

## Commande pour lancer notre IA

```shell
java -cp ./build/libs/TP1B_KC_player.jar:commons-cli-1.4.jar iialib.games.contest.Client -p 4536 -s localhost -c games.kac.MyChallenger
```

## Algorithme utilisé pour possibleMoves

La méthode `possibleMoves` est un élément crucial de notre IA pour le jeu "Roi et Courtesan". Elle est conçue pour
identifier tous les coups valides qu'un joueur peut effectuer à un moment donné dans le jeu. Cette méthode prend en
compte le rôle du joueur actuel et l'état actuel du plateau de jeu pour générer une liste de coups potentiels.

### Vue d'Ensemble de la Méthode

- **Entrée** : La méthode accepte un `KingAndCourtesanRole` indiquant le rôle du joueur actuel (soit Rouge, soit Bleu).
- **Sortie** : Elle renvoie une `ArrayList<KingAndCourtesanMove>` contenant tous les coups valides que le joueur peut
  exécuter.

### Fonctionnement

1. **Identifier les Pièces Possédées** : La méthode parcourt tout le plateau de jeu, identifiant les cases occupées par
   les pièces appartenant au joueur.

2. **Vérifier les Cases Environnantes** : Pour chaque pièce identifiée, elle examine les 8 cases environnantes dont les
   coordonnées sont valides pour déterminer si le déplacement vers l'une de ces cases constitue un coup valide, selon
   les règles du jeu.

3. **Génération de Coups** :
    - **Déplacement** : Vérifie si une case est vide, permettant ainsi un mouvement de base vers l'avant.
    - **Capture** : Valide si une pièce adverse occupe une case adjacente, permettant une capture.
    - **Échange Roi-Courtesan** : Prend en compte spécialement les coups impliquant le roi et un courtesan, vérifiant
      s'ils peuvent échanger leurs positions conformément aux règles du jeu.

4. **Validation** : Chaque coup potentiel est validé à l'aide de la méthode `isValidMove`, s'assurant qu'il respecte les
   règles du jeu telles que les directions de mouvement et les mécanismes de capture.

5. **Compilation des Résultats** : Les coups valides sont compilés dans une liste et retournés comme sortie de la
   méthode.

### Composants Clés

- **Tableau d'Aide** : Utilise un tableau d'aide pour vérifier systématiquement toutes les 8 directions autour d'une
  pièce.
- **Traduction des Coordonnées** : Convertit les coordonnées du plateau en coups actionnables.
- **Validation des Coups** : Repose sur `isValidMove` pour filtrer les coups illégaux, s'assurant que seules les options
  réalisables sont considérées.

### Algorithme de Vérification de la Direction "Vers l'Avant"

La vérification de la légalité des coups est essentielle pour garantir que les mouvements générés sont légaux selon les
règles du jeu. La méthode `isValidMove` joue un rôle clé dans ce processus en vérifiant la légalité de chaque mouvement.
Voici comment fonctionne le processus de validation, avec un accent particulier sur l'algorithme utilisé pour vérifier
la direction "Vers l'avant" des mouvements :

#### Aspects Clés de l'Algorithme :

- **Vérification de la Direction** : L'algorithme calcule les différences d'indices de lignes et de
  colonnes (`helper = ligne de départ - ligne cible `) et (`helper2 = colonne de départ - colonne cible`). Ces
  différences sont utilisées pour déterminer la direction du mouvement proposé.

- **Logique Spécifique au Rôle** :
    - Pour le joueur Rouge, "vers l'avant" signifie soit se déplacer tout droit en
      haut (`helper == -1 et helper2 == -1`) soit diagonalement en haut à gauche (`helper == -1 et helper2 == 0`) soit
      en haut à droite (`helper == 0 et helper2 == -1`).
    - Pour le joueur Bleu, "vers l'avant" signifie soit se déplacer tout droit en bas (`helper == 1 et helper2 == 1`)
      soit diagonalement en bas à sa gauche (`helper == 1 et helper2 == 0`) soit en bas à sa
      droite (`helper == 0 et helper2 == 1`).