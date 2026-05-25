# Analyse de sons d'oiseaux en notes

Ce projet contient un script Python qui extrait des notes approximatives depuis un fichier audio d'oiseau, par exemple `sound/canard.mp3`.

Le script produit:

- `analysis/<nom>.notes.json`: timecodes, note, frequence et confiance.
- `analysis/<nom>.notes.csv`: le meme contenu pour tableur.
- `analysis/<nom>.html`: une page de lecture avec audio et portee synchronisee.

## Installation

```powershell
python -m pip install -r requirements.txt
```

Pour les MP3, Python doit pouvoir decoder l'audio. Si le chargement echoue, installez FFmpeg et ajoutez-le au `PATH`.

## Utilisation

```powershell
python scripts/analyze_bird_notes.py sound/canard.mp3
```

Ouvrez ensuite `analysis/canard.html` dans le navigateur.

Par defaut, la frise HTML pioche une image aleatoire par evenement dans un dossier qui porte le meme nom que l'audio:

- `sound/canard.mp3` utilise `img/canard/`.
- `sound/merle.mp3` utilise `img/merle/`.

La position horizontale suit le timecode, et la hauteur est pseudo-aleatoire pour produire une frise visuelle.

Pour utiliser une autre image ou un autre dossier:

```powershell
python scripts/analyze_bird_notes.py sound/canard.mp3 --note-image "img/canard"
```

Pour un chant plus aigu ou plus grave, ajustez la plage detectee:

```powershell
python scripts/analyze_bird_notes.py sound/merle.mp3 --fmin C4 --fmax C8
```

## Reglages utiles

- `--min-note-duration`: ignore les notes tres courtes.
- `--min-voiced-probability`: augmentez pour reduire les fausses notes, baissez pour detecter plus de sons.
- `--note-change-semitones`: augmentez pour fusionner des variations de hauteur proches.
- `--fmin` et `--fmax`: plage musicale analysee.

L'analyse d'un cri d'oiseau reste approximative: beaucoup de sons d'oiseaux sont bruitistes, glissants ou polyphoniques. Le JSON exporte donc une transcription jouable et synchronisable, pas une partition parfaite.

Si `notes` est vide, cela signifie que l'algorithme n'a pas trouve de hauteur stable. C'est courant pour un cri de canard. Le script inclut maintenant un mode de secours: il detecte les evenements sonores par energie et leur attribue une note approximative via le centre spectral. Ces notes ont `"method": "energy_centroid"` et une confiance basse (`0.25`), car elles servent surtout a synchroniser l'affichage.



----


- Higlight de la couleur de la note ? actuellement figé à rouge
- Auto scroll / défilement
- hide les notes
- assouplir la détections des notes ? merle ? parametrable ? 
- fichier de configuration par oiseau. Ratio/Scaling des notes, ...