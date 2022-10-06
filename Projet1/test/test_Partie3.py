#les codes pour tester les fonctions dans le repertoire model (partie3)
from model import *
from strategy import *
alea=Alea_str()
print(alea.jouer())
#alea.affiche_stat(1000)

heur=Heur_str()
print(heur.jouer())
heur.affiche_stat(1000)
