from model import *
from strategy import *

heur=Heur_str()
"""heur.bat.g1.grille=[
	[0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0 ],
	[0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0 ],
	[0  ,0  ,0  ,4  ,4  ,4  ,0  ,0  ,0  ,0 ],
	[0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0 ],
	[0  ,0  ,1  ,0  ,0  ,0  ,0  ,0  ,0  ,0 ],
	[0  ,0  ,1  ,0  ,0  ,0  ,0  ,0  ,0  ,0 ],
	[0  ,0  ,1  ,0  ,0  ,0  ,0  ,0  ,0  ,0 ],
	[0  ,0  ,1  ,0  ,0  ,0  ,0  ,0  ,0  ,0 ],
	[0  ,0  ,1  ,0  ,3  ,3  ,3  ,0  ,5  ,5 ],
	[0  ,2  ,2  ,2  ,2  ,0  ,0  ,0  ,0  ,0 ]]"""
print(heur.bat.g1.grille)
print(heur.jouer())
print(heur.bat.g1.grille)
print(heur.bat.test)
print(heur.bat.count)