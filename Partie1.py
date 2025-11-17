import csv
import random
import numpy as np

LARGEUR = 800
HAUTEUR = 600
NB_LIEUX = 1000


class Lieu:
    def __init__(self, nom, x, y):
        self.nom = nom
        self.x = x
        self.y = y
        self.nbr_route = 0

    def distance(self, autre_lieu):
        return ((self.x - autre_lieu.x) ** 2 + (self.y - autre_lieu.y) ** 2) ** 0.5

    def __repr__(self):
        return f"({self.nom},{self.x},{self.y})"


class Graph:
    def __init__(self):
        self.liste_lieux = []
        self.matrice_od = None
        self.distances = None
        self.route_numerotee = []

    def generer_lieux_aleatoires(self, nb_lieux=NB_LIEUX):
        global NB_LIEUX
        NB_LIEUX = nb_lieux
        self.liste_lieux = [
            Lieu(f"L{i}", random.uniform(0, LARGEUR), random.uniform(0, HAUTEUR))
            for i in range(nb_lieux)
        ]
        self.calcul_matrice_cout_od()

    def charger_graph(self, chemin_fichier):
        global NB_LIEUX
        self.liste_lieux = []
        with open(chemin_fichier, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                x = float(row['x'])
                y = float(row['y'])
                self.liste_lieux.append(Lieu(f"L{i}", x, y))
        NB_LIEUX = len(self.liste_lieux)
        self.calcul_matrice_cout_od()

    def calcul_matrice_cout_od(self):
        n = len(self.liste_lieux)
        self.matrice_od = np.zeros((n, n))

        for i in range(n - 1):
            self.matrice_od[i, i] = np.inf
            for j in range(i + 1, n):
                d = self.liste_lieux[i].distance(self.liste_lieux[j])
                self.matrice_od[i, j] = d
                self.matrice_od[j, i] = d

        self.matrice_od[n - 1, n - 1] = np.inf
        self.distances = self.matrice_od.copy()

    def plus_proche_voisin(self, nom_lieu, lieu_fait):
        self.distances[:, lieu_fait] = np.inf

        i = int(nom_lieu[1:])
        j = np.argmin(self.distances[i])
        return f"L{j}"

    def calcul_distance_route(self, route):
        distance_totale = 0.0
        for i in range(len(route) - 1):
            a = int(route[i][1:])
            b = int(route[i + 1][1:])
            distance_totale += self.matrice_od[a, b]
        return distance_totale

    def heuristique_voisin(self):
        self.route_numerotee = []
        self.route_numerotee.append("L0")
        lieu_fait = 0
        numero = 0

        n = len(self.liste_lieux)
        for _ in range(n - 1):
            voisin = self.plus_proche_voisin(f"L{numero}", lieu_fait)
            lieu_fait = int(voisin[1:])
            self.route_numerotee.append(voisin)
            numero = lieu_fait

        # retour au point de départ
        self.route_numerotee.append("L0")

    def get_route_numerotee(self):
        return self.route_numerotee


class Route:
    def __init__(self, classe_graph):
        self.graph = classe_graph
        self.ordre = classe_graph.get_route_numerotee()
        self.distance = 0.0

    def __repr__(self):
        return f"Route: {' -> '.join(str(i) for i in self.ordre)}"

    def distance_totale(self):
        self.distance = self.graph.calcul_distance_route(self.ordre)
        return self.distance



if __name__ == "__main__":
    graph_test = Graph()
    graph_test.generer_lieux_aleatoires(20)
    graph_test.heuristique_voisin()

    route1 = Route(graph_test)
    dist = route1.distance_totale()
    print(route1)
    print("Distance totale :", dist)
