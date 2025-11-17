# tsp_graph_init.py

import random
import math
import csv
import tkinter as tk

LARGEUR = 800
HAUTEUR = 600
NB_LIEUX = 10  # à adapter


class Lieu:
    def __init__(self, x: float, y: float, nom: str | int):
        self.x = x
        self.y = y
        self.nom = nom

    def distance(self, autre: "Lieu") -> float:
        dx = self.x - autre.x
        dy = self.y - autre.y
        return math.sqrt(dx * dx + dy * dy)

    def __repr__(self):
        return f"Lieu({self.nom}, x={self.x}, y={self.y})"


class Graph:
    def __init__(self):
        self.liste_lieux: list[Lieu] = []
        self.matrice_od: list[list[float]] = []

    def generer_lieux_aleatoires(self, nb_lieux: int = NB_LIEUX):
        self.liste_lieux = []
        for i in range(nb_lieux):
            x = random.uniform(20, LARGEUR - 20)
            y = random.uniform(20, HAUTEUR - 20)
            self.liste_lieux.append(Lieu(x, y, i))

    def charger_graph(self, chemin_csv: str):
        """Lit un fichier CSV avec des coordonnées x,y par ligne."""
        self.liste_lieux = []
        with open(chemin_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            # adapter si le fichier a un header
            for i, row in enumerate(reader):
                if not row:
                    continue
                x = float(row[0])
                y = float(row[1])
                self.liste_lieux.append(Lieu(x, y, i))

    def calcul_matrice_cout_od(self):
        n = len(self.liste_lieux)
        self.matrice_od = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    self.matrice_od[i][j] = 0.0
                else:
                    self.matrice_od[i][j] = self.liste_lieux[i].distance(
                        self.liste_lieux[j]
                    )

    def plus_proche_voisin(self, indice_lieu: int, indices_non_visites: set[int]) -> int | None:
        """Retourne l'indice du plus proche voisin parmi indices_non_visites."""
        if not indices_non_visites:
            return None
        meilleur_indice = None
        meilleure_distance = float("inf")
        for j in indices_non_visites:
            d = self.matrice_od[indice_lieu][j]
            if d < meilleure_distance:
                meilleure_distance = d
                meilleur_indice = j
        return meilleur_indice

    def calcul_distance_route(self, route: "Route") -> float:
        distance_totale = 0.0
        for i in range(len(route.ordre) - 1):
            a = route.ordre[i]
            b = route.ordre[i + 1]
            distance_totale += self.matrice_od[a][b]
        return distance_totale


class Route:
    def __init__(self, ordre: list[int]):
        # on s'assure que la route commence et finit par 0
        if ordre[0] != 0:
            ordre = [0] + ordre
        if ordre[-1] != 0:
            ordre = ordre + [0]
        self.ordre = ordre

    @classmethod
    def route_random(cls, nb_lieux: int) -> "Route":
        indices = list(range(1, nb_lieux))  # on ne mélange pas 0
        random.shuffle(indices)
        ordre = [0] + indices + [0]
        return cls(ordre)

    def __repr__(self):
        return f"Route({self.ordre})"


class Affichage:
    def __init__(self, graph: Graph, meilleures_routes: list[Route] | None = None, groupe: str = "Mon Groupe"):
        self.graph = graph
        self.meilleures_routes = meilleures_routes or []

        self.root = tk.Tk()
        self.root.title(f"TSP - Groupe {groupe}")

        self.canvas = tk.Canvas(self.root, width=LARGEUR, height=HAUTEUR, bg="white")
        self.canvas.pack()

        self.zone_texte = tk.Text(self.root, height=5)
        self.zone_texte.pack(fill="x")

        # binding des touches
        self.root.bind("<Escape>", self.quitter)
        self.root.bind("r", self.afficher_routes_grises)
        self.root.bind("m", self.afficher_matrice_couts)

        self.dessiner_lieux()

    def dessiner_lieux(self):
        rayon = 8
        for lieu in self.graph.liste_lieux:
            x, y = lieu.x, lieu.y
            self.canvas.create_oval(x - rayon, y - rayon, x + rayon, y + rayon)
            # numéro au milieu
            self.canvas.create_text(x, y, text=str(lieu.nom))

    def dessiner_route(self, route: Route, couleur="blue", pointille=True):
        points = []
        for idx in route.ordre:
            lieu = self.graph.liste_lieux[idx]
            points.append((lieu.x, lieu.y))

        dash = (4, 2) if pointille else None

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            self.canvas.create_line(x1, y1, x2, y2, dash=dash, fill=couleur)

        # numéro d'ordre au-dessus de chaque lieu
        for ordre_visite, idx in enumerate(route.ordre):
            lieu = self.graph.liste_lieux[idx]
            self.canvas.create_text(lieu.x, lieu.y - 15, text=str(ordre_visite))

    def afficher_message(self, texte: str):
        self.zone_texte.insert(tk.END, texte + "\n")
        self.zone_texte.see(tk.END)

    def afficher_routes_grises(self, event=None, N: int = 5):
        self.afficher_message("Affichage des meilleures routes en gris...")
        for route in self.meilleures_routes[:N]:
            self.dessiner_route(route, couleur="grey", pointille=False)

    def afficher_matrice_couts(self, event=None):
        self.afficher_message("Matrice des coûts :")
        for ligne in self.graph.matrice_od:
            self.afficher_message("  " + "  ".join(f"{d:.1f}" for d in ligne))

    def quitter(self, event=None):
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Exemple d'utilisation simple pour tester
    g = Graph()
    g.generer_lieux_aleatoires(NB_LIEUX)
    g.calcul_matrice_cout_od()

    route_random = Route.route_random(len(g.liste_lieux))
    print("Route aléatoire :", route_random)
    print("Distance :", g.calcul_distance_route(route_random))

    aff = Affichage(g, meilleures_routes=[route_random], groupe="ISEN")
    aff.dessiner_route(route_random)
    aff.afficher_message(f"Distance route aléatoire : {g.calcul_distance_route(route_random):.2f}")
    aff.run()
