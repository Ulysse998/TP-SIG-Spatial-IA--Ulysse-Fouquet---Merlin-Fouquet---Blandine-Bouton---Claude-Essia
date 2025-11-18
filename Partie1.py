import csv
import random
import numpy as np
import tkinter as tk


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
    def __init__(self, classe_graph, ordre=None):
        """
        classe_graph : instance de Graph
        ordre : liste de noms de lieux ("L0", "L1", ..., "L0")
                Si None, on prend la route générée par heuristique_voisin().
        """
        self.graph = classe_graph

        if ordre is None:
            # Route issue de l’heuristique du plus proche voisin
            self.ordre = classe_graph.get_route_numerotee()
        else:
            # Route fournie (par l'algorithme des fourmis par exemple)
            self.ordre = ordre

        self.distance = 0.0

    def __repr__(self):
        return f"Route: {' -> '.join(str(i) for i in self.ordre)}"

    def distance_totale(self):
        self.distance = self.graph.calcul_distance_route(self.ordre)
        return self.distance


class TSP_ACO:
    """
    Classe implémentant l'algorithme des Colonies de Fourmis pour le TSP.
    Elle utilise :
    - Graph : pour les distances (matrice_od)
    - Route : pour représenter les circuits
    """

    def __init__(self, graph,
                 nb_fourmis=20,
                 nb_iterations=50,
                 alpha=1.0,
                 beta=3.0,
                 evaporation=0.3,
                 Q=100.0):
        """
        graph : instance de Graph déjà initialisée avec liste_lieux et matrice_od
        nb_fourmis : nombre de fourmis par itération
        nb_iterations : nombre total d'itérations de l'algorithme
        alpha : importance des phéromones
        beta : importance de l'heuristique (1 / distance)
        evaporation : taux d'évaporation des phéromones (rho)
        Q : quantité de phéromone déposée par une fourmi (constante)
        """
        self.graph = graph
        self.n = len(self.graph.liste_lieux)

        self.nb_fourmis = nb_fourmis
        self.nb_iterations = nb_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q

        # Matrice des phéromones, initialisée avec une valeur constante
        valeur_initiale = 1.0
        self.pheromones = np.full((self.n, self.n), valeur_initiale, dtype=float)
        np.fill_diagonal(self.pheromones, 0.0)  # pas de phéromone sur les boucles i->i

        # Meilleure route globale (indices) et sa longueur
        self.meilleure_route_indices = None
        self.meilleure_distance = float('inf')

    # ---------------------------------------------------
    # Méthode utilitaire : longueur d'une route d'indices
    # ---------------------------------------------------
    def longueur_route_indices(self, route_indices):
        """
        Calcule la longueur totale d'une route donnée sous forme de liste d'indices [0, 5, 2, ..., 0].
        """
        dist_totale = 0.0
        for i in range(len(route_indices) - 1):
            a = route_indices[i]
            b = route_indices[i + 1]
            dist_totale += self.graph.matrice_od[a, b]
        return dist_totale

    # ---------------------------------------------------
    # Choix de la prochaine ville pour une fourmi
    # ---------------------------------------------------
    def choisir_ville_suivante(self, ville_actuelle, non_visitees):
        """
        Choisit la prochaine ville parmi les non_visitees en utilisant :
        P(i->j) ~ (pheromones[i][j]^alpha) * (1/distance[i][j])^beta
        """
        if not non_visitees:
            return None

        # On convertit l'ensemble en liste pour itérer
        candidats = list(non_visitees)

        # On calcule les "attractivités" pour chaque candidat
        attractivites = []
        for j in candidats:
            tau = self.pheromones[ville_actuelle, j] ** self.alpha
            dist = self.graph.matrice_od[ville_actuelle, j]
            if dist > 0:
                eta = (1.0 / dist) ** self.beta
            else:
                eta = 0.0  # cas théorique, normalement pas de distance nulle entre 2 lieux distincts

            attractivites.append(tau * eta)

        somme = sum(attractivites)
        if somme == 0:
            # Si toutes les attractivités sont nulles, on choisit au hasard
            return random.choice(candidats)

        # On tire une ville au hasard selon la distribution de probabilité
        r = random.random()
        cumul = 0.0
        for j, attr in zip(candidats, attractivites):
            p = attr / somme
            cumul += p
            if r <= cumul:
                return j

        # Sécurité si les arrondis posent problème
        return candidats[-1]

    # ---------------------------------------------------
    # Construction d'une route pour UNE fourmi
    # ---------------------------------------------------
    def construire_route_indices(self):
        """
        Construit un circuit complet (tour) pour une fourmi.
        On impose :
        - départ = 0
        - retour à 0 à la fin
        """
        depart = 0
        route = [depart]

        # Ensemble des villes non visitées : 1..n-1
        non_visitees = set(range(1, self.n))
        ville_actuelle = depart

        while non_visitees:
            prochaine = self.choisir_ville_suivante(ville_actuelle, non_visitees)
            route.append(prochaine)
            non_visitees.remove(prochaine)
            ville_actuelle = prochaine

        # Retour au point de départ
        route.append(depart)
        return route

    # ---------------------------------------------------
    # Lancement de l'algorithme ACO
    # ---------------------------------------------------
    def run(self, affichage=None):
        """
        Lance l'algorithme ACO pendant nb_iterations itérations.

        affichage : instance de Affichage OU None.
        Si non None, on met à jour l'affichage à chaque itération.
        Retourne un objet Route correspondant à la meilleure solution trouvée.
        """
        for it in range(1, self.nb_iterations + 1):
            toutes_routes = []
            toutes_longueurs = []

            # 1) Chaque fourmi construit un circuit
            for _ in range(self.nb_fourmis):
                route_indices = self.construire_route_indices()
                longueur = self.longueur_route_indices(route_indices)

                toutes_routes.append(route_indices)
                toutes_longueurs.append(longueur)

                # Mise à jour de la meilleure route globale
                if longueur < self.meilleure_distance:
                    self.meilleure_distance = longueur
                    self.meilleure_route_indices = route_indices

            # 2) Evaporation des phéromones
            self.pheromones *= (1.0 - self.evaporation)

            # 3) Dépôt de phéromones
            for route_indices, longueur in zip(toutes_routes, toutes_longueurs):
                depot = self.Q / longueur
                for i in range(len(route_indices) - 1):
                    a = route_indices[i]
                    b = route_indices[i + 1]
                    self.pheromones[a, b] += depot
                    self.pheromones[b, a] += depot  # graphe symétrique

            # 4) Meilleure route de l'itération (pour suivi)
            meilleur_iter = min(toutes_longueurs)
            index_best_iter = toutes_longueurs.index(meilleur_iter)
            meilleure_route_iter_indices = toutes_routes[index_best_iter]

            print(
                f"Itération {it}/{self.nb_iterations} - "
                f"Meilleure distance (itération) : {meilleur_iter:.2f} - "
                f"Meilleure distance globale : {self.meilleure_distance:.2f}"
            )

            # 5) Mise à jour graphique si un affichage est fourni
            if affichage is not None:
                # Conversion des indices en noms "L0", "L1", ...
                ordre_noms_iter = [f"L{i}" for i in meilleure_route_iter_indices]
                route_affichage = Route(self.graph, ordre=ordre_noms_iter)
                route_affichage.distance_totale()

                affichage.mettre_a_jour(
                    route_affichage,
                    route_affichage.distance,
                    it,
                    self.meilleure_distance
                )

        # À la fin : on construit un objet Route avec la meilleure route trouvée
        ordre_noms = [f"L{i}" for i in self.meilleure_route_indices]
        meilleure_route = Route(self.graph, ordre=ordre_noms)
        meilleure_route.distance_totale()

        return meilleure_route



class Affichage:
    """
    Classe d'affichage graphique avec Tkinter.
    Elle affiche :
    - les lieux du graphe sous forme de cercles
    - la meilleure route sous forme de ligne bleue pointillée
    - un texte d'information (itération, distance, etc.)
    """
    def __init__(self, graph, nom_groupe="Groupe X"):
        self.graph = graph
        self.nom_groupe = nom_groupe

        # Création de la fenêtre principale
        self.racine = tk.Tk()
        self.racine.title(f"TSP - Colonies de fourmis - {nom_groupe}")

        # Canvas pour dessiner les lieux et les routes
        self.canvas = tk.Canvas(self.racine, width=LARGEUR, height=HAUTEUR, bg="white")
        self.canvas.pack()

        # Zone de texte pour afficher les infos
        self.label_info = tk.Label(self.racine, text="", anchor="w", justify="left")
        self.label_info.pack(fill="x")

        # Pour mémoriser les positions des lieux
        # positions[i] = (x, y) pour le lieu d'indice i
        self.positions = []

        # ID de la ligne représentant la route affichée
        self.id_route = None
        # IDs des numéros d'ordre de visite
        self.id_ordre_textes = []

        # Dessin initial des lieux
        self.dessiner_lieux()

        # Raccourci clavier pour quitter
        self.racine.bind("<Escape>", self.quitter)

    def dessiner_lieux(self):
        """Dessine tous les lieux sous forme de cercles avec leur numéro."""
        rayon = 5
        for i, lieu in enumerate(self.graph.liste_lieux):
            x = lieu.x
            y = lieu.y
            self.positions.append((x, y))

            # Cercle
            self.canvas.create_oval(
                x - rayon, y - rayon, x + rayon, y + rayon,
                outline="black", fill="white"
            )

            # Numéro du lieu (indice)
            self.canvas.create_text(x, y - 10, text=str(i), fill="black")

    def dessiner_route(self, route):
        """
        Dessine une route (objet Route) sous forme de ligne bleue pointillée,
        et indique l'ordre de visite au-dessus de chaque lieu.
        """
        # Effacer l'ancienne route et les anciens numéros d'ordre
        if self.id_route is not None:
            self.canvas.delete(self.id_route)
        for tid in self.id_ordre_textes:
            self.canvas.delete(tid)
        self.id_ordre_textes = []

        # Construire la liste de coordonnées (x1, y1, x2, y2, ..., xn, yn)
        coords = []
        for nom_lieu in route.ordre:
            # nom_lieu = "L0", "L5", ...
            index = int(nom_lieu[1:])    # on récupère l'indice après le 'L'
            x, y = self.positions[index]
            coords.extend([x, y])

        # Dessiner la ligne pointillée
        self.id_route = self.canvas.create_line(
            *coords,
            fill="blue",
            dash=(4, 2),
            width=2
        )

        # Afficher l'ordre de visite au-dessus des lieux
        for ordre, nom_lieu in enumerate(route.ordre[:-1]):  # on ignore le dernier "L0" de retour
            index = int(nom_lieu[1:])
            x, y = self.positions[index]
            tid = self.canvas.create_text(
                x, y - 20,
                text=str(ordre),
                fill="blue",
                font=("Arial", 8)
            )
            self.id_ordre_textes.append(tid)

    def mettre_a_jour(self, route, distance, iteration, meilleure_distance_globale):
        """
        Met à jour l'affichage :
        - dessine la route
        - met à jour le texte d'information
        - force le rafraîchissement de la fenêtre
        """
        # Dessiner la route
        self.dessiner_route(route)

        # Texte d'information
        texte = (
            f"Itération : {iteration}\n"
            f"Distance de la meilleure route (itération) : {distance:.2f}\n"
            f"Meilleure distance globale : {meilleure_distance_globale:.2f}\n"
            f"(Appuyez sur ESC pour quitter)"
        )
        self.label_info.config(text=texte)

        # Rafraîchir la fenêtre Tkinter
        self.racine.update_idletasks()
        self.racine.update()

    def quitter(self, event=None):
        """Ferme la fenêtre."""
        self.racine.destroy()



if __name__ == "__main__":
    # 1) Création du graphe
    graph_test = Graph()
    graph_test.generer_lieux_aleatoires(2000)  

    # 2) Création de la fenêtre d'affichage
    affichage = Affichage(graph_test, nom_groupe="Groupe 2")

    # 3) Lancement de l'algorithme ACO
    aco = TSP_ACO(
        graph=graph_test,
        nb_fourmis=20,
        nb_iterations=50,
        alpha=1.0,
        beta=3.0,
        evaporation=0.3,
        Q=100.0
    )

    meilleure_route = aco.run(affichage)

    print("\n=== Résultat final ACO ===")
    print(meilleure_route)
    print(f"Distance totale : {meilleure_route.distance:.2f}")

    # 4) On laisse la fenêtre Tkinter ouverte
    try:
        affichage.racine.mainloop()
    except:
        pass

