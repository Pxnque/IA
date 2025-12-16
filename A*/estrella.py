import pygame
import math
import heapq

# Configuraciones iniciales
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
TURQUESA = (64, 224, 208)

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_abierto(self):
        self.color = VERDE

    def hacer_cerrado(self):
        self.color = ROJO

    def hacer_camino(self):
        self.color = TURQUESA

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

    def actualizar_vecinos(self, grid):
        self.vecinos = []
        filas = len(grid)
        col = self.col
        fila = self.fila

        # Movimientos cardinales (arriba, abajo, izquierda, derecha)
        if fila < filas - 1 and not grid[fila + 1][col].es_pared():  # Abajo
            self.vecinos.append(grid[fila + 1][col])
        if fila > 0 and not grid[fila - 1][col].es_pared():  # Arriba
            self.vecinos.append(grid[fila - 1][col])
        if col < filas - 1 and not grid[fila][col + 1].es_pared():  # Derecha
            self.vecinos.append(grid[fila][col + 1])
        if col > 0 and not grid[fila][col - 1].es_pared():  # Izquierda
            self.vecinos.append(grid[fila][col - 1])

        # Movimientos diagonales
        if fila < filas - 1 and col < filas - 1 and not grid[fila + 1][col + 1].es_pared():  # Abajo-Derecha
            self.vecinos.append(grid[fila + 1][col + 1])
        if fila < filas - 1 and col > 0 and not grid[fila + 1][col - 1].es_pared():  # Abajo-Izquierda
            self.vecinos.append(grid[fila + 1][col - 1])
        if fila > 0 and col < filas - 1 and not grid[fila - 1][col + 1].es_pared():  # Arriba-Derecha
            self.vecinos.append(grid[fila - 1][col + 1])
        if fila > 0 and col > 0 and not grid[fila - 1][col - 1].es_pared():  # Arriba-Izquierda
            self.vecinos.append(grid[fila - 1][col - 1])

    def __lt__(self, otro):
        return False


def h(p1, p2):
    """Heurística: distancia Euclidiana (soporta diagonales)"""
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def reconstruir_camino(origen, actual, dibujar):
    while actual in origen:
        actual = origen[actual]
        if not actual.es_inicio():
            actual.hacer_camino()
        dibujar()


def algoritmo_astar(dibujar, grid, inicio, fin):
    contador = 0
    conjunto_abierto = []
    heapq.heappush(conjunto_abierto, (0, contador, inicio))
    origen = {}
    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = h(inicio.get_pos(), fin.get_pos())

    conjunto_abierto_hash = {inicio}

    while conjunto_abierto:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        actual = heapq.heappop(conjunto_abierto)[2]
        conjunto_abierto_hash.remove(actual)

        if actual == fin:
            reconstruir_camino(origen, fin, dibujar)
            fin.hacer_fin()
            return True

        for vecino in actual.vecinos:
            # Costo: 1 para cardinales, sqrt(2) ≈ 1.414 para diagonales
            if abs(actual.fila - vecino.fila) == 1 and abs(actual.col - vecino.col) == 1:
                movimiento_costo = math.sqrt(2)
            else:
                movimiento_costo = 1

            temp_g_score = g_score[actual] + movimiento_costo

            if temp_g_score < g_score[vecino]:
                origen[vecino] = actual
                g_score[vecino] = temp_g_score
                f_score[vecino] = temp_g_score + h(vecino.get_pos(), fin.get_pos())
                if vecino not in conjunto_abierto_hash:
                    contador += 1
                    heapq.heappush(conjunto_abierto, (f_score[vecino], contador, vecino))
                    conjunto_abierto_hash.add(vecino)
                    vecino.hacer_abierto()

        dibujar()

        if actual != inicio:
            actual.hacer_cerrado()

    return False


def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid


def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))


def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()


def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col


def main(ventana, ancho):
    FILAS = 7
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None

    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()

                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()

                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)

                    algoritmo_astar(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)

                if event.key == pygame.K_c:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)

    pygame.quit()


main(VENTANA, ANCHO_VENTANA)