import pygame

# Configuraciones iniciales
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Algoritmo A* - Visualización")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
AZUL = (0, 0, 255)

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
    
    def hacer_cerrado(self):
        self.color = ROJO
    
    def hacer_abierto(self):
        self.color = VERDE
    
    def hacer_camino(self):
        self.color = AZUL
    
    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))
    
    def actualizar_vecinos(self, grid):
        """
        PASO 1: Encuentra todos los vecinos válidos del nodo actual
        Solo considera movimientos ortogonales (arriba, abajo, izquierda, derecha)
        """
        self.vecinos = []
        
        # Abajo
        if self.fila < self.total_filas - 1 and not grid[self.fila + 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col])
        
        # Arriba
        if self.fila > 0 and not grid[self.fila - 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col])
        
        # Derecha
        if self.col < self.total_filas - 1 and not grid[self.fila][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col + 1])
        
        # Izquierda
        if self.col > 0 and not grid[self.fila][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col - 1])

def heuristica(p1, p2):
    """
    FUNCIÓN HEURÍSTICA: Distancia de Manhattan
    
    Calcula: |x1 - x2| + |y1 - y2|
    
    Esta es una estimación ADMISIBLE (nunca sobreestima) del costo real
    porque en un grid con movimientos ortogonales, la distancia Manhattan
    es exactamente el número mínimo de pasos necesarios sin obstáculos.
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruir_camino(vino_de, actual, dibujar):
    """
    RECONSTRUCCIÓN DEL CAMINO:
    Una vez que llegamos al objetivo, retrocedemos desde el fin hasta el inicio
    usando el diccionario 'vino_de' que almacena de dónde vino cada nodo.
    """
    while actual in vino_de:
        actual = vino_de[actual]
        actual.hacer_camino()
        dibujar()

def algoritmo_a_estrella(dibujar, grid, inicio, fin):
    """
    ALGORITMO A* - Implementación sin PriorityQueue
    
    CONCEPTOS CLAVE:
    - g_score: costo REAL desde el inicio hasta el nodo actual
    - h_score: costo ESTIMADO (heurística) desde el nodo actual hasta el fin
    - f_score: g_score + h_score (costo total estimado)
    
    ESTRATEGIA:
    Siempre expandimos el nodo con menor f_score, lo que garantiza
    encontrar el camino más corto (si la heurística es admisible).
    """
    
    # Lista abierta: nodos por explorar
    lista_abierta = [inicio]
    
    # Diccionario para rastrear el camino
    vino_de = {}
    
    # Inicializar g_score: todos infinito excepto el inicio
    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0
    
    # Inicializar f_score: todos infinito excepto el inicio
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = heuristica(inicio.get_pos(), fin.get_pos())
    
    while lista_abierta:
        # Permitir cerrar la ventana durante la ejecución
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        # PASO 2: Encontrar el nodo con MENOR f_score en la lista abierta
        # (Esto reemplaza a PriorityQueue)
        nodo_actual = min(lista_abierta, key=lambda nodo: f_score[nodo])
        
        # PASO 3: ¿Llegamos al objetivo?
        if nodo_actual == fin:
            reconstruir_camino(vino_de, fin, dibujar)
            fin.hacer_fin()
            inicio.hacer_inicio()
            return True
        
        # Remover de la lista abierta
        lista_abierta.remove(nodo_actual)
        
        # PASO 4: Explorar todos los vecinos del nodo actual
        for vecino in nodo_actual.vecinos:
            # Calcular el g_score tentativo
            # Como cada paso cuesta 1, sumamos 1 al g_score actual
            temp_g_score = g_score[nodo_actual] + 1
            
            # PASO 5: ¿Encontramos un MEJOR camino hacia este vecino?
            if temp_g_score < g_score[vecino]:
                # ¡Sí! Actualizar el camino
                vino_de[vecino] = nodo_actual
                g_score[vecino] = temp_g_score
                f_score[vecino] = temp_g_score + heuristica(vecino.get_pos(), fin.get_pos())
                
                # Si el vecino no está en la lista abierta, agregarlo
                if vecino not in lista_abierta:
                    lista_abierta.append(vecino)
                    vecino.hacer_abierto()
        
        # Actualizar visualización
        dibujar()
        
        # PASO 6: Marcar el nodo actual como explorado (cerrado)
        if nodo_actual != inicio:
            nodo_actual.hacer_cerrado()
    
    # Si salimos del while, no hay camino posible
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
    FILAS = 50
    grid = crear_grid(FILAS, ancho)
    
    inicio = None
    fin = None
    
    corriendo = True
    
    print("=" * 60)
    print("INSTRUCCIONES - ALGORITMO A*")
    print("=" * 60)
    print("1. Click IZQUIERDO:")
    print("   - Primer click: Coloca el INICIO (naranja)")
    print("   - Segundo click: Coloca el FIN (morado)")
    print("   - Siguientes clicks: Dibuja PAREDES (negro)")
    print()
    print("2. Click DERECHO: Borra nodos")
    print()
    print("3. Tecla ESPACIO: Ejecuta el algoritmo A*")
    print()
    print("4. Tecla C: Limpia todo el grid")
    print()
    print("COLORES:")
    print("  🟧 Naranja = Inicio")
    print("  🟪 Morado = Fin")
    print("  ⬛ Negro = Pared")
    print("  🟩 Verde = Nodos en lista abierta (por explorar)")
    print("  🟥 Rojo = Nodos cerrados (ya explorados)")
    print("  🟦 Azul = CAMINO ÓPTIMO encontrado")
    print("=" * 60)
    
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
                # Presionar ESPACIO para iniciar el algoritmo
                if event.key == pygame.K_SPACE and inicio and fin:
                    print("\n🚀 Ejecutando A*...")
                    
                    # Actualizar vecinos de todos los nodos
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)
                    
                    # Ejecutar el algoritmo
                    resultado = algoritmo_a_estrella(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)
                    
                    if resultado:
                        print("✅ ¡Camino encontrado!")
                    else:
                        print("❌ No existe camino posible")
                
                # Presionar C para limpiar todo
                if event.key == pygame.K_c:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)
                    print("\n🧹 Grid limpiado")
    
    pygame.quit()

main(VENTANA, ANCHO_VENTANA)