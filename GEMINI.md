# Resumen de Contexto: "Lo que arde" y Modelado de Propagación

---

## 1. El Problema Central: *Lo que arde*

El núcleo del trabajo proviene del documento técnico y concurso de modelado matemático titulado **"Lo que arde"** (con fecha 1 de octubre de 2026), inspirado en el contexto de incendios forestales de la película homónima de Oliver Laxe[cite: 1].

### Modelado del Bosque
* **Estructura espacial:** El bosque se discretiza en $n$ zonas homogéneas conectadas entre sí, representadas formalmente mediante un **grafo dirigido** $G = (N, A)$[cite: 1, 3]:
  * $N = \{1, \dots, n\}$ es el conjunto de nodos/zonas[cite: 3].
  * $A \subset N \times N$ es el conjunto de aristas dirigidas (arcos), cumpliendo que $(i, j) \in A \iff (j, i) \in A$, pero con probabilidades de transmisión generalmente asimétricas[cite: 1, 3].
* **Pesos estocásticos:** Cada arco $(i, j) \in A$ posee una probabilidad $p_{ij} \in [0, 1]$ de que el fuego se propague desde la zona $i$ hacia la zona $j$[cite: 3].

### Dinámica Temporal y Estocástica
* **Tiempo discreto:** La propagación ocurre en etapas sucesivas $t = 0, 1, 2, \dots$[cite: 3]
* **Frentes de fuego:** Se define $I_t$ como el conjunto de zonas que se encienden exactamente en la etapa $t$[cite: 3].
* **Mecanismo de ignición:** 
  * Los intentos de propagación desde cada nodo activo $i \in I_t$ hacia una zona vecina susceptible ocurren **una única vez** y de forma **estadísticamente independiente**[cite: 3].
* **Acumulación y parada:** El conjunto total de zonas quemadas hasta la etapa $t$ es $Q_t = \bigcup_{t' < t} I_{t'}$[cite: 3]. El proceso finaliza en la etapa $T$ tal que $I_T = \emptyset$[cite: 3].

---

## 2. Estructura de Retos del Proyecto

1. **Reto 1 (Prevención a priori):**
   * Determinación estática de $k = 4$ cortafuegos óptimos antes del inicio del fuego para minimizar el número esperado de zonas quemadas tras $s = 2$ etapas, bajo una distribución de ignición inicial $\sum_l p_l = 1$[cite: 3].
2. **Reto 2 (Escenario determinista / explosivo):**
   * Caso con $p_{ij} = 1$ para todo $(i, j) \in A$[cite: 3].
   * Juego por turnos: tras cada expansión, los bomberos colocan $k = 2$ cortafuegos dinámicos permanentes[cite: 3].
   * Desarrollo de métodos exactos (sin fuerza bruta exhaustiva) y heurísticas aproximadas escalables[cite: 3].
3. **Reto 3 (Propagación estocástica y dinámica adaptativa):**
   * Propagación general con $p_{ij} \in [0, 1]$ variables[cite: 4].
   * Colocación dinámica de $k = 1$ cortafuego por etapa basada en el estado observado[cite: 4].
   * Algoritmos de optimización estocástica y simulación Monte Carlo para maximizar el valor esperado de zonas salvadas[cite: 4].
4. **Reto 4 (Optimización adversaria / Teoría de juegos):**
   * Un pirómano selecciona el nodo inicial $q \in N$ para maximizar el daño esperado frente a la respuesta óptima de los bomberos[cite: 4].
5. **Reto 5 (Generalización y realismo):**
   * Extensiones hacia dinámicas continuas, factores topográficos/meteorológicos variables y formulaciones matemáticas avanzadas[cite: 4].

---

## 3. El Planteamiento Evaluado (Ejercicio 1)

Se evaluó la propuesta algebraica para predecir si un nodo $x_i$ arde en la siguiente etapa:
* Un vector dual/estado $\mathbf{b} \in \{0, 1\}^n$, donde $b_j = 1$ indica que el nodo $j$ está activo/ardiendo y $b_j = 0$ inactivo.
* Un vector de pesos entrantes $\mathbf{w}_i = (p_{1i}, p_{2i}, \dots, p_{ni})^\top$.
* La hipótesis de que la probabilidad de que $x_i$ se prenda en llamas corresponde a la forma bilineal:
  $$b(x_i) = \mathbf{b}^\top \mathbf{w}_i = \sum_{j=1}^n b_j p_{ji}$$

---

## 4. El Enfoque y la Resolución Técnica

El análisis contrastó rigurosamente la formulación lineal con la teoría axiomática de la probabilidad y la física del modelo:

1. **Invalidez de la linealidad simple:**
   * La suma $\sum_{j=1}^n b_j p_{ji}$ calcula el **número esperado de transmisiones exitosas** hacia $x_i$, no la probabilidad del evento unión $\bigcup_j \{j \to i\}$.
   * Viola el axioma de acotación de Kolmogorov ($\mathbb{P} \le 1$); si dos vecinos activos tienen $p_{j_1 i} = 0.8$ y $p_{j_2 i} = 0.7$, la suma arroja $1.5 > 1$.
2. **Formulación estocástica exacta:**
   * Dada la independencia de los eventos de propagación, la probabilidad de que $x_i$ **no** se encienda es el producto de que todos los vecinos activos fallen[cite: 3]:
     $$\mathbb{P}(\text{no arde } x_i) = \prod_{j=1}^n (1 - p_{ji})^{b_j}$$
   * La probabilidad exacta de que $x_i$ se encienda en la siguiente etapa es la complementaria:
     $$\mathbb{P}(x_i \text{ arde}) = 1 - \prod_{j=1}^n (1 - p_{ji})^{b_j}$$
3. **Aproximación de primer orden:**
   * La expresión lineal $\mathbf{b}^\top \mathbf{w}_i$ únicamente es admisible asintóticamente cuando $p_{ji} \ll 1$ mediante la aproximación de Taylor:
     $$1 - \prod_{j: b_j=1}(1 - p_{ji}) \approx \sum_{j: b_j=1} p_{ji}$$

---

## 5. Perspectiva Global del Proyecto

* **Marco metodológico:** Enfoque analítico y formal propio de la física matemática, procesos estocásticos de percolación/contacto en redes complejas, combinatoria poliédrica y teoría de grafos.
* **Propósito:** Establecer bases analíticas exactas para el diseño e implementación de algoritmos eficientes (Monte Carlo, programación entera mixta o búsqueda heurística) orientados a resolver los retos de toma de decisiones dinámicas del concurso.
