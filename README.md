
# Requisitos

Python >= 3.10
NumPy >= 1.20.0

# Instalar

## En caso de no tener NumPy intalado

Dado que la unica libreria externa que ocupa es NumPy, se puede
instalar por el metodo de preferencia del evaluador,
aqui se adjuntan algunos comandos que deberian funcionar:

Linux y Macos:

``` Bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

Powershell:

``` Powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

```

CMD:

``` CMD
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

# Correr

Las soluciones para cada reto se encuentra en el archivo respectivo con cada nombre.
En el caso del reto 2, se adjunta tanto la solucion exacta de la frafica 7
como aquella aproximada.

Linux y Macos:

``` Bash
python3 reto1.py
python3 reto2.py
python3 reto2exacto.py
python3 reto3.py
python3 reto4.py
```

Windows:

``` Powershell
python reto1.py
python reto2.py
python reto2exacto.py
python reto3.py
python reto4.py
```

Tambien se puede ejecutar desde el editor de preferencia del evaluador

# Estructura del proyecto

Se escogio la siguiente estructura para poder reutilizar codigo sin caer
en dependencias recursivas.

### loaddata.py

Se usa para cargar los datos de /Data y seleccionar la semilla aleatoria.

### wildfire_core.py

Funciones basicas para el simulador de incendio forestal, poner cortafuegos,
encontrar la frontera y la frontera en llamas.

### wildfire_simulator.py

Es el simulador de incendio forestal que se uso para la solucion de los ejercicios
1, 3, 4 y 5. Funciona tanto con bomberos como sin bomberos, se adjunta tambien la funcion
de monte carlo, para poder realizar miles de simulaciones.

### heuristics.py

La base de la logica de los algoritmos de prevencion y colocacion de cortafuegos de manera
dinamica se encuentran aqui. Estos algoritmos son usados practicamente para los 5 retos,
(Menos la solucion exacta del reto 2). Se bassan en la busqueda en la busqueda de cotas
superiores para la colocacion de los cortafuegos.

### reto1.py a reto5.py

Aqui se encuentran las soluciones de cada reto. En el caso del reto 2 se adjunta tanto la
solucion exacta en el grafo 7 como la aproximiada para todos los demas, la cual es ademas
si es generalizabe a grafos mas grandes.

En el caso del reto 1 se adjunta la solucion generalizabe aproximada a tiempos mayor a 2.
La solcuion teorica exacta se adjunta en la memoria.
