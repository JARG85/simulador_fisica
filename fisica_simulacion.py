import numpy as np

# Constantes físicas (ejemplo, si algunas no fueran configurables por UI)
# Por ahora, las masas y la velocidad inicial del neutrón se pasarán como argumentos,
# ya que son configurables desde la UI.

def calcular_colision_n_c1(m_n, m_c1, v_n1x_real, v_n1y_real, pos_n_inicial_x, pos_n_inicial_y, pos_c1_inicial_x, pos_c1_inicial_y):
    """
    Calcula las velocidades del neutrón (N) y del carbono 1 (C1) después de una colisión elástica 2D.
    C1 se considera inicialmente en reposo en (pos_c1_inicial_x, pos_c1_inicial_y).
    El neutrón se aproxima desde (pos_n_inicial_x, pos_n_inicial_y).

    Args:
        m_n (float): Masa del neutrón.
        m_c1 (float): Masa del carbono 1.
        v_n1x_real (float): Velocidad inicial del neutrón en X (real, m/s).
        v_n1y_real (float): Velocidad inicial del neutrón en Y (real, m/s).
        pos_n_inicial_x (float): Posición X inicial del neutrón (unidades de simulación).
        pos_n_inicial_y (float): Posición Y inicial del neutrón (unidades de simulación).
        pos_c1_inicial_x (float): Posición X inicial de C1 (unidades de simulación, punto de colisión).
        pos_c1_inicial_y (float): Posición Y inicial de C1 (unidades de simulación, punto de colisión).

    Returns:
        tuple: Contiene las velocidades reales post-colisión:
               (v_n2x_real, v_n2y_real, v_c1_after_n_x_real, v_c1_after_n_y_real, angle_of_impact_n_c1)
               Las velocidades son en m/s. angle_of_impact_n_c1 es en radianes.
    """
    # Vector desde la posición inicial del neutrón hasta el punto de colisión (C1)
    dx_approach_n_c1 = pos_c1_inicial_x - pos_n_inicial_x
    dy_approach_n_c1 = pos_c1_initial_y - pos_n_inicial_y

    v_n1_magnitud_real = np.sqrt(v_n1x_real**2 + v_n1y_real**2)

    # Ángulo de la línea de centros en el momento del impacto N-C1.
    # Esta es la dirección a lo largo de la cual se transfiere el momento en el modelo simplificado.
    if dx_approach_n_c1 == 0 and dy_approach_n_c1 == 0: # N y C1 empiezan en el mismo sitio
        if v_n1_magnitud_real > 1e-9: # Si el neutrón tiene velocidad, usar su dirección
            angle_of_impact_n_c1 = np.arctan2(v_n1y_real, v_n1x_real)
        else: # Ambas partículas en el mismo sitio y N sin velocidad, impacto indefinido (o sin impacto)
            angle_of_impact_n_c1 = 0.0 # Asumir un ángulo, aunque no habrá cambio de momento
    else:
        angle_of_impact_n_c1 = np.arctan2(dy_approach_n_c1, dx_approach_n_c1)

    # Componentes de la velocidad del neutrón ANTES del choque N-C1, a lo largo de la línea de impacto.
    # C1 está en reposo, por lo que su velocidad proyectada es 0.
    # v_n1_parallel_impact: componente de v_n1 a lo largo de la línea de impacto.
    # v_n1_perp_impact: componente de v_n1 perpendicular a la línea de impacto.
    cos_impact_angle = np.cos(angle_of_impact_n_c1)
    sin_impact_angle = np.sin(angle_of_impact_n_c1)

    v_n1_parallel_impact = v_n1x_real * cos_impact_angle + v_n1y_real * sin_impact_angle
    v_n1_perp_impact = -v_n1x_real * sin_impact_angle + v_n1y_real * cos_impact_angle

    # C1 está en reposo, v_c1_initial_parallel_impact = 0, v_c1_initial_perp_impact = 0.

    # Calcular velocidades DESPUÉS del choque N-C1 a lo largo de la línea de impacto (fórmulas 1D elásticas).
    # La partícula 1 es el neutrón (m_n, v_n1_parallel_impact).
    # La partícula 2 es C1 (m_c1, 0).
    v_n2_parallel_impact = ((m_n - m_c1) / (m_n + m_c1)) * v_n1_parallel_impact
    v_c1_after_n_parallel_impact = (2 * m_n / (m_n + m_c1)) * v_n1_parallel_impact

    # Las componentes perpendiculares a la línea de impacto no cambian en una colisión ideal sin fricción/rotación.
    v_n2_perp_impact = v_n1_perp_impact
    v_c1_after_n_perp_impact = 0  # C1 estaba en reposo, su componente perpendicular era 0 y sigue siendo 0.

    # Convertir velocidades post-choque N-C1 de nuevo a coordenadas X, Y.
    v_n2x_real = v_n2_parallel_impact * cos_impact_angle - v_n2_perp_impact * sin_impact_angle
    v_n2y_real = v_n2_parallel_impact * sin_impact_angle + v_n2_perp_impact * cos_impact_angle

    v_c1_after_n_x_real = v_c1_after_n_parallel_impact * cos_impact_angle - v_c1_after_n_perp_impact * sin_impact_angle
    v_c1_after_n_y_real = v_c1_after_n_parallel_impact * sin_impact_angle + v_c1_after_n_perp_impact * cos_impact_angle

    return v_n2x_real, v_n2y_real, v_c1_after_n_x_real, v_c1_after_n_y_real, angle_of_impact_n_c1


def calcular_colision_c1_c2(m_c1, m_c2, v_c1_before_c2_x_real, v_c1_before_c2_y_real):
    """
    Calcula las velocidades del carbono 1 (C1) y del carbono 2 (C2) después de una colisión elástica 2D.
    C2 se considera inicialmente en reposo. C1 tiene velocidad (v_c1_before_c2_x_real, v_c1_before_c2_y_real).
    Se asume una colisión frontal a lo largo de la dirección de movimiento de C1.

    Args:
        m_c1 (float): Masa del carbono 1.
        m_c2 (float): Masa del carbono 2.
        v_c1_before_c2_x_real (float): Velocidad de C1 en X antes de chocar con C2 (real, m/s).
        v_c1_before_c2_y_real (float): Velocidad de C1 en Y antes de chocar con C2 (real, m/s).

    Returns:
        tuple: Contiene las velocidades reales post-colisión:
               (v_c1_after_c2_x_real, v_c1_after_c2_y_real, v_c2_after_c1_x_real, v_c2_after_c1_y_real)
               Las velocidades son en m/s.
    """
    # Ángulo de impacto para C1-C2 es la dirección de la velocidad de C1 antes del choque.
    # Esto asume que C1 se mueve directamente hacia el centro de C2 (colisión frontal).
    angle_of_impact_c1_c2 = np.arctan2(v_c1_before_c2_y_real, v_c1_before_c2_x_real)

    cos_impact_angle = np.cos(angle_of_impact_c1_c2)
    sin_impact_angle = np.sin(angle_of_impact_c1_c2)

    # Proyectar velocidad de C1 (v_c1_before_c2) sobre esta línea de impacto.
    # v_c1_parallel_impact es la magnitud de la velocidad de C1, ya que el ángulo de impacto es su dirección.
    v_c1_parallel_impact = v_c1_before_c2_x_real * cos_impact_angle + v_c1_before_c2_y_real * sin_impact_angle
    # v_c1_perp_impact será 0 si angle_of_impact_c1_c2 es la dirección exacta de v_c1_before_c2.
    v_c1_perp_impact = -v_c1_before_c2_x_real * sin_impact_angle + v_c1_before_c2_y_real * cos_impact_angle

    # C2 está en reposo, v_c2_initial_parallel_impact = 0, v_c2_initial_perp_impact = 0.

    # Fórmulas 1D elásticas para componentes paralelas.
    # Partícula 1 es C1 (m_c1, v_c1_parallel_impact).
    # Partícula 2 es C2 (m_c2, 0).
    v_c1_after_c2_parallel = ((m_c1 - m_c2) / (m_c1 + m_c2)) * v_c1_parallel_impact
    v_c2_after_c1_parallel = (2 * m_c1 / (m_c1 + m_c2)) * v_c1_parallel_impact

    # Componentes perpendiculares no cambian.
    v_c1_after_c2_perp = v_c1_perp_impact # Debería ser 0
    v_c2_after_c1_perp = 0.0

    # Convertir de nuevo a X,Y.
    v_c1_after_c2_x_real = v_c1_after_c2_parallel * cos_impact_angle - v_c1_after_c2_perp * sin_impact_angle
    v_c1_after_c2_y_real = v_c1_after_c2_parallel * sin_impact_angle + v_c1_after_c2_perp * cos_impact_angle

    v_c2_after_c1_x_real = v_c2_after_c1_parallel * cos_impact_angle # Perp es 0 para C2
    v_c2_after_c1_y_real = v_c2_after_c1_parallel * sin_impact_angle

    return v_c1_after_c2_x_real, v_c1_after_c2_y_real, v_c2_after_c1_x_real, v_c2_after_c1_y_real

def calcular_tiempo_hasta_colision(pos_inicial_particula1_x, pos_inicial_particula1_y,
                                   pos_particula2_x, pos_particula2_y,
                                   vel_particula1_x, vel_particula1_y):
    """
    Calcula el tiempo para que la partícula 1, moviéndose desde su posición inicial,
    alcance la posición de la partícula 2 (asumida estática).
    Esto es una simplificación para colisión con el centro de la partícula 2.

    Args:
        pos_inicial_particula1_x (float): Posición X inicial de la partícula 1.
        pos_inicial_particula1_y (float): Posición Y inicial de la partícula 1.
        pos_particula2_x (float): Posición X de la partícula 2 (objetivo).
        pos_particula2_y (float): Posición Y de la partícula 2 (objetivo).
        vel_particula1_x (float): Velocidad en X de la partícula 1 (m/s).
        vel_particula1_y (float): Velocidad en Y de la partícula 1 (m/s).

    Returns:
        float: Tiempo en segundos hasta la colisión, o float('inf') si no hay colisión directa.
    """
    # Vector desde partícula 1 hacia partícula 2
    dx_to_target = pos_particula2_x - pos_inicial_particula1_x
    dy_to_target = pos_particula2_y - pos_inicial_particula1_y

    # Proyección de la velocidad de la partícula 1 sobre el vector que une partícula 1 y 2
    # dot_product = vel_particula1_x * dx_to_target + vel_particula1_y * dy_to_target
    # if dot_product <= 0: # Partícula 1 no se mueve hacia partícula 2
    #     return float('inf')

    time_to_collision_sec = float('inf')

    # Calcular el tiempo 't' tal que:
    # P1_x(t) = pos_inicial_particula1_x + vel_particula1_x * t = pos_particula2_x
    # P1_y(t) = pos_inicial_particula1_y + vel_particula1_y * t = pos_particula2_y

    time_tx = float('inf')
    if abs(vel_particula1_x) > 1e-9: # Evitar división por cero si no hay movimiento en X
        time_tx = dx_to_target / vel_particula1_x
    elif abs(dx_to_target) < 1e-9: # Ya está en la coordenada X correcta
        time_tx = 0.0 # O un tiempo "infinitesimal" si se prefiere, pero 0 es más simple para coincidencia

    time_ty = float('inf')
    if abs(vel_particula1_y) > 1e-9: # Evitar división por cero si no hay movimiento en Y
        time_ty = dy_to_target / vel_particula1_y
    elif abs(dy_to_target) < 1e-9: # Ya está en la coordenada Y correcta
        time_ty = 0.0

    # Para una colisión directa con el centro, los tiempos para alcanzar X e Y deben ser iguales y positivos.
    if time_tx >= 0 and time_ty >= 0:
        if abs(time_tx - time_ty) < 1e-9: # Los tiempos son consistentes
            time_to_collision_sec = time_tx # o time_ty, son iguales
        elif time_tx == float('inf') and time_ty != float('inf'): # Movimiento puramente vertical hacia el objetivo
             if abs(dx_to_target) < 1e-9: # Si ya está alineado en X
                 time_to_collision_sec = time_ty
        elif time_ty == float('inf') and time_tx != float('inf'): # Movimiento puramente horizontal hacia el objetivo
             if abs(dy_to_target) < 1e-9: # Si ya está alineado en Y
                 time_to_collision_sec = time_tx

    # Si uno es 0 y el otro inf, significa que está alineado en un eje pero no se mueve en el otro para alcanzarlo.
    # ej. tx=0 (alineado en x), ty=inf (no se mueve en y para alcanzar y_target) -> no colisión.
    if (time_tx == 0 and time_ty == float('inf') and abs(dy_to_target) > 1e-9) or \
       (time_ty == 0 and time_tx == float('inf') and abs(dx_to_target) > 1e-9):
        time_to_collision_sec = float('inf')

    # Si ambos son 0, significa que ya están en el mismo punto.
    if abs(time_tx) < 1e-9 and abs(time_ty) < 1e-9 and (abs(dx_to_target) > 1e-9 or abs(dy_to_target) > 1e-9):
        # Esto puede ocurrir si vel_particula1 es cero pero no están en el mismo punto.
        # O si vel_particula1 no es cero, pero dx/dy son cero.
        # La condición abs(dx_to_target) < 1e-9 dentro de los cálculos de time_tx/ty ya maneja dx=0.
        # Si dx=0, tx=0. Si dy=0, ty=0. Si ambos son 0, time_to_collision_sec = 0.
        pass # El resultado de 0 es correcto si están en el mismo sitio o si las velocidades los llevan allí instantáneamente.


    if time_to_collision_sec < 0: # Sucedió en el pasado, no es una colisión futura
        time_to_collision_sec = float('inf')

    return time_to_collision_sec

def calcular_tiempo_salida_pantalla(pos_inicial_x, pos_inicial_y, vel_x, vel_y, x_lim_left, x_lim_right, y_lim_bottom, y_lim_top, margin):
    """
    Calcula el tiempo teórico para que una partícula salga de los límites de la pantalla.

    Args:
        pos_inicial_x (float): Posición X inicial de la partícula.
        pos_inicial_y (float): Posición Y inicial de la partícula.
        vel_x (float): Velocidad en X de la partícula (m/s).
        vel_y (float): Velocidad en Y de la partícula (m/s).
        x_lim_left (float): Límite izquierdo del área visible.
        x_lim_right (float): Límite derecho del área visible.
        y_lim_bottom (float): Límite inferior del área visible.
        y_lim_top (float): Límite superior del área visible.
        margin (float): Margen para considerar el tamaño de la partícula.

    Returns:
        float: Tiempo mínimo en segundos para salir de la pantalla, o float('inf') si no sale.
    """
    tiempos_salida = []

    if vel_x > 1e-9: # Se mueve a la derecha
        dist_a_borde_derecho = (x_lim_right - margin) - pos_inicial_x
        if dist_a_borde_derecho > 0:
            tiempos_salida.append(dist_a_borde_derecho / vel_x)
    elif vel_x < -1e-9: # Se mueve a la izquierda
        dist_a_borde_izquierdo = pos_inicial_x - (x_lim_left + margin)
        if dist_a_borde_izquierdo > 0:
            tiempos_salida.append(dist_a_borde_izquierdo / abs(vel_x))

    if vel_y > 1e-9: # Se mueve hacia arriba
        dist_a_borde_superior = (y_lim_top - margin) - pos_inicial_y
        if dist_a_borde_superior > 0:
            tiempos_salida.append(dist_a_borde_superior / vel_y)
    elif vel_y < -1e-9: # Se mueve hacia abajo
        dist_a_borde_inferior = pos_inicial_y - (y_lim_bottom + margin)
        if dist_a_borde_inferior > 0:
            tiempos_salida.append(dist_a_borde_inferior / abs(vel_y))

    if not tiempos_salida:
        return float('inf') # No se mueve o se mueve paralelo a un borde sin cruzarlo.

    return min(t for t in tiempos_salida if t >= 0) # Considerar solo tiempos positivos

# Más funciones de física podrían ir aquí, por ejemplo, para calcular energía, momento, etc.
# si fueran necesarias para la lógica o visualización.
