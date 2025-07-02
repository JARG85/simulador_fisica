import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import matplotlib

# Importar los nuevos módulos de física y animación
import fisica_simulacion as fs
import animacion_simulacion as anim_sim

# matplotlib.use('TkAgg') # Asegura que la ventana de la animación se muestre

# --- Configuración Inicial de la Simulación y UI ---

# Valores iniciales por defecto (pueden ser modificados por la UI)
v_n1_magnitud_real_default = 2.6e7
m_c1_default = 12.0
m_c2_default = 12.0
m_n_default = 1.0 # Masa del neutrón (u)
angulo_neutron_default = np.pi / 4  # 45 grados, dirección del neutrón inicial
DIST_C1_TO_C2_TARGET_default = 0.3 # Distancia visual deseada para colocar C2 en la trayectoria de C1

# Factor de escala para la animación (velocidad real a velocidad de animación)
ANIMATION_SPEED_FACTOR = 2.69e-10
# Intervalo de tiempo entre frames de la animación en milisegundos
FIXED_ANIMATION_INTERVAL_MS_CONFIG = 20

# --- Configuración de la Figura y Ejes de Matplotlib ---
# Crear la figura principal que contendrá la simulación y los controles
fig = plt.figure(figsize=(12, 8)) # Tamaño de la figura en pulgadas

# Añadir ejes para el área de simulación ([left, bottom, width, height] en fracciones de la figura)
ax_simulacion = fig.add_axes([0.08, 0.1, 0.55, 0.8])

# Establecer límites y aspecto del área de simulación
ax_simulacion.set_xlim(-0.1, 1.7)
ax_simulacion.set_ylim(-0.1, 1.0)
ax_simulacion.set_aspect('equal') # Asegura que las escalas de X e Y sean iguales

# Guardar límites para pasarlos a otros módulos o usarlos en cálculos
x_lim_sim = ax_simulacion.get_xlim()
y_lim_sim = ax_simulacion.get_ylim()
# Margen alrededor de las partículas para la detección de salida de pantalla
margin_for_particle_size_config = 0.05

# Elementos visuales de las partículas (objetos Line2D de Matplotlib)
# Se inicializan con datos vacíos; se actualizarán en cada frame.
neutron_line_mpl, = ax_simulacion.plot([], [], 'o', color='orange', markersize=15, label='Neutrón', zorder=2)
carbon1_line_mpl, = ax_simulacion.plot([], [], 'o', color='gray', markersize=10, label='Carbono 1 (C1)', zorder=2) # Tamaño se actualizará
carbon2_line_mpl, = ax_simulacion.plot([], [], 'o', color='blue', markersize=10, label='Carbono 2 (C2)', zorder=2) # Tamaño se actualizará

# Texto para mostrar el estado actual de la simulación (ej. Fase 1, Fase 2)
text_state_mpl = ax_simulacion.text(0.5, 0.99, '', transform=ax_simulacion.transAxes, fontsize=12, weight='bold', ha='center', va='top')

# Dibujar ejes X e Y visuales como referencia en la simulación
ax_simulacion.axhline(0, color='black', linewidth=0.5, zorder=1) # Eje X
ax_simulacion.axvline(0, color='black', linewidth=0.5, zorder=1) # Eje Y
# Flechas para los ejes
ax_simulacion.arrow(0.0, 0, x_lim_sim[1] - 0.2, 0, head_width=0.03, head_length=0.05, fc='black', ec='black', zorder=1)
ax_simulacion.text(x_lim_sim[1] - 0.15, -0.05, 'X', fontsize=12, zorder=1)
ax_simulacion.arrow(0, 0.0, 0, y_lim_sim[1] - 0.1, head_width=0.03, head_length=0.05, fc='black', ec='black', zorder=1)
ax_simulacion.text(-0.05, y_lim_sim[1] - 0.05, 'Y', fontsize=12, zorder=1)
ax_simulacion.legend(loc='upper right', bbox_to_anchor=(1.0, 0.95)) # Leyenda de partículas
ax_simulacion.set_yticks([]) # Ocultar los ticks del eje Y para limpiar la visualización

# --- Estado de la Simulación (gestionado centralmente por el orquestador) ---
# Este diccionario almacenará todos los parámetros y resultados relevantes.
# Se pasará (o partes de él) a los módulos de física y animación.
simulation_state = {
    # Parámetros configurables por el usuario (con valores por defecto)
    'm_n': m_n_default,
    'm_c1': m_c1_default,
    'm_c2': m_c2_default,
    'v_n1_magnitud_real': v_n1_magnitud_real_default,
    'angulo_neutron': angulo_neutron_default, # Ángulo fijo para la trayectoria inicial del neutrón
    'DIST_C1_TO_C2_TARGET': DIST_C1_TO_C2_TARGET_default, # Distancia visual C1 a C2

    # Posiciones iniciales de las partículas (unidades de simulación)
    'pos_n_initial_x': 0.1, 'pos_n_initial_y': 0.1, # Neutrón comienza en (0.1, 0.1)
    'pos_c1_initial_x': 0.8, 'pos_c1_initial_y': 0.2, # C1 en (0.8, 0.2)
    # Posición de C2 se calcula dinámicamente, pero necesita un valor inicial
    'pos_c2_initial_x': 0.8 + DIST_C1_TO_C2_TARGET_default,
    'pos_c2_initial_y': 0.2,

    # Puntos de colisión (calculados por el módulo de física, en unidades de simulación)
    'collision_point_n_c1_x': 0.0, 'collision_point_n_c1_y': 0.0,
    'collision_point_c1_c2_x': 0.0, 'collision_point_c1_c2_y': 0.0,

    # Velocidades reales (m/s), calculadas por el módulo de física
    'v_n1x_real': 0.0, 'v_n1y_real': 0.0, # Neutrón inicial
    'v_c1_initial_x_real': 0.0, 'v_c1_initial_y_real': 0.0, # C1 inicial (en reposo)
    'v_c2_initial_x_real': 0.0, 'v_c2_initial_y_real': 0.0, # C2 inicial (en reposo)
    'v_n2x_real': 0.0, 'v_n2y_real': 0.0, # Neutrón después de colisión con C1
    'v_c1_after_n_x_real': 0.0, 'v_c1_after_n_y_real': 0.0, # C1 después de colisión con Neutrón
    'v_c1_after_c2_x_real': 0.0, 'v_c1_after_c2_y_real': 0.0, # C1 después de colisión con C2
    'v_c2_after_c1_x_real': 0.0, 'v_c2_after_c1_y_real': 0.0, # C2 después de colisión con C1

    # Velocidades de animación (unidades de simulación / frame), calculadas en el orquestador
    'v_n1_anim_x': 0.0, 'v_n1_anim_y': 0.0,
    'v_n2_anim_x': 0.0, 'v_n2_anim_y': 0.0,
    'v_c1_after_n_anim_x': 0.0, 'v_c1_after_n_anim_y': 0.0,
    'v_c1_after_c2_anim_x': 0.0, 'v_c1_after_c2_anim_y': 0.0,
    'v_c2_after_c1_anim_x': 0.0, 'v_c2_after_c1_anim_y': 0.0,

    # Tiempos teóricos (s), calculados por el módulo de física
    'theoretical_time_to_n_c1_collision': 0.0,
    'theoretical_time_to_c1_c2_collision': 0.0,
    'total_theoretical_time_c1': float('inf'), # Tiempo total para que C1 salga o se detenga
    'total_theoretical_time_c2': float('inf'), # Tiempo total para que C2 salga o se detenga
    # Tiempos de colisión para la gráfica de velocidad (en s desde el inicio de la animación)
    't_col_n_c1_graph': 0.0,
    't_col_c1_c2_graph': 0.0,

    # Frames de animación para controlar las fases
    'frames_until_n_c1_collision': 0, # Frame en el que ocurre la colisión N-C1
    'frames_until_c1_c2_collision': 0, # Frame en el que ocurre la colisión C1-C2
    'total_frames_anim': 2000, # Número máximo de frames para la animación

    # Listas para almacenar datos para la gráfica de velocidad vs. tiempo
    'animation_times_list': [],
    'neutron_velocidades_escalar_list': [],
    'c1_velocidades_escalar_list': [],
    'c2_velocidades_escalar_list': [],
}

# --- Funciones Auxiliares del Orquestador ---
CARBON_MARKER_SCALE_FACTOR = 1.66 # Factor para escalar el tamaño visual del carbono con su masa
MIN_CARBON_MARKERSIZE = 5        # Tamaño mínimo del marcador de carbono
def get_carbon_markersize(mass_c):
    """Calcula el tamaño visual de una partícula de carbono basado en su masa."""
    return max(MIN_CARBON_MARKERSIZE, mass_c * CARBON_MARKER_SCALE_FACTOR)

def update_simulation_parameters_from_ui():
    """
    Actualiza el diccionario `simulation_state` con los valores ingresados en los
    TextBoxes de la interfaz de usuario. Realiza validación básica.
    Retorna True si la actualización es exitosa, False si hay error.
    """
    try:
        # Leer valores de los TextBoxes y convertirlos a float
        simulation_state['v_n1_magnitud_real'] = float(textbox_v_n.text)
        simulation_state['m_c1'] = float(textbox_m_c1_input.text)
        simulation_state['m_c2'] = float(textbox_m_c2_input.text)
        # m_n (masa del neutrón) y angulo_neutron son fijos por ahora, pero podrían añadirse a la UI.

        # Validar que las masas y la velocidad sean positivas
        if simulation_state['m_c1'] <= 0: raise ValueError("Masa C1 debe ser > 0.")
        if simulation_state['m_c2'] <= 0: raise ValueError("Masa C2 debe ser > 0.")
        if simulation_state['v_n1_magnitud_real'] <= 0: raise ValueError("Velocidad Neutrón debe ser > 0.")

        # Actualizar tamaño visual de las partículas de carbono en el gráfico
        carbon1_line_mpl.set_markersize(get_carbon_markersize(simulation_state['m_c1']))
        carbon2_line_mpl.set_markersize(get_carbon_markersize(simulation_state['m_c2']))
        return True
    except ValueError as e:
        # Mostrar error en el texto de estado si la entrada es inválida
        text_state_mpl.set_text(f"ERROR: {e}")
        if fig: fig.canvas.draw_idle() # Actualizar canvas para mostrar el mensaje de error
        return False

def run_simulation_logic():
    """
    Función principal que orquesta toda la lógica de la simulación.
    Se llama cuando el usuario inicia o reinicia la simulación.
    """
    # 1. Actualizar parámetros de simulación desde la UI
    if not update_simulation_parameters_from_ui():
        return # Detener si hay error en los parámetros de entrada

    # 2. Detener cualquier animación anterior y limpiar datos
    anim_sim.stop_current_animation_if_running() # Llama a la función del módulo de animación
    simulation_state['animation_times_list'].clear()
    simulation_state['neutron_velocidades_escalar_list'].clear()
    simulation_state['c1_velocidades_escalar_list'].clear()
    simulation_state['c2_velocidades_escalar_list'].clear()
    # Reiniciar textos de timers en la UI
    if global_text_timer_mpl: global_text_timer_mpl.set_text('Tiempo Sim.: 0.00 s')
    if global_text_theoretical_timer_mpl: global_text_theoretical_timer_mpl.set_text('Tiempo Teórico: Calculando...')

    # --- 3. Cálculos de Física (utilizando el módulo `fisica_simulacion`) ---
    
    # Calcular componentes X e Y de la velocidad inicial del neutrón
    v_n1x = simulation_state['v_n1_magnitud_real'] * np.cos(simulation_state['angulo_neutron'])
    v_n1y = simulation_state['v_n1_magnitud_real'] * np.sin(simulation_state['angulo_neutron'])
    simulation_state['v_n1x_real'], simulation_state['v_n1y_real'] = v_n1x, v_n1y
    # C1 y C2 están inicialmente en reposo
    simulation_state['v_c1_initial_x_real'], simulation_state['v_c1_initial_y_real'] = 0.0, 0.0
    simulation_state['v_c2_initial_x_real'], simulation_state['v_c2_initial_y_real'] = 0.0, 0.0

    # Calcular velocidades después de la colisión Neutrón-C1
    v_n2x, v_n2y, v_c1_an_x, v_c1_an_y, _ = fs.calcular_colision_n_c1(
        simulation_state['m_n'], simulation_state['m_c1'],
        v_n1x, v_n1y,
        simulation_state['pos_n_initial_x'], simulation_state['pos_n_initial_y'],
        simulation_state['pos_c1_initial_x'], simulation_state['pos_c1_initial_y']
    )
    simulation_state['v_n2x_real'], simulation_state['v_n2y_real'] = v_n2x, v_n2y
    simulation_state['v_c1_after_n_x_real'], simulation_state['v_c1_after_n_y_real'] = v_c1_an_x, v_c1_an_y
    # El punto de colisión N-C1 es la posición inicial de C1
    simulation_state['collision_point_n_c1_x'] = simulation_state['pos_c1_initial_x']
    simulation_state['collision_point_n_c1_y'] = simulation_state['pos_c1_initial_y']
    
    # Actualizar posición inicial de C2 para que esté en la trayectoria de C1 después del choque N-C1
    v_c1_an_magnitud = np.sqrt(v_c1_an_x**2 + v_c1_an_y**2)
    if v_c1_an_magnitud > 1e-9: # Si C1 se mueve después del choque con N
        dir_c1_x = v_c1_an_x / v_c1_an_magnitud
        dir_c1_y = v_c1_an_y / v_c1_an_magnitud
        # Colocar C2 a una distancia DIST_C1_TO_C2_TARGET a lo largo de la trayectoria de C1
        simulation_state['pos_c2_initial_x'] = simulation_state['collision_point_n_c1_x'] + dir_c1_x * simulation_state['DIST_C1_TO_C2_TARGET']
        simulation_state['pos_c2_initial_y'] = simulation_state['collision_point_n_c1_y'] + dir_c1_y * simulation_state['DIST_C1_TO_C2_TARGET']
    else: # Si C1 no se mueve, C2 permanece en una posición por defecto (o la que tenía)
        simulation_state['pos_c2_initial_x'] = simulation_state['pos_c1_initial_x'] + simulation_state['DIST_C1_TO_C2_TARGET']
        simulation_state['pos_c2_initial_y'] = simulation_state['pos_c1_initial_y']

    # El punto de colisión C1-C2 es la posición inicial (actualizada) de C2
    simulation_state['collision_point_c1_c2_x'] = simulation_state['pos_c2_initial_x']
    simulation_state['collision_point_c1_c2_y'] = simulation_state['pos_c2_initial_y']
    # Actualizar la posición visual de C2 en el gráfico antes de iniciar la animación
    carbon2_line_mpl.set_data([simulation_state['pos_c2_initial_x']], [simulation_state['pos_c2_initial_y']])

    # Calcular tiempo (real, en segundos) hasta la colisión N-C1
    simulation_state['theoretical_time_to_n_c1_collision'] = fs.calcular_tiempo_hasta_colision(
        simulation_state['pos_n_initial_x'], simulation_state['pos_n_initial_y'], # Desde N
        simulation_state['pos_c1_initial_x'], simulation_state['pos_c1_initial_y'], # Hacia C1
        v_n1x, v_n1y # Con velocidad inicial de N
    )

    # Calcular tiempo (real, en segundos) para que C1 alcance C2
    simulation_state['theoretical_time_to_c1_c2_collision'] = fs.calcular_tiempo_hasta_colision(
        simulation_state['collision_point_n_c1_x'], simulation_state['collision_point_n_c1_y'], # C1 parte de aquí
        simulation_state['pos_c2_initial_x'], simulation_state['pos_c2_initial_y'], # Hacia C2
        v_c1_an_x, v_c1_an_y # Con velocidad de C1 post N-C1
    )

    # Calcular velocidades después de la colisión C1-C2 (si ocurre)
    if simulation_state['theoretical_time_to_c1_c2_collision'] != float('inf') and \
       simulation_state['theoretical_time_to_c1_c2_collision'] > 1e-9: # Evitar colisión en tiempo cero o negativo
        v_c1_ac2_x, v_c1_ac2_y, v_c2_ac1_x, v_c2_ac1_y = fs.calcular_colision_c1_c2(
            simulation_state['m_c1'], simulation_state['m_c2'], v_c1_an_x, v_c1_an_y
        )
        simulation_state['v_c1_after_c2_x_real'] = v_c1_ac2_x
        simulation_state['v_c1_after_c2_y_real'] = v_c1_ac2_y
        simulation_state['v_c2_after_c1_x_real'] = v_c2_ac1_x
        simulation_state['v_c2_after_c1_y_real'] = v_c2_ac1_y
    else: # Si no hay colisión C1-C2 (o C1 no se mueve hacia C2)
        simulation_state['v_c1_after_c2_x_real'] = v_c1_an_x # C1 no cambia su velocidad (post N-C1)
        simulation_state['v_c1_after_c2_y_real'] = v_c1_an_y
        simulation_state['v_c2_after_c1_x_real'] = 0.0 # C2 permanece en reposo
        simulation_state['v_c2_after_c1_y_real'] = 0.0

    # --- 4. Cálculos de Parámetros Específicos de Animación ---
    # Convertir velocidades reales a velocidades de animación

    # Vector from Neutron's start to C1's start for initial animation phase
    dx_n_to_c1_visual = simulation_state['pos_c1_initial_x'] - simulation_state['pos_n_initial_x']
    dy_n_to_c1_visual = simulation_state['pos_c1_initial_y'] - simulation_state['pos_n_initial_y']
    dist_n_to_c1_visual_mag = np.sqrt(dx_n_to_c1_visual**2 + dy_n_to_c1_visual**2)

    # Magnitude of the neutron's initial animation velocity (consistent with overall animation speed scaling)
    # v_n1x and v_n1y here refer to the real physics velocities calculated earlier
    v_n1_overall_anim_magnitude = simulation_state['v_n1_magnitud_real'] * ANIMATION_SPEED_FACTOR

    if dist_n_to_c1_visual_mag > 1e-9: # Avoid division by zero
        # Direction for animation is directly towards C1
        dir_n_to_c1_x = dx_n_to_c1_visual / dist_n_to_c1_visual_mag
        dir_n_to_c1_y = dy_n_to_c1_visual / dist_n_to_c1_visual_mag
        simulation_state['v_n1_anim_x'] = dir_n_to_c1_x * v_n1_overall_anim_magnitude
        simulation_state['v_n1_anim_y'] = dir_n_to_c1_y * v_n1_overall_anim_magnitude
    else: # Neutron and C1 start at the same position
        simulation_state['v_n1_anim_x'] = 0.0
        simulation_state['v_n1_anim_y'] = 0.0

    # Subsequent animation velocities are based on post-collision real velocities
    # v_n2x, v_n2y, v_c1_an_x, v_c1_an_y are results from fs.calcular_colision_n_c1
    simulation_state['v_n2_anim_x'] = v_n2x * ANIMATION_SPEED_FACTOR
    simulation_state['v_n2_anim_y'] = v_n2y * ANIMATION_SPEED_FACTOR
    simulation_state['v_c1_after_n_anim_x'] = v_c1_an_x * ANIMATION_SPEED_FACTOR
    simulation_state['v_c1_after_n_anim_y'] = v_c1_an_y * ANIMATION_SPEED_FACTOR
    simulation_state['v_c1_after_c2_anim_x'] = simulation_state['v_c1_after_c2_x_real'] * ANIMATION_SPEED_FACTOR
    simulation_state['v_c1_after_c2_anim_y'] = simulation_state['v_c1_after_c2_y_real'] * ANIMATION_SPEED_FACTOR
    simulation_state['v_c2_after_c1_anim_x'] = simulation_state['v_c2_after_c1_x_real'] * ANIMATION_SPEED_FACTOR
    simulation_state['v_c2_after_c1_anim_y'] = simulation_state['v_c2_after_c1_y_real'] * ANIMATION_SPEED_FACTOR

    # Calcular número de frames hasta la colisión N-C1 (para la animación)
    # Distancia visual que el neutrón debe recorrer hasta C1
    dist_visual_n_to_c1 = np.sqrt(
        (simulation_state['pos_c1_initial_x'] - simulation_state['pos_n_initial_x'])**2 +
        (simulation_state['pos_c1_initial_y'] - simulation_state['pos_n_initial_y'])**2
    )
    v_n1_anim_mag = np.sqrt(simulation_state['v_n1_anim_x']**2 + simulation_state['v_n1_anim_y']**2)
    if v_n1_anim_mag > 1e-9: # Si el neutrón tiene velocidad de animación
        simulation_state['frames_until_n_c1_collision'] = int(dist_visual_n_to_c1 / v_n1_anim_mag)
    else: # Si el neutrón no se mueve en la animación
        simulation_state['frames_until_n_c1_collision'] = 1 # Colisión (o no) ocurre en el primer frame relevante
    # Asegurar al menos 1 frame para la fase, si no es instantáneo
    simulation_state['frames_until_n_c1_collision'] = max(1, simulation_state['frames_until_n_c1_collision'])
    # Tiempo de colisión N-C1 para la gráfica (basado en frames de animación)
    simulation_state['t_col_n_c1_graph'] = simulation_state['frames_until_n_c1_collision'] * (FIXED_ANIMATION_INTERVAL_MS_CONFIG / 1000.0)


    # Calcular número de frames para que C1 alcance C2 (para la animación)
    v_c1_an_anim_mag = np.sqrt(simulation_state['v_c1_after_n_anim_x']**2 + simulation_state['v_c1_after_n_anim_y']**2)
    num_frames_c1_to_c2_anim = 0
    # Solo calcular si hay una colisión C1-C2 física y C1 se mueve hacia C2
    if simulation_state['theoretical_time_to_c1_c2_collision'] != float('inf') and \
       simulation_state['theoretical_time_to_c1_c2_collision'] > 1e-9:
        if v_c1_an_anim_mag > 1e-9: # Si C1 tiene velocidad de animación
            # DIST_C1_TO_C2_TARGET es la distancia visual que C1 debe recorrer
            num_frames_c1_to_c2_anim = int(simulation_state['DIST_C1_TO_C2_TARGET'] / v_c1_an_anim_mag)
            # Si C1 tiene velocidad real pero la de animación es tan alta que da 0 frames, dar al menos 1
            if num_frames_c1_to_c2_anim == 0 and v_c1_an_magnitud > 1e-9:
                num_frames_c1_to_c2_anim = 1
        elif v_c1_an_magnitud > 1e-9: # C1 tiene velocidad real pero no de animación (ANIMATION_SPEED_FACTOR bajo)
            num_frames_c1_to_c2_anim = 1 # Darle al menos 1 frame para la fase

        simulation_state['frames_until_c1_c2_collision'] = simulation_state['frames_until_n_c1_collision'] + num_frames_c1_to_c2_anim
        # Tiempo de colisión C1-C2 para la gráfica (basado en tiempos físicos reales)
        simulation_state['t_col_c1_c2_graph'] = simulation_state['theoretical_time_to_n_c1_collision'] + \
                                               simulation_state['theoretical_time_to_c1_c2_collision']
    else: # No hay colisión C1-C2 programada en la animación
        simulation_state['frames_until_c1_c2_collision'] = simulation_state['total_frames_anim'] + 1 # Marcar como inalcanzable
        simulation_state['t_col_c1_c2_graph'] = -1 # Indicar que no ocurre para la gráfica

    # Calcular y mostrar tiempos teóricos de salida (simplificado)
    # TODO: Refinar estos cálculos usando fs.calcular_tiempo_salida_pantalla para mayor precisión,
    # considerando todas las fases del movimiento de cada partícula y los límites correctos.
    # La lógica actual es una placeholder para mantener la funcionalidad del texto en UI.
    simulation_state['total_theoretical_time_c1'] = float('inf')
    simulation_state['total_theoretical_time_c2'] = float('inf')

    if simulation_state['theoretical_time_to_n_c1_collision'] != float('inf'):
        # Tiempo para C1 salir (muy simplificado)
        # Asumimos que C1 sale después de la colisión N-C1 si no hay C1-C2, o después de C1-C2 si la hay.
        # Esta es una lógica muy básica y necesita ser mejorada con fs.calcular_tiempo_salida_pantalla.
        vel_final_c1_x, vel_final_c1_y = simulation_state['v_c1_after_n_x_real'], simulation_state['v_c1_after_n_y_real']
        pos_inicial_c1_x_para_salida, pos_inicial_c1_y_para_salida = simulation_state['collision_point_n_c1_x'], simulation_state['collision_point_n_c1_y']
        tiempo_base_c1 = simulation_state['theoretical_time_to_n_c1_collision']

        if simulation_state['theoretical_time_to_c1_c2_collision'] != float('inf') and simulation_state['theoretical_time_to_c1_c2_collision'] > 1e-9:
            vel_final_c1_x, vel_final_c1_y = simulation_state['v_c1_after_c2_x_real'], simulation_state['v_c1_after_c2_y_real']
            pos_inicial_c1_x_para_salida, pos_inicial_c1_y_para_salida = simulation_state['collision_point_c1_c2_x'], simulation_state['collision_point_c1_c2_y']
            tiempo_base_c1 += simulation_state['theoretical_time_to_c1_c2_collision']
        
        tiempo_salida_c1 = fs.calcular_tiempo_salida_pantalla(
            pos_inicial_c1_x_para_salida, pos_inicial_c1_y_para_salida, vel_final_c1_x, vel_final_c1_y,
            x_lim_sim[0], x_lim_sim[1], y_lim_sim[0], y_lim_sim[1], margin_for_particle_size_config
        )
        if tiempo_salida_c1 != float('inf'):
            simulation_state['total_theoretical_time_c1'] = tiempo_base_c1 + tiempo_salida_c1

        # Tiempo para C2 salir
        if simulation_state['theoretical_time_to_c1_c2_collision'] != float('inf') and simulation_state['theoretical_time_to_c1_c2_collision'] > 1e-9:
            tiempo_base_c2 = simulation_state['theoretical_time_to_n_c1_collision'] + simulation_state['theoretical_time_to_c1_c2_collision']
            tiempo_salida_c2 = fs.calcular_tiempo_salida_pantalla(
                simulation_state['collision_point_c1_c2_x'], simulation_state['collision_point_c1_c2_y'], # C2 parte del punto de colisión C1-C2
                simulation_state['v_c2_after_c1_x_real'], simulation_state['v_c2_after_c1_y_real'],
                x_lim_sim[0], x_lim_sim[1], y_lim_sim[0], y_lim_sim[1], margin_for_particle_size_config
            )
            if tiempo_salida_c2 != float('inf'):
                simulation_state['total_theoretical_time_c2'] = tiempo_base_c2 + tiempo_salida_c2

    # Actualizar el texto del cronómetro teórico en la UI
    if global_text_theoretical_timer_mpl:
        if simulation_state['total_theoretical_time_c2'] != float('inf') and \
           simulation_state['frames_until_c1_c2_collision'] < simulation_state['total_frames_anim'] : # Si C2 se espera que choque y salga
            global_text_theoretical_timer_mpl.set_text(f"Tiempo Teórico (C2 salida): {simulation_state['total_theoretical_time_c2']:.2e} s")
        elif simulation_state['total_theoretical_time_c1'] != float('inf'): # Si no, mostrar el de C1
            global_text_theoretical_timer_mpl.set_text(f"Tiempo Teórico (C1 salida): {simulation_state['total_theoretical_time_c1']:.2e} s")
        else: # Si ninguno tiene un tiempo de salida calculado
            global_text_theoretical_timer_mpl.set_text('Tiempo Teórico: ∞ s')


    # --- 5. Preparar Parámetros para el Módulo de Animación ---
    # El diccionario `simulation_state` ya contiene la mayoría de los datos necesarios.
    # Se añaden referencias a los objetos de Matplotlib y constantes de configuración de animación.
    anim_params = simulation_state.copy() # Usar una copia para evitar modificaciones inesperadas
    anim_params.update({
        'fig': fig, 'ax': ax_simulacion,
        'neutron_line': neutron_line_mpl, 'carbon1_line': carbon1_line_mpl, 'carbon2_line': carbon2_line_mpl,
        'text_state': text_state_mpl, 'text_neutron_vel': global_text_neutron_vel_mpl,
        'text_carbon_vel': global_text_carbon_vel_mpl, 'text_carbon2_vel': global_text_carbon2_vel_mpl,
        'text_timer': global_text_timer_mpl, 'text_theoretical_timer': global_text_theoretical_timer_mpl,
        'text_center_of_mass': global_text_center_of_mass_mpl, # Pass the new text object
        'm_n': simulation_state['m_n'], # Pass mass of neutron
        'm_c1': simulation_state['m_c1'], # Pass mass of C1
        'm_c2': simulation_state['m_c2'], # Pass mass of C2
        'FIXED_ANIMATION_INTERVAL_MS': FIXED_ANIMATION_INTERVAL_MS_CONFIG, # Pasar la constante
        'margin_for_particle_size': margin_for_particle_size_config, # Pasar la constante
        'x_lim': x_lim_sim, 'y_lim': y_lim_sim, # Pasar los límites de los ejes
        # Las listas para datos de velocidad (`animation_times_list`, etc.) ya están en `simulation_state`
        # y se pasarán por referencia si `anim_params` es una copia superficial o si se actualizan en `simulation_state`.
        # Para asegurar que el módulo de animación escribe en las listas correctas, se pasan explícitamente:
        'animation_times_list': simulation_state['animation_times_list'],
        'neutron_velocidades_escalar_list': simulation_state['neutron_velocidades_escalar_list'],
        'c1_velocidades_escalar_list': simulation_state['c1_velocidades_escalar_list'],
        'c2_velocidades_escalar_list': simulation_state['c2_velocidades_escalar_list'],
    })

    # --- 6. Iniciar Animación ---
    # Llamar a la función del módulo de animación para iniciar el bucle de animación
    anim_sim.start_animation_loop(anim_params)
    if fig: fig.canvas.draw_idle() # Asegurar que la figura se redibuje


# --- Configuración de Widgets de la Interfaz de Usuario (Botones y TextBoxes) ---
# Posicionamiento de widgets
widget_left_col = 0.68 # Columna izquierda para widgets
widget_label_offset_y = 0.05 # Desplazamiento Y para etiquetas de TextBoxes
widget_height = 0.04 # Altura de TextBoxes y Botones
widget_row_spacing = 0.06 # Espaciado vertical entre filas de widgets
initial_bottom_pos = 0.88 # Posición Y (inferior) del primer widget

# TextBox para la Velocidad del Neutrón
fig.text(widget_left_col, initial_bottom_pos + widget_label_offset_y, 'Velocidad Neutrón (m/s):', fontsize=10, ha='left', va='center')
ax_v_n_textbox = fig.add_axes([widget_left_col, initial_bottom_pos, 0.25, widget_height])
textbox_v_n = TextBox(ax_v_n_textbox, '', initial=str(v_n1_magnitud_real_default))

# TextBox para la Masa del Carbono 1
fig.text(widget_left_col, initial_bottom_pos - widget_row_spacing + widget_label_offset_y, 'Masa Carbono 1 (u):', fontsize=10, ha='left', va='center')
ax_m_c1_textbox = fig.add_axes([widget_left_col, initial_bottom_pos - widget_row_spacing, 0.25, widget_height])
textbox_m_c1_input = TextBox(ax_m_c1_textbox, '', initial=str(m_c1_default))

# TextBox para la Masa del Carbono 2
fig.text(widget_left_col, initial_bottom_pos - 2 * widget_row_spacing + widget_label_offset_y, 'Masa Carbono 2 (u):', fontsize=10, ha='left', va='center')
ax_m_c2_textbox = fig.add_axes([widget_left_col, initial_bottom_pos - 2 * widget_row_spacing, 0.25, widget_height])
textbox_m_c2_input = TextBox(ax_m_c2_textbox, '', initial=str(m_c2_default))

# Botón para Iniciar / Reiniciar la Simulación
# Ahora un solo botón que llama a la función orquestadora principal.
ax_start_button = fig.add_axes([widget_left_col, initial_bottom_pos - 3.5 * widget_row_spacing, 0.25, widget_height + 0.01])
start_button = Button(ax_start_button, 'Iniciar / Reiniciar Simulación')
start_button.on_clicked(lambda event: run_simulation_logic())

# Botón para Mostrar Gráfica de Velocidad
def generate_velocity_plot_callback(event):
    """
    Callback para el botón de mostrar gráfica de velocidad.
    Genera y muestra una nueva figura con las velocidades de las partículas vs. tiempo.
    """
    # Asegurar que todas las listas tengan la misma longitud para graficar
    min_len = min(len(simulation_state['animation_times_list']),
                  len(simulation_state['neutron_velocidades_escalar_list']),
                  len(simulation_state['c1_velocidades_escalar_list']),
                  len(simulation_state['c2_velocidades_escalar_list']))

    # Convertir listas a arrays de NumPy para graficar
    times_np = np.array(simulation_state['animation_times_list'][:min_len])
    neutron_vel_np = np.array(simulation_state['neutron_velocidades_escalar_list'][:min_len])
    c1_vel_np = np.array(simulation_state['c1_velocidades_escalar_list'][:min_len])
    c2_vel_np = np.array(simulation_state['c2_velocidades_escalar_list'][:min_len])

    # Crear una nueva figura y ejes para la gráfica de velocidad
    fig_vel, ax_vel = plt.subplots(figsize=(10, 6))
    ax_vel.plot(times_np, neutron_vel_np, label='Neutrón', color='orange')
    ax_vel.plot(times_np, c1_vel_np, label='Carbono 1', color='gray')
    ax_vel.plot(times_np, c2_vel_np, label='Carbono 2', color='blue')

    # Añadir líneas verticales para marcar los tiempos de colisión
    ax_vel.axvline(simulation_state['t_col_n_c1_graph'], color='black', linestyle='--', label='Colisión N-C1')
    # Solo mostrar línea de colisión C1-C2 si ocurrió y es dentro del rango de tiempo de la animación
    if simulation_state['t_col_c1_c2_graph'] > 0 and \
       (len(times_np) == 0 or simulation_state['t_col_c1_c2_graph'] < times_np[-1] if len(times_np)>0 else True) : # Manejar caso de lista vacía
        ax_vel.axvline(simulation_state['t_col_c1_c2_graph'], color='red', linestyle=':', label='Colisión C1-C2')

    ax_vel.set_xlabel("Tiempo (s)")
    ax_vel.set_ylabel("Velocidad (m/s)")
    ax_vel.set_title("Velocidad vs. Tiempo del Choque Elástico")
    ax_vel.legend() # Mostrar leyenda de la gráfica
    ax_vel.grid(True, linestyle='--', alpha=0.7) # Añadir rejilla
    plt.tight_layout() # Ajustar layout para que todo quepa bien
    plt.show() # Mostrar la figura de la gráfica

ax_plot_button = fig.add_axes([widget_left_col, initial_bottom_pos - 4.5 * widget_row_spacing, 0.25, widget_height + 0.01])
plot_button = Button(ax_plot_button, 'Mostrar Gráfica Velocidad')
plot_button.on_clicked(generate_velocity_plot_callback)


# Textos en la UI para mostrar velocidades actuales y timers
velocity_text_bottom_pos = initial_bottom_pos - 6 * widget_row_spacing
global_text_neutron_vel_mpl = fig.text(widget_left_col, velocity_text_bottom_pos, 'Neutrón: ', fontsize=10, ha='left', va='center')
global_text_carbon_vel_mpl = fig.text(widget_left_col, velocity_text_bottom_pos - 0.04, 'Carbono 1: ', fontsize=10, ha='left', va='center')
global_text_carbon2_vel_mpl = fig.text(widget_left_col, velocity_text_bottom_pos - 0.08, 'Carbono 2: ', fontsize=10, ha='left', va='center')
global_text_timer_mpl = fig.text(widget_left_col, velocity_text_bottom_pos - 0.12, 'Tiempo Sim.: 0.00 s', fontsize=10, ha='left', va='center', weight='bold', color='blue')
global_text_theoretical_timer_mpl = fig.text(widget_left_col, velocity_text_bottom_pos - 0.16, 'Tiempo Teórico: N/A', fontsize=10, ha='left', va='center', weight='bold', color='green')
global_text_center_of_mass_mpl = fig.text(widget_left_col, velocity_text_bottom_pos - 0.20, 'CM: (---, ---)', fontsize=10, ha='left', va='center', weight='bold', color='purple')

# Título General de la Figura Principal
fig.suptitle('Simulación de Choque Elástico Refactorizada: Neutrón -> C1 -> C2', fontsize=14, weight='bold', y=0.97)
# Ajustar el layout de la figura principal para que los widgets no se solapen con el área de simulación
plt.tight_layout(rect=[0, 0, widget_left_col - 0.02, 1])

# Mostrar la figura principal con la simulación y controles
plt.show()
