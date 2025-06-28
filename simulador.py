import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, TextBox
import matplotlib
matplotlib.use('TkAgg') # Asegura que la ventana de la animación se muestre

# --- Datos del Problema (Variables Globales que se pueden modificar) ---
v_n1_magnitud_real = 2.6e7 # Magnitud de la velocidad inicial del neutrón
m_c1 = 12.0 # Masa inicial del carbono 1 (anteriormente m_c)
m_c2 = 12.0 # Masa inicial del carbono 2
m_n = 1.0 # La masa del neutrón se mantiene constante según el problema original

# --- Ángulo de la trayectoria diagonal del neutrón (en radianes) ---
# Por ejemplo, 45 grados (pi/4). El neutrón se moverá hacia arriba y a la derecha.
angulo_neutron = np.pi / 4

# --- Distancia deseada para colocar C2 en la trayectoria de C1 ---
DIST_C1_TO_C2_TARGET = 0.3

# --- Componentes de la velocidad inicial del neutrón ---
v_n1x_real = v_n1_magnitud_real * np.cos(angulo_neutron)
v_n1y_real = v_n1_magnitud_real * np.sin(angulo_neutron)


# --- Cálculos para las velocidades después del choque (se recalcularán para 2D) ---
# C1 (Carbono 1)
v_c1_initial_x_real = 0.0 # Renombrado de v_c1x_real
v_c1_initial_y_real = 0.0 # Renombrado de v_c1y_real. C1 inicialmente en reposo.
v_c1_after_n_x_real = 0.0 # Velocidad de C1 después del choque con Neutrón
v_c1_after_n_y_real = 0.0
v_c1_after_c2_x_real = 0.0 # Velocidad de C1 después del choque con C2
v_c1_after_c2_y_real = 0.0

# Neutrón
v_n2x_real = 0.0 # Velocidad del Neutrón después del choque con C1
v_n2y_real = 0.0

# C2 (Carbono 2)
v_c2_initial_x_real = 0.0 # C2 inicialmente en reposo
v_c2_initial_y_real = 0.0
v_c2_after_c1_x_real = 0.0 # Velocidad de C2 después del choque con C1
v_c2_after_c1_y_real = 0.0


# --- Factor de escala para el tamaño visual del carbono ---
CARBON_MARKER_SCALE_FACTOR = 1.66
MIN_CARBON_MARKERSIZE = 5

# --- Función para escalar el tamaño del carbono ---
def get_carbon_markersize(mass_c):
    return max(MIN_CARBON_MARKERSIZE, mass_c * CARBON_MARKER_SCALE_FACTOR)

# --- Configuración de la Animación ---
fig = plt.figure(figsize=(12, 8)) # Aumentar altura para trayectoria diagonal

ax = fig.add_axes([0.08, 0.1, 0.55, 0.8])

# Ajustar los límites X para dar un poco más de "espacio de salida"
ax.set_xlim(-0.1, 1.7)
x_lim_left = ax.get_xlim()[0]
x_lim_right = ax.get_xlim()[1]

# Ajustar los límites Y para la trayectoria diagonal
ax.set_ylim(-0.1, 1.0) # Ajustar según sea necesario para la trayectoria
y_lim_bottom = ax.get_ylim()[0]
y_lim_top = ax.get_ylim()[1]

ax.set_aspect('equal')

# Posiciones iniciales
pos_n_initial_x = 0.1
pos_n_initial_y = 0.1 # Neutrón comienza desde (0.1, 0.1) para la diagonal
pos_c1_initial_x = 0.8
pos_c1_initial_y = 0.2 # C1 más bajo

# Posición inicial de C2: se colocará en la trayectoria esperada de C1 después del choque con el neutrón.
# Esto es una estimación y podría necesitar ajuste.
# Por ahora, la colocaremos un poco más adelante en x y a la misma y que C1 (antes del choque N-C1)
# La lógica exacta para su posición se refinará en run_animation_logic o cuando se calcule la trayectoria de C1.
pos_c2_initial_x = pos_c1_initial_x + 0.3 # Un poco más a la derecha de C1
pos_c2_initial_y = pos_c1_initial_y       # A la misma altura inicial que C1 (esto se ajustará)


# Punto de colisión N-C1 (posición inicial de C1)
collision_point_n_c1_x = pos_c1_initial_x
collision_point_n_c1_y = pos_c1_initial_y

# Punto de colisión C1-C2 (posición inicial de C2, se actualizará dinámicamente)
collision_point_c1_c2_x = pos_c2_initial_x
collision_point_c1_c2_y = pos_c2_initial_y


neutron, = ax.plot([], [], 'o', color='orange', markersize=15, label='Neutrón', zorder=2)
carbon1, = ax.plot([], [], 'o', color='gray', markersize=get_carbon_markersize(m_c1), label='Carbono 1 (C1)', zorder=2) # Usa m_c1
carbon2, = ax.plot([], [], 'o', color='blue', markersize=get_carbon_markersize(m_c2), label='Carbono 2 (C2)', zorder=2) # Nuevo: Carbono 2, usa m_c2

text_state = ax.text(0.4, 0.99, '', transform=ax.transAxes, fontsize=12, weight='bold', ha='center', va='top')

# Ejes X e Y
ax.axhline(0, color='black', linewidth=0.5, zorder=1) # Eje X
ax.axvline(0, color='black', linewidth=0.5, zorder=1) # Eje Y
ax.arrow(0.0, 0, x_lim_right - 0.2, 0, head_width=0.03, head_length=0.05, fc='black', ec='black', zorder=1)
ax.text(x_lim_right - 0.15, -0.05, 'X', fontsize=12, zorder=1)
ax.arrow(0, 0.0, 0, y_lim_top - 0.1, head_width=0.03, head_length=0.05, fc='black', ec='black', zorder=1)
ax.text(-0.05, y_lim_top - 0.05, 'Y', fontsize=12, zorder=1)

ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.95))

# --- Variables globales para los objetos Text de velocidad, cronómetro de animación y cronómetro teórico ---
global_text_neutron_vel = None
global_text_carbon_vel = None
global_text_timer = None
global_text_theoretical_timer = None # <--- NUEVO: Objeto de texto para el cronómetro teórico

# --- Parámetros de Duración Visual de las Fases (en frames) ---
frames_despues_choque_fijos = 300
final_hold_frames = 100

# --- Escalado de Velocidades para la Animación Visual (se recalcularán) ---
v_n1_anim_x, v_n1_anim_y = 0.0, 0.0 # Neutrón antes de N-C1
v_n2_anim_x, v_n2_anim_y = 0.0, 0.0 # Neutrón después de N-C1

v_c1_after_n_anim_x, v_c1_after_n_anim_y = 0.0, 0.0 # C1 después de N-C1
v_c1_after_c2_anim_x, v_c1_after_c2_anim_y = 0.0, 0.0 # C1 después de C1-C2

v_c2_after_c1_anim_x, v_c2_after_c1_anim_y = 0.0, 0.0 # C2 después de C1-C2


# --- Listas para almacenar datos de velocidad para la gráfica ---
animation_times = []
neutron_velocidades_escalar = [] # Renombrado neutron_velocities
c1_velocidades_escalar = []    # Renombrado carbon_velocities, ahora para C1
c2_velocidades_escalar = []    # Nueva lista para C2

# --- Variables globales para la duración total y tiempo de colisión en segundos ---
t_col_n_c1_graph = 0.0 # Renombrado t_col_graph para la primera colisión
t_col_c1_c2_graph = 0.0 # Tiempo de la segunda colisión para la gráfica

# --- Variable global para la instancia de animación ---
current_animation = None

# --- CONSTANTE PARA LA SENSIBILIDAD DE LA ANIMACIÓN ---
ANIMATION_SPEED_FACTOR = 2.69e-10

# --- Intervalo de animación fijo ---
FIXED_ANIMATION_INTERVAL_MS = 20

# --- Variable para almacenar el número de frames hasta la colisión ---
frames_until_n_c1_collision = 0 # Renombrado frames_until_collision
frames_until_c1_c2_collision = 0 # Nueva variable para la segunda colisión
total_frames = 20000 # Puede necesitar ajuste

# --- NUEVAS VARIABLES GLOBALES PARA TIEMPOS TEÓRICOS ---
# Para N-C1
theoretical_time_to_n_c1_collision = 0.0 # Renombrado
theoretical_time_post_n_c1_collision_c1 = 0.0 # Renombrado (tiempo para C1 salir)
total_theoretical_time_c1 = 0.0 # Renombrado

# Para C1-C2 (a implementar más adelante)
theoretical_time_to_c1_c2_collision = 0.0
theoretical_time_post_c1_c2_collision_c2 = 0.0 # Tiempo para C2 salir
total_theoretical_time_c2 = 0.0


# --- Variables globales para los objetos Text de C2 ---
global_text_carbon2_vel = None # Para la velocidad de C2

# --- Función de Inicialización de la Animación (limpia) ---
def init():
    neutron.set_data([], [])
    carbon1.set_data([], [])
    carbon2.set_data([], []) # Inicializar C2
    text_state.set_text('')
    if global_text_neutron_vel is not None:
        global_text_neutron_vel.set_text('')
    if global_text_carbon_vel is not None:
        global_text_carbon_vel.set_text('')
    if global_text_carbon2_vel is not None: # Limpiar texto de C2
        global_text_carbon2_vel.set_text('')
    if global_text_timer is not None:
        global_text_timer.set_text('Tiempo: 0.00 s')
    if global_text_theoretical_timer is not None:
        global_text_theoretical_timer.set_text('Tiempo Teórico (C1): Calculando...')
    
    # Reiniciar la posición de los objetos para la próxima animación
    neutron.set_data([pos_n_initial_x], [pos_n_initial_y])
    carbon1.set_data([pos_c1_initial_x], [pos_c1_initial_y])
    carbon2.set_data([pos_c2_initial_x], [pos_c2_initial_y]) # Posición inicial de C2
    
    return neutron, carbon1, carbon2, text_state, global_text_neutron_vel, global_text_carbon_vel, global_text_carbon2_vel, global_text_timer, global_text_theoretical_timer

# --- Función de Animación (MODIFICADA para trayectoria diagonal) ---
margin_for_particle_size = 0.05
def animate(frame):
    # Declarar globales para todas las velocidades y masas necesarias, incluyendo C2
    global m_c1, m_c2, \
           v_n1x_real, v_n1y_real, \
           v_n2x_real, v_n2y_real, \
           v_c1_initial_x_real, v_c1_initial_y_real, v_c1_after_n_x_real, v_c1_after_n_y_real, v_c1_after_c2_x_real, v_c1_after_c2_y_real, \
           v_c2_initial_x_real, v_c2_initial_y_real, v_c2_after_c1_x_real, v_c2_after_c1_y_real, \
           v_n1_anim_x, v_n1_anim_y, v_n2_anim_x, v_n2_anim_y, \
           v_c1_after_n_anim_x, v_c1_after_n_anim_y, v_c1_after_c2_anim_x, v_c1_after_c2_anim_y, \
           v_c2_after_c1_anim_x, v_c2_after_c1_anim_y, \
           t_col_n_c1_graph, t_col_c1_c2_graph, global_text_neutron_vel, global_text_carbon_vel, global_text_carbon2_vel, global_text_timer, \
           global_text_theoretical_timer, frames_until_n_c1_collision, frames_until_c1_c2_collision, current_animation, \
           total_theoretical_time_c1, total_theoretical_time_c2, collision_point_c1_c2_x, collision_point_c1_c2_y


    time_in_s = frame * (FIXED_ANIMATION_INTERVAL_MS / 1000.0)
    animation_times.append(time_in_s)

    current_pos_n_x, current_pos_n_y = 0.0, 0.0
    current_pos_c1_x, current_pos_c1_y = 0.0, 0.0
    current_pos_c2_x, current_pos_c2_y = 0.0, 0.0 # Posiciones para C2


    # Fase 1: Antes de la colisión Neutrón-C1
    if frame <= frames_until_n_c1_collision:
        # Movimiento del neutrón antes de la colisión (diagonal hacia el punto de colisión)
        # La velocidad del neutrón debe dirigirse hacia (collision_point_x, collision_point_y)
        # Esto se maneja en run_animation_logic al calcular v_n1_anim_x, v_n1_anim_y
        current_pos_n_x = pos_n_initial_x + v_n1_anim_x * frame
        current_pos_n_y = pos_n_initial_y + v_n1_anim_y * frame
        
        # Carbono 1 permanece en su posición inicial
        current_pos_c1_x = pos_c1_initial_x
        current_pos_c1_y = pos_c1_initial_y
        current_pos_c2_x = pos_c2_initial_x # C2 también permanece en su posición inicial
        current_pos_c2_y = pos_c2_initial_y

        # Asegurar que el neutrón no pase el punto de colisión N-C1
        if np.sqrt((current_pos_n_x - collision_point_n_c1_x)**2 + (current_pos_n_y - collision_point_n_c1_y)**2) < np.sqrt(v_n1_anim_x**2 + v_n1_anim_y**2) * 0.5 :
            current_pos_n_x = collision_point_n_c1_x
            current_pos_n_y = collision_point_n_c1_y

        text_state.set_text('Fase 1: Antes de N-C1')
        if global_text_neutron_vel: global_text_neutron_vel.set_text(f'Neutrón: {np.sqrt(v_n1x_real**2 + v_n1y_real**2):.2e} m/s')
        if global_text_carbon_vel: global_text_carbon_vel.set_text(f'C1: {np.sqrt(v_c1_initial_x_real**2 + v_c1_initial_y_real**2):.2e} m/s') # C1 en reposo
        if global_text_carbon2_vel: global_text_carbon2_vel.set_text(f'C2: {np.sqrt(v_c2_initial_x_real**2 + v_c2_initial_y_real**2):.2e} m/s') # C2 en reposo

        neutron_velocidades_escalar.append(np.sqrt(v_n1x_real**2 + v_n1y_real**2))
        c1_velocidades_escalar.append(np.sqrt(v_c1_initial_x_real**2 + v_c1_initial_y_real**2))
        c2_velocidades_escalar.append(np.sqrt(v_c2_initial_x_real**2 + v_c2_initial_y_real**2))

    # Fase 2: Entre colisión N-C1 y C1-C2 (o fin si no hay C1-C2)
    elif frame <= frames_until_c1_c2_collision: # frames_until_c1_c2_collision será muy grande si no hay colisión C1-C2
        frame_since_n_c1_collision = (frame - frames_until_n_c1_collision)

        # Neutrón se mueve con su velocidad post N-C1
        current_pos_n_x = collision_point_n_c1_x + v_n2_anim_x * frame_since_n_c1_collision
        current_pos_n_y = collision_point_n_c1_y + v_n2_anim_y * frame_since_n_c1_collision

        # C1 se mueve con su velocidad post N-C1 (antes de golpear C2)
        current_pos_c1_x = pos_c1_initial_x + v_c1_after_n_anim_x * frame_since_n_c1_collision
        current_pos_c1_y = pos_c1_initial_y + v_c1_after_n_anim_y * frame_since_n_c1_collision

        # C2 permanece en reposo hasta que C1 lo golpea
        current_pos_c2_x = pos_c2_initial_x
        current_pos_c2_y = pos_c2_initial_y

        # Asegurar que C1 no pase el punto de colisión C1-C2 (si está definido y es alcanzable)
        if frames_until_c1_c2_collision < total_frames : # Solo si hay una colisión C1-C2 programada
             if np.sqrt((current_pos_c1_x - collision_point_c1_c2_x)**2 + (current_pos_c1_y - collision_point_c1_c2_y)**2) < np.sqrt(v_c1_after_n_anim_x**2 + v_c1_after_n_anim_y**2) * 0.5 :
                current_pos_c1_x = collision_point_c1_c2_x
                current_pos_c1_y = collision_point_c1_c2_y


        text_state.set_text('Fase 2: N-C1 hecho, antes de C1-C2')
        if global_text_neutron_vel: global_text_neutron_vel.set_text(f'Neutrón: {np.sqrt(v_n2x_real**2 + v_n2y_real**2):.2e} m/s')
        if global_text_carbon_vel: global_text_carbon_vel.set_text(f'C1: {np.sqrt(v_c1_after_n_x_real**2 + v_c1_after_n_y_real**2):.2e} m/s')
        if global_text_carbon2_vel: global_text_carbon2_vel.set_text(f'C2: {np.sqrt(v_c2_initial_x_real**2 + v_c2_initial_y_real**2):.2e} m/s') # C2 aún en reposo

        neutron_velocidades_escalar.append(np.sqrt(v_n2x_real**2 + v_n2y_real**2))
        c1_velocidades_escalar.append(np.sqrt(v_c1_after_n_x_real**2 + v_c1_after_n_y_real**2))
        c2_velocidades_escalar.append(np.sqrt(v_c2_initial_x_real**2 + v_c2_initial_y_real**2))

    # Fase 3: Después de la colisión C1-C2 (si ocurre)
    else:
        frame_since_c1_c2_collision = (frame - frames_until_c1_c2_collision)
        frame_since_n_c1_collision_for_n = (frame - frames_until_n_c1_collision) # Neutrón sigue su camino

        # Neutrón continúa con su velocidad post N-C1
        current_pos_n_x = collision_point_n_c1_x + v_n2_anim_x * frame_since_n_c1_collision_for_n
        current_pos_n_y = collision_point_n_c1_y + v_n2_anim_y * frame_since_n_c1_collision_for_n

        # C1 se mueve con su velocidad post C1-C2
        # Su movimiento parte del punto de colisión C1-C2
        current_pos_c1_x = collision_point_c1_c2_x + v_c1_after_c2_anim_x * frame_since_c1_c2_collision
        current_pos_c1_y = collision_point_c1_c2_y + v_c1_after_c2_anim_y * frame_since_c1_c2_collision

        # C2 se mueve con su velocidad post C1-C2
        # Su movimiento parte de su posición inicial (que es el punto de colisión C1-C2)
        current_pos_c2_x = pos_c2_initial_x + v_c2_after_c1_anim_x * frame_since_c1_c2_collision
        current_pos_c2_y = pos_c2_initial_y + v_c2_after_c1_anim_y * frame_since_c1_c2_collision

        text_state.set_text('Fase 3: Después de C1-C2')
        if global_text_neutron_vel: global_text_neutron_vel.set_text(f'Neutrón: {np.sqrt(v_n2x_real**2 + v_n2y_real**2):.2e} m/s')
        if global_text_carbon_vel: global_text_carbon_vel.set_text(f'C1: {np.sqrt(v_c1_after_c2_x_real**2 + v_c1_after_c2_y_real**2):.2e} m/s')
        if global_text_carbon2_vel: global_text_carbon2_vel.set_text(f'C2: {np.sqrt(v_c2_after_c1_x_real**2 + v_c2_after_c1_y_real**2):.2e} m/s')

        neutron_velocidades_escalar.append(np.sqrt(v_n2x_real**2 + v_n2y_real**2))
        c1_velocidades_escalar.append(np.sqrt(v_c1_after_c2_x_real**2 + v_c1_after_c2_y_real**2))
        c2_velocidades_escalar.append(np.sqrt(v_c2_after_c1_x_real**2 + v_c2_after_c1_y_real**2))


    neutron.set_data([current_pos_n_x], [current_pos_n_y])
    carbon1.set_data([current_pos_c1_x], [current_pos_c1_y])
    carbon2.set_data([current_pos_c2_x], [current_pos_c2_y]) # Actualizar C2


    # Actualizar el texto del cronómetro de animación
    if global_text_timer is not None:
        global_text_timer.set_text(f'Tiempo Sim.: {time_in_s:.2f} s') # Renombrado para claridad

    # LÓGICA DE DETENCIÓN AJUSTADA para 2D (considerar límites en X e Y)
    # LÓGICA DE DETENCIÓN: Prioriza la salida de C2.
    stop_animation = False
    
    c2_is_off_screen = not (x_lim_left + margin_for_particle_size < current_pos_c2_x < x_lim_right - margin_for_particle_size and \
                            y_lim_bottom + margin_for_particle_size < current_pos_c2_y < y_lim_top - margin_for_particle_size)

    if c2_is_off_screen:
        text_state.set_text('¡Animación Terminada! Carbono 2 fuera de pantalla.')
        stop_animation = True
    else:
        # C2 is still on screen.
        # Determine if C2 is expected to move meaningfully in the future.
        c2_expected_to_move = False
        if frame < frames_until_c1_c2_collision: # C1-C2 collision hasn't happened yet
            if frames_until_c1_c2_collision < total_frames: # And it's expected to happen
                 c2_expected_to_move = True # because C1 is heading towards it
        elif np.sqrt(v_c2_after_c1_x_real**2 + v_c2_after_c1_y_real**2) >= 1e-9: # Collision happened, and C2 got some speed
            c2_expected_to_move = True

        if not c2_expected_to_move:
            # If C2 is not going to move (or has finished its significant movement),
            # then Neutron or C1 leaving the screen can stop the animation.
            n_is_off_screen = not (x_lim_left + margin_for_particle_size < current_pos_n_x < x_lim_right - margin_for_particle_size and \
                                   y_lim_bottom + margin_for_particle_size < current_pos_n_y < y_lim_top - margin_for_particle_size)
            c1_is_off_screen = not (x_lim_left + margin_for_particle_size < current_pos_c1_x < x_lim_right - margin_for_particle_size and \
                                    y_lim_bottom + margin_for_particle_size < current_pos_c1_y < y_lim_top - margin_for_particle_size)

            if n_is_off_screen and c1_is_off_screen:
                 text_state.set_text('¡Animación Terminada! N y C1 fuera (C2 no se mueve).')
                 stop_animation = True
            elif n_is_off_screen:
                # Check if C1 is also effectively static or already off screen
                c1_static_or_off = c1_is_off_screen or \
                                   (frame >= frames_until_n_c1_collision and np.sqrt(v_c1_after_n_x_real**2 + v_c1_after_n_y_real**2) < 1e-9 and \
                                    (frame >= frames_until_c1_c2_collision or np.sqrt(v_c1_after_c2_x_real**2 + v_c1_after_c2_y_real**2) < 1e-9) )
                if c1_static_or_off:
                    text_state.set_text('¡Animación Terminada! Neutrón fuera (C1 y C2 no se mueven).')
                    stop_animation = True
            elif c1_is_off_screen:
                 # Check if N is also effectively static or already off screen
                n_static_or_off = n_is_off_screen or \
                                  (frame >= frames_until_n_c1_collision and np.sqrt(v_n2x_real**2 + v_n2y_real**2) < 1e-9)
                if n_static_or_off:
                    text_state.set_text('¡Animación Terminada! C1 fuera (N y C2 no se mueven).')
                    stop_animation = True

    if stop_animation:
        if current_animation is not None and current_animation.event_source is not None:
            current_animation.event_source.stop()
            if global_text_timer: global_text_timer.set_text(f'Tiempo Sim.: {time_in_s:.2f} s (Final)')
            if global_text_theoretical_timer: # Display the relevant theoretical time
                if total_theoretical_time_c2 != float('inf') and frames_until_c1_c2_collision < total_frames : # If C2 path was calculated
                    global_text_theoretical_timer.set_text(f'Tiempo Teórico (C2 salida): {total_theoretical_time_c2:.2e} s')
                elif total_theoretical_time_c1 != float('inf'):
                    global_text_theoretical_timer.set_text(f'Tiempo Teórico (C1 salida): {total_theoretical_time_c1:.2e} s')
                else:
                    global_text_theoretical_timer.set_text('Tiempo Teórico: ∞ s')
            fig.canvas.draw_idle()

    return neutron, carbon1, carbon2, text_state, global_text_neutron_vel, global_text_carbon_vel, global_text_carbon2_vel, global_text_timer, global_text_theoretical_timer

# --- Función para generar la gráfica de velocidad ---
def generate_velocity_plot(event):
    min_len = min(len(animation_times), len(neutron_velocidades_escalar), len(c1_velocidades_escalar), len(c2_velocidades_escalar))
    times_np = np.array(animation_times[:min_len])
    neutron_vel_np = np.array(neutron_velocidades_escalar[:min_len])
    c1_vel_np = np.array(c1_velocidades_escalar[:min_len])
    c2_vel_np = np.array(c2_velocidades_escalar[:min_len])


    fig_vel, ax_vel = plt.subplots(figsize=(10, 6))

    ax_vel.plot(times_np, neutron_vel_np, label='Neutrón', color='orange')
    ax_vel.plot(times_np, c1_vel_np, label='Carbono 1', color='gray')
    ax_vel.plot(times_np, c2_vel_np, label='Carbono 2', color='blue') # Añadir C2 a la gráfica

    ax_vel.axvline(t_col_n_c1_graph, color='black', linestyle='--', label='Colisión N-C1')
    if t_col_c1_c2_graph > 0 and t_col_c1_c2_graph < times_np[-1] if len(times_np)>0 else False: # Solo mostrar si ocurrió
        ax_vel.axvline(t_col_c1_c2_graph, color='red', linestyle=':', label='Colisión C1-C2')

    ax_vel.set_xlabel("Tiempo (s)")
    ax_vel.set_ylabel("Velocidad (m/s)")
    ax_vel.set_title("Velocidad vs. Tiempo del Choque Elástico")
    ax_vel.legend()
    ax_vel.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# --- Función para iniciar o reiniciar la animación (MODIFICADA para 2D) ---
def run_animation_logic():
    # Actualizar lista de globales para incluir C2 y sus colisiones
    global v_n1_magnitud_real, m_c1, m_c2, angulo_neutron, \
           v_n1x_real, v_n1y_real, \
           v_c1_initial_x_real, v_c1_initial_y_real, \
           v_c2_initial_x_real, v_c2_initial_y_real, \
           v_n2x_real, v_n2y_real, \
           v_c1_after_n_x_real, v_c1_after_n_y_real, v_c1_after_c2_x_real, v_c1_after_c2_y_real, \
           v_c2_after_c1_x_real, v_c2_after_c1_y_real, \
           v_n1_anim_x, v_n1_anim_y, v_n2_anim_x, v_n2_anim_y, \
           v_c1_after_n_anim_x, v_c1_after_n_anim_y, v_c1_after_c2_anim_x, v_c1_after_c2_anim_y, \
           v_c2_after_c1_anim_x, v_c2_after_c1_anim_y, \
           t_col_n_c1_graph, t_col_c1_c2_graph, current_animation, \
           frames_until_n_c1_collision, frames_until_c1_c2_collision, total_frames, \
           theoretical_time_to_n_c1_collision, theoretical_time_post_n_c1_collision_c1, total_theoretical_time_c1, \
           theoretical_time_to_c1_c2_collision, theoretical_time_post_c1_c2_collision_c2, total_theoretical_time_c2, \
           pos_c1_initial_x, pos_c1_initial_y, pos_c2_initial_x, pos_c2_initial_y, \
           collision_point_n_c1_x, collision_point_n_c1_y, collision_point_c1_c2_x, collision_point_c1_c2_y


    # Detener cualquier animación en curso de forma segura
    if current_animation is not None and current_animation.event_source is not None:
        current_animation.event_source.stop()
        current_animation = None

    # Limpiar los datos de la animación anterior
    animation_times.clear()
    neutron_velocidades_escalar.clear() # Usa nuevos nombres de lista
    c1_velocidades_escalar.clear()
    c2_velocidades_escalar.clear()
    
    # Reiniciar el texto del cronómetro de animación
    if global_text_timer is not None:
        global_text_timer.set_text('Tiempo Sim.: 0.00 s')
    
    # <--- NUEVO: Reiniciar el cronómetro teórico y calcularlo ---
    if global_text_theoretical_timer is not None:
        global_text_theoretical_timer.set_text('Tiempo Teórico: Calculando...')


    try:
        new_v_n_magnitud_real = float(textbox_v_n.text)
        new_m_c1 = float(textbox_m_c1_input.text) # Changed textbox_m_c to textbox_m_c1_input
        new_m_c2 = float(textbox_m_c2_input.text) # New input for C2 mass

        if new_m_c1 <= 0:
            raise ValueError("Masa C1 debe ser > 0.")
        if new_m_c2 <= 0: # Check for C2 mass
            raise ValueError("Masa C2 debe ser > 0.")
        if new_v_n_magnitud_real <= 0:
             raise ValueError("La magnitud de la velocidad del neutrón debe ser mayor que cero.")

        v_n1_magnitud_real = new_v_n_magnitud_real
        m_c1 = new_m_c1
        m_c2 = new_m_c2 # Assign to global m_c2

        # Recalcular componentes de velocidad inicial del neutrón con la nueva magnitud y el ángulo fijo
        v_n1x_real = v_n1_magnitud_real * np.cos(angulo_neutron)
        v_n1y_real = v_n1_magnitud_real * np.sin(angulo_neutron)
        
        # Carbono 1 y 2 inicialmente en reposo
        v_c1_initial_x_real = 0.0
        v_c1_initial_y_real = 0.0
        v_c2_initial_x_real = 0.0 # C2 initial velocity
        v_c2_initial_y_real = 0.0 # C2 initial velocity


        carbon1.set_markersize(get_carbon_markersize(m_c1))
        carbon2.set_markersize(get_carbon_markersize(m_c2)) # Set C2 marker size

        # --- Colisión Elástica 2D (Neutrón - C1) ---
        # Conservación del momento lineal en X e Y
        # m_n*v_n1x + m_c*v_c1x = m_n*v_n2x + m_c*v_c2x
        # m_n*v_n1y + m_c*v_c1y = m_n*v_n2y + m_c*v_c2y
        # Conservación de la energía cinética (para choque elástico)
        # 0.5*m_n*v_n1^2 + 0.5*m_c*v_c1^2 = 0.5*m_n*v_n2^2 + 0.5*m_c*v_c2^2
        # donde v^2 = vx^2 + vy^2

        # Simplificación: El carbono está inicialmente en reposo (v_c1x = 0, v_c1y = 0)
        # Las ecuaciones para un choque elástico 2D con una partícula en reposo son más complejas
        # que el caso 1D. Aquí se usa una simplificación común para dispersión,
        # asumiendo que el ángulo de dispersión del neutrón es conocido o se puede parametrizar.
        # Para una trayectoria FIJA diagonal y un choque con un blanco en reposo en esa línea,
        # el problema se puede tratar como un choque frontal 1D a lo largo de la línea de colisión.
        # Sin embargo, el enunciado pide "choque elástico", lo que implica que las partículas
        # pueden desviarse.
        #
        # Si la trayectoria del neutrón es FIJA, esto significa que después del choque, el neutrón
        # DEBE continuar en la misma dirección diagonal o rebotar en la dirección opuesta.
        # Esto es una restricción fuerte.
        #
        # Si el neutrón DEBE seguir la misma línea diagonal (o su opuesta):
        # Podemos proyectar las velocidades sobre la línea de colisión (dada por angulo_neutron)
        # y tratarlo como un choque 1D a lo largo de esa línea.
        # Vector director de la línea de colisión: u_coll = (cos(angulo_neutron), sin(angulo_neutron))
        
        # Velocidad inicial del neutrón a lo largo de la línea de colisión:
        v_n1_paralelo = v_n1x_real * np.cos(angulo_neutron) + v_n1y_real * np.sin(angulo_neutron)
        # (Esto es igual a v_n1_magnitud_real ya que el neutrón se mueve a lo largo de esta línea)
        
        # Velocidad inicial del carbono a lo largo de la línea de colisión (es 0):
        v_c1_paralelo = 0 

        # Colisión Elástica 2D (Neutrón - C1)
        # C1 (m_c1) está inicialmente en reposo.
        # El neutrón (m_n) tiene una velocidad inicial v_n1_magnitud_real.
        # Su dirección de movimiento inicial es angulo_neutron (v_n1x_real, v_n1y_real).

        # La colisión N-C1 ocurre en collision_point_n_c1_x, collision_point_n_c1_y (la posición inicial de C1).
        # Determinar el ángulo de la línea de centros en el momento del impacto N-C1.
        dx_approach_n_c1 = collision_point_n_c1_x - pos_n_initial_x
        dy_approach_n_c1 = collision_point_n_c1_y - pos_n_initial_y
        
        if dx_approach_n_c1 == 0 and dy_approach_n_c1 == 0:
            if v_n1_magnitud_real > 0:
                 angle_of_impact_n_c1 = np.arctan2(v_n1y_real, v_n1x_real)
            else:
                 angle_of_impact_n_c1 = 0
        else:
            angle_of_impact_n_c1 = np.arctan2(dy_approach_n_c1, dx_approach_n_c1)

        # Componentes de la velocidad del neutrón ANTES del choque N-C1, a lo largo de la línea de impacto N-C1.
        v_n1_parallel_impact_n_c1 = v_n1x_real * np.cos(angle_of_impact_n_c1) + v_n1y_real * np.sin(angle_of_impact_n_c1)
        v_n1_perp_impact_n_c1 = -v_n1x_real * np.sin(angle_of_impact_n_c1) + v_n1y_real * np.cos(angle_of_impact_n_c1)
        
        # C1 está en reposo.

        # Calcular velocidades DESPUÉS del choque N-C1 a lo largo de la línea de impacto N-C1.
        # Usar m_c1 para la masa de Carbono 1.
        v_n2_parallel_impact_n_c1 = ((m_n - m_c1) / (m_n + m_c1)) * v_n1_parallel_impact_n_c1
        v_c1_after_n_parallel_impact = (2 * m_n / (m_n + m_c1)) * v_n1_parallel_impact_n_c1

        # Las componentes perpendiculares a la línea de impacto no cambian.
        v_n2_perp_impact_n_c1 = v_n1_perp_impact_n_c1 # Corrected variable name
        v_c1_after_n_perp_impact = 0 # Era 0 para C1 y permanece 0.

        # Convertir velocidades post-choque N-C1 de nuevo a coordenadas X, Y.
        # v_n2_perp_impact_n_c1 (which is v_n1_perp_impact_n_c1) is used here:
        v_n2x_real = v_n2_parallel_impact_n_c1 * np.cos(angle_of_impact_n_c1) - v_n2_perp_impact_n_c1 * np.sin(angle_of_impact_n_c1)
        v_n2y_real = v_n2_parallel_impact_n_c1 * np.sin(angle_of_impact_n_c1) + v_n2_perp_impact_n_c1 * np.cos(angle_of_impact_n_c1)

        v_c1_after_n_x_real = v_c1_after_n_parallel_impact * np.cos(angle_of_impact_n_c1) # Perp es 0 para C1
        v_c1_after_n_y_real = v_c1_after_n_parallel_impact * np.sin(angle_of_impact_n_c1)

        # --- Actualizar posición inicial de C2 para estar en la trayectoria de C1 ---
        v_c1_after_n_magnitud = np.sqrt(v_c1_after_n_x_real**2 + v_c1_after_n_y_real**2)
        if v_c1_after_n_magnitud > 1e-9: # Si C1 se mueve después del primer choque
            dir_c1_x = v_c1_after_n_x_real / v_c1_after_n_magnitud
            dir_c1_y = v_c1_after_n_y_real / v_c1_after_n_magnitud

            # Colocar C2 a lo largo de la trayectoria de C1 desde el punto de colisión N-C1
            pos_c2_initial_x = collision_point_n_c1_x + dir_c1_x * DIST_C1_TO_C2_TARGET
            pos_c2_initial_y = collision_point_n_c1_y + dir_c1_y * DIST_C1_TO_C2_TARGET

            # Actualizar el punto de colisión C1-C2 para que sea esta nueva posición de C2
            collision_point_c1_c2_x = pos_c2_initial_x
            collision_point_c1_c2_y = pos_c2_initial_y

            # Actualizar el objeto visual C2 con su nueva posición inicial antes de que la animación comience
            carbon2.set_data([pos_c2_initial_x], [pos_c2_initial_y])
        else:
            # Si C1 no se mueve, C2 permanece en su posición por defecto (o podría marcarse como no colisionable)
            # La lógica de colisión C1-C2 actual ya maneja esto (time_to_c1_c2_collision_sec será inf)
            pass

        # --- Configuración de la Animación (N-C1) ---
        # Velocidad de animación del neutrón ANTES del choque N-C1: debe dirigirse hacia C1.
        dist_visual_to_n_c1_collision = np.sqrt(dx_approach_n_c1**2 + dy_approach_n_c1**2)

        if dist_visual_to_n_c1_collision == 0:
            v_n1_anim_x = 0
            v_n1_anim_y = 0
            frames_until_n_c1_collision = 1
        else:
            # Dirección de animación del neutrón hacia C1
            dir_anim_n_to_c1_x = dx_approach_n_c1 / dist_visual_to_n_c1_collision
            dir_anim_n_to_c1_y = dy_approach_n_c1 / dist_visual_to_n_c1_collision

            # Magnitud de la velocidad de animación basada en la velocidad real inicial del neutrón
            v_n1_anim_magnitude = v_n1_magnitud_real * ANIMATION_SPEED_FACTOR
            v_n1_anim_x = v_n1_anim_magnitude * dir_anim_n_to_c1_x
            v_n1_anim_y = v_n1_anim_magnitude * dir_anim_n_to_c1_y

            current_v_n1_anim_mag_calc = np.sqrt(v_n1_anim_x**2 + v_n1_anim_y**2)
            if current_v_n1_anim_mag_calc > 1e-9:
                frames_until_n_c1_collision = int(dist_visual_to_n_c1_collision / current_v_n1_anim_mag_calc)
            else:
                frames_until_n_c1_collision = 1

        # Velocidades de animación POST-N-C1-choque
        v_n2_anim_x = v_n2x_real * ANIMATION_SPEED_FACTOR
        v_n2_anim_y = v_n2y_real * ANIMATION_SPEED_FACTOR
        v_c1_after_n_anim_x = v_c1_after_n_x_real * ANIMATION_SPEED_FACTOR # Ya usa la variable global correcta
        v_c1_after_n_anim_y = v_c1_after_n_y_real * ANIMATION_SPEED_FACTOR # Ya usa la variable global correcta

        frames_until_n_c1_collision = max(1, frames_until_n_c1_collision)

        # --- CÁLCULOS DEL TIEMPO TEÓRICO REAL para N-C1 ---
        # 1. Tiempo hasta la colisión N-C1
        distance_to_n_c1_collision_real = np.sqrt((pos_c1_initial_x - pos_n_initial_x)**2 + (pos_c1_initial_y - pos_n_initial_y)**2)

        if v_n1_magnitud_real != 0:
            theoretical_time_to_n_c1_collision = distance_to_n_c1_collision_real / v_n1_magnitud_real
        else:
            theoretical_time_to_n_c1_collision = float('inf')

        # 2. Tiempo post-colisión N-C1 (para Carbono 1, C1, hasta salir de pantalla o chocar con C2)
        # Esto se refinará después para incluir la colisión C1-C2. Por ahora, solo salida de C1.
        
        time_c1_exit_x_right = float('inf')
        if v_c1_after_n_x_real > 0:
            dist_c1_exit_x_right = (x_lim_right - margin_for_particle_size) - pos_c1_initial_x
            if dist_c1_exit_x_right > 0 :
                 time_c1_exit_x_right = dist_c1_exit_x_right / v_c1_after_n_x_real
        
        time_c1_exit_x_left = float('inf')
        if v_c1_after_n_x_real < 0:
            dist_c1_exit_x_left = pos_c1_initial_x - (x_lim_left + margin_for_particle_size)
            if dist_c1_exit_x_left > 0:
                time_c1_exit_x_left = dist_c1_exit_x_left / abs(v_c1_after_n_x_real)

        time_c1_exit_y_top = float('inf')
        if v_c1_after_n_y_real > 0:
            dist_c1_exit_y_top = (y_lim_top - margin_for_particle_size) - pos_c1_initial_y
            if dist_c1_exit_y_top > 0:
                time_c1_exit_y_top = dist_c1_exit_y_top / v_c1_after_n_y_real
        
        time_c1_exit_y_bottom = float('inf')
        if v_c1_after_n_y_real < 0:
            dist_c1_exit_y_bottom = pos_c1_initial_y - (y_lim_bottom + margin_for_particle_size)
            if dist_c1_exit_y_bottom > 0:
                time_c1_exit_y_bottom = dist_c1_exit_y_bottom / abs(v_c1_after_n_y_real)

        # El tiempo post-colisión es el mínimo de estos tiempos de salida para C1
        possible_exit_times_c1 = [t for t in [time_c1_exit_x_right, time_c1_exit_x_left, time_c1_exit_y_top, time_c1_exit_y_bottom] if t > 0]
        if not possible_exit_times_c1:
            theoretical_time_post_n_c1_collision_c1 = float('inf')
        else:
            theoretical_time_post_n_c1_collision_c1 = min(possible_exit_times_c1)

        if theoretical_time_to_n_c1_collision != float('inf') and theoretical_time_post_n_c1_collision_c1 != float('inf'):
            total_theoretical_time_c1 = theoretical_time_to_n_c1_collision + theoretical_time_post_n_c1_collision_c1
        else:
            total_theoretical_time_c1 = float('inf')

        # Actualizar el temporizador teórico para C1 (C2 se manejará después)
        if global_text_theoretical_timer is not None:
            if total_theoretical_time_c1 == float('inf'):
                global_text_theoretical_timer.set_text('Tiempo Teórico (C1): ∞ s')
            else:
                global_text_theoretical_timer.set_text(f'Tiempo Teórico (C1): {total_theoretical_time_c1:.2e} s')

        # --- Colisión C1-C2 ---
        # C1 (masa m_c1) tiene velocidad (v_c1_after_n_x_real, v_c1_after_n_y_real)
        # C2 (masa m_c2) está en reposo en (pos_c2_initial_x, pos_c2_initial_y)

        # Calcular tiempo hasta colisión C1-C2
        # Vector de C1 (desde su pos inicial post N-C1, que es collision_point_n_c1) hacia C2 (pos_c2_initial)
        # Velocidad relativa de C1 respecto a C2 (que está en reposo) es (v_c1_after_n_x_real, v_c1_after_n_y_real)

        # Para que haya colisión, C1 debe moverse hacia C2.
        # Si v_c1_after_n_x_real es muy pequeño y C2 está lejos en X, puede no haber colisión.

        # Punto de partida de C1 para esta fase: collision_point_n_c1_x, collision_point_n_c1_y
        # Punto de destino (C2): pos_c2_initial_x, pos_c2_initial_y

        dx_c1_to_c2 = pos_c2_initial_x - collision_point_n_c1_x # C1 parte de donde fue golpeado por N
        dy_c1_to_c2 = pos_c2_initial_y - collision_point_n_c1_y
        dist_c1_to_c2_sq = dx_c1_to_c2**2 + dy_c1_to_c2**2

        # Proyección de la velocidad de C1 sobre el vector que une C1 y C2
        dot_product_c1_c2 = v_c1_after_n_x_real * dx_c1_to_c2 + v_c1_after_n_y_real * dy_c1_to_c2

        time_to_c1_c2_collision_sec = float('inf')
        actual_collision_point_c1_c2_x = pos_c2_initial_x # Por defecto, si no se mueve C1 o no hay colisión
        actual_collision_point_c1_c2_y = pos_c2_initial_y

        if dot_product_c1_c2 > 0: # C1 se mueve hacia C2
            # Tiempo para que C1 alcance la línea perpendicular a su trayectoria que pasa por C2
            # Esto es una simplificación para colisión frontal. Para colisión de esferas es más complejo.
            # Asumimos que C1 debe llegar a la posición (pos_c2_initial_x, pos_c2_initial_y)
            # Esto es válido si C2 es un punto y C1 se mueve directamente hacia él.

            # Para una colisión más realista de "discos", necesitamos calcular el tiempo hasta que sus bordes se toquen.
            # Simplificación: C1 choca con el centro de C2.
            # C1 se mueve desde (collision_point_n_c1_x, collision_point_n_c1_y) con velocidad v_c1_after_n

            # Calcular el tiempo 't' tal que:
            # C1_x(t) = collision_point_n_c1_x + v_c1_after_n_x_real * t
            # C1_y(t) = collision_point_n_c1_y + v_c1_after_n_y_real * t
            # Queremos que (C1_x(t), C1_y(t)) sea (pos_c2_initial_x, pos_c2_initial_y)
            # Esto solo funciona si C1 se dirige exactamente a C2.

            # Si v_c1_after_n_x_real no es cero:
            if abs(v_c1_after_n_x_real) > 1e-9:
                time_tx = (pos_c2_initial_x - collision_point_n_c1_x) / v_c1_after_n_x_real
            else:
                time_tx = float('inf') if pos_c2_initial_x != collision_point_n_c1_x else 0 # Si ya está en la X correcta

            if abs(v_c1_after_n_y_real) > 1e-9:
                time_ty = (pos_c2_initial_y - collision_point_n_c1_y) / v_c1_after_n_y_real
            else:
                time_ty = float('inf') if pos_c2_initial_y != collision_point_n_c1_y else 0 # Si ya está en la Y correcta

            # Si los tiempos son consistentes y positivos, hay colisión en ese punto.
            if time_tx != float('inf') and time_ty != float('inf') and abs(time_tx - time_ty) < 1e-9 and time_tx > 0:
                 time_to_c1_c2_collision_sec = time_tx
                 actual_collision_point_c1_c2_x = pos_c2_initial_x
                 actual_collision_point_c1_c2_y = pos_c2_initial_y
            elif time_tx == float('inf') and time_ty > 0: # Movimiento puramente vertical
                 time_to_c1_c2_collision_sec = time_ty
                 actual_collision_point_c1_c2_x = pos_c2_initial_x # C1 debe estar en la X de C2
                 actual_collision_point_c1_c2_y = pos_c2_initial_y
            elif time_ty == float('inf') and time_tx > 0: # Movimiento puramente horizontal
                 time_to_c1_c2_collision_sec = time_tx
                 actual_collision_point_c1_c2_x = pos_c2_initial_x
                 actual_collision_point_c1_c2_y = pos_c2_initial_y # C1 debe estar en la Y de C2
            else: # No se dirige directamente al centro de C2. Para este ejemplo, no habrá colisión C1-C2.
                 time_to_c1_c2_collision_sec = float('inf')

        # Actualizar el punto de colisión global para C1-C2 si se va a producir
        if time_to_c1_c2_collision_sec != float('inf'):
            collision_point_c1_c2_x = actual_collision_point_c1_c2_x
            collision_point_c1_c2_y = actual_collision_point_c1_c2_y
        else: # No hay colisión C1-C2, C1 y C2 continúan con sus velocidades pre-C1-C2
            collision_point_c1_c2_x = -1000 # Un valor que no se alcanzará, para la animación
            collision_point_c1_c2_y = -1000


        if time_to_c1_c2_collision_sec != float('inf') and time_to_c1_c2_collision_sec > 1e-9: # Evitar colisión en el mismo frame
            # Hay una colisión C1-C2
            frames_until_c1_c2_collision = frames_until_n_c1_collision + int(time_to_c1_c2_collision_sec / (FIXED_ANIMATION_INTERVAL_MS / 1000.0))
            t_col_c1_c2_graph = frames_until_c1_c2_collision * (FIXED_ANIMATION_INTERVAL_MS / 1000.0)

            # Calcular velocidades post C1-C2 colisión
            # C1 tiene velocidad (v_c1_after_n_x_real, v_c1_after_n_y_real)
            # C2 está en reposo (0,0)
            # Masas m_c1, m_c2
            # Ángulo de impacto para C1-C2 es la dirección de C1 (v_c1_after_n) si C2 es un punto.
            # O, si C1 choca con C2 en pos_c2_initial, la línea de centros es de C1 a C2.

            # C1 está en (collision_point_c1_c2_x, collision_point_c1_c2_y) en el momento del impacto C1-C2
            # Pero su velocidad es v_c1_after_n.
            # C2 está en (pos_c2_initial_x, pos_c2_initial_y)
            # La línea de centros es desde la posición de C1 en el impacto hacia C2.
            # Dado que C1 se mueve hacia C2, angle_of_impact_c1_c2 es la dirección de v_c1_after_n

            angle_of_impact_c1_c2 = np.arctan2(v_c1_after_n_y_real, v_c1_after_n_x_real)

            # Proyectar velocidad de C1 (v_c1_after_n) sobre esta línea de impacto
            v_c1_an_parallel = v_c1_after_n_x_real * np.cos(angle_of_impact_c1_c2) + v_c1_after_n_y_real * np.sin(angle_of_impact_c1_c2)
            v_c1_an_perp = -v_c1_after_n_x_real * np.sin(angle_of_impact_c1_c2) + v_c1_after_n_y_real * np.cos(angle_of_impact_c1_c2)

            # C2 está en reposo, sus componentes son 0.

            # Fórmulas 1D elásticas para componentes paralelas
            v_c1_ac2_parallel = ((m_c1 - m_c2) / (m_c1 + m_c2)) * v_c1_an_parallel
            v_c2_ac1_parallel = (2 * m_c1 / (m_c1 + m_c2)) * v_c1_an_parallel

            # Componentes perpendiculares no cambian
            v_c1_ac2_perp = v_c1_an_perp
            v_c2_ac1_perp = 0.0

            # Convertir de nuevo a X,Y
            v_c1_after_c2_x_real = v_c1_ac2_parallel * np.cos(angle_of_impact_c1_c2) - v_c1_ac2_perp * np.sin(angle_of_impact_c1_c2)
            v_c1_after_c2_y_real = v_c1_ac2_parallel * np.sin(angle_of_impact_c1_c2) + v_c1_ac2_perp * np.cos(angle_of_impact_c1_c2)
            v_c2_after_c1_x_real = v_c2_ac1_parallel * np.cos(angle_of_impact_c1_c2) # Perp es 0
            v_c2_after_c1_y_real = v_c2_ac1_parallel * np.sin(angle_of_impact_c1_c2)

            # Velocidades de animación post C1-C2
            v_c1_after_c2_anim_x = v_c1_after_c2_x_real * ANIMATION_SPEED_FACTOR
            v_c1_after_c2_anim_y = v_c1_after_c2_y_real * ANIMATION_SPEED_FACTOR
            v_c2_after_c1_anim_x = v_c2_after_c1_x_real * ANIMATION_SPEED_FACTOR
            v_c2_after_c1_anim_y = v_c2_after_c1_y_real * ANIMATION_SPEED_FACTOR

        else: # No hay colisión C1-C2
            frames_until_c1_c2_collision = total_frames + 1
            t_col_c1_c2_graph = -1
            # C1 y C2 continúan con sus velocidades previas (C1 post N-C1, C2 en reposo)
            v_c1_after_c2_x_real, v_c1_after_c2_y_real = v_c1_after_n_x_real, v_c1_after_n_y_real
            v_c2_after_c1_x_real, v_c2_after_c1_y_real = v_c2_initial_x_real, v_c2_initial_y_real # que es 0,0

            v_c1_after_c2_anim_x, v_c1_after_c2_anim_y = v_c1_after_n_anim_x, v_c1_after_n_anim_y
            v_c2_after_c1_anim_x, v_c2_after_c1_anim_y = v_c2_initial_x_real * ANIMATION_SPEED_FACTOR, v_c2_initial_y_real * ANIMATION_SPEED_FACTOR

        # --- Theoretical Time for C2 exit ---
        # This depends on whether C1-C2 collision happened.
        # theoretical_time_to_c1_c2_collision is time_to_c1_c2_collision_sec

        if time_to_c1_c2_collision_sec != float('inf'):
            # Tiempo para C2 salir de pantalla después de ser golpeado por C1
            time_c2_exit_x_right = float('inf')
            if v_c2_after_c1_x_real > 0:
                dist_c2_exit_x_right = (x_lim_right - margin_for_particle_size) - pos_c2_initial_x
                if dist_c2_exit_x_right > 0: time_c2_exit_x_right = dist_c2_exit_x_right / v_c2_after_c1_x_real

            time_c2_exit_x_left = float('inf')
            if v_c2_after_c1_x_real < 0:
                dist_c2_exit_x_left = pos_c2_initial_x - (x_lim_left + margin_for_particle_size)
                if dist_c2_exit_x_left > 0: time_c2_exit_x_left = dist_c2_exit_x_left / abs(v_c2_after_c1_x_real)

            time_c2_exit_y_top = float('inf')
            if v_c2_after_c1_y_real > 0:
                dist_c2_exit_y_top = (y_lim_top - margin_for_particle_size) - pos_c2_initial_y
                if dist_c2_exit_y_top > 0: time_c2_exit_y_top = dist_c2_exit_y_top / v_c2_after_c1_y_real

            time_c2_exit_y_bottom = float('inf')
            if v_c2_after_c1_y_real < 0:
                dist_c2_exit_y_bottom = pos_c2_initial_y - (y_lim_bottom + margin_for_particle_size)
                if dist_c2_exit_y_bottom > 0: time_c2_exit_y_bottom = dist_c2_exit_y_bottom / abs(v_c2_after_c1_y_real)

            possible_exit_times_c2 = [t for t in [time_c2_exit_x_right, time_c2_exit_x_left, time_c2_exit_y_top, time_c2_exit_y_bottom] if t > 0]
            if not possible_exit_times_c2:
                theoretical_time_post_c1_c2_collision_c2 = float('inf')
            else:
                theoretical_time_post_c1_c2_collision_c2 = min(possible_exit_times_c2)

            # Total theoretical time for C2 to exit screen involves all stages
            if theoretical_time_to_n_c1_collision != float('inf') and \
               time_to_c1_c2_collision_sec != float('inf') and \
               theoretical_time_post_c1_c2_collision_c2 != float('inf'):
                total_theoretical_time_c2 = theoretical_time_to_n_c1_collision + \
                                            time_to_c1_c2_collision_sec + \
                                            theoretical_time_post_c1_c2_collision_c2
            else:
                total_theoretical_time_c2 = float('inf')

            # Update UI text for theoretical time, prioritizing C2 if it collides and exits
            if global_text_theoretical_timer is not None:
                 if total_theoretical_time_c2 != float('inf'):
                    global_text_theoretical_timer.set_text(f'Tiempo Teórico (C2 salida): {total_theoretical_time_c2:.2e} s')
                 # If C2 doesn't exit but C1 does, C1's exit time is already set.
                 # If C2 collides but doesn't exit, C1's exit time is still the primary one shown unless C2's is also inf.
                 elif total_theoretical_time_c1 != float('inf'):
                    global_text_theoretical_timer.set_text(f'Tiempo Teórico (C1 salida): {total_theoretical_time_c1:.2e} s')
                 else: # Both are infinity
                    global_text_theoretical_timer.set_text('Tiempo Teórico: ∞ s')
        else: # No C1-C2 collision, C2's total theoretical time is effectively infinite from this path
            total_theoretical_time_c2 = float('inf')
            # The UI for C1's exit time is already set correctly in this case.

        total_frames = 1000
        t_col_n_c1_graph = frames_until_n_c1_collision * (FIXED_ANIMATION_INTERVAL_MS / 1000.0)


        current_animation = animation.FuncAnimation(fig, animate, frames=total_frames,
                                                  init_func=init, blit=False, interval=FIXED_ANIMATION_INTERVAL_MS, repeat=False)
        fig.canvas.draw_idle()

    except ValueError as e:
        print(f"Error en los valores de entrada: {e}")
        text_state.set_text(f"ERROR: {e}")
        fig.canvas.draw_idle()

# --- Funciones de callback para los botones ---
def on_start_button_clicked(event):
    run_animation_logic()

def on_restart_button_clicked(event):
    run_animation_logic()

# --- Configuración de los Widgets de Entrada y Botones ---
widget_left_col = 0.68
widget_label_offset_y = 0.05
widget_height = 0.04
widget_row_spacing = 0.06 # Reducido para hacer espacio
initial_bottom_pos = 0.88 # Ajustado para subir todo

# V Neutrón (Magnitud)
fig.text(widget_left_col, initial_bottom_pos + widget_label_offset_y, 'Velocidad Neutrón (m/s):',
         fontsize=10, ha='left', va='center')
ax_v_n_textbox = fig.add_axes([widget_left_col, initial_bottom_pos, 0.25, widget_height])
textbox_v_n = TextBox(ax_v_n_textbox, '', initial=str(v_n1_magnitud_real))

# Masa Carbono 1
fig.text(widget_left_col, initial_bottom_pos - widget_row_spacing + widget_label_offset_y, 'Masa Carbono 1 (u):',
         fontsize=10, ha='left', va='center')
ax_m_c1_textbox = fig.add_axes([widget_left_col, initial_bottom_pos - widget_row_spacing, 0.25, widget_height])
textbox_m_c1_input = TextBox(ax_m_c1_textbox, '', initial=str(m_c1)) # Renombrado textbox_m_c

# Masa Carbono 2
fig.text(widget_left_col, initial_bottom_pos - 2 * widget_row_spacing + widget_label_offset_y, 'Masa Carbono 2 (u):',
         fontsize=10, ha='left', va='center')
ax_m_c2_textbox = fig.add_axes([widget_left_col, initial_bottom_pos - 2 * widget_row_spacing, 0.25, widget_height])
textbox_m_c2_input = TextBox(ax_m_c2_textbox, '', initial=str(m_c2))


# Botón Iniciar Animación
ax_start_button = fig.add_axes([widget_left_col, initial_bottom_pos - 3.5 * widget_row_spacing, 0.25, widget_height + 0.01]) # Ajustado
start_button = Button(ax_start_button, 'Iniciar Animación')
start_button.on_clicked(on_start_button_clicked)

# Botón Reiniciar Animación
ax_restart_button = fig.add_axes([widget_left_col, initial_bottom_pos - 4.5 * widget_row_spacing, 0.25, widget_height + 0.01]) # Ajustado
restart_button = Button(ax_restart_button, 'Reiniciar Animación')
restart_button.on_clicked(on_restart_button_clicked)

# Botón Mostrar Gráfica
ax_plot_button = fig.add_axes([widget_left_col, initial_bottom_pos - 5.5 * widget_row_spacing, 0.25, widget_height + 0.01]) # Ajustado
plot_button = Button(ax_plot_button, 'Mostrar Gráfica')
plot_button.on_clicked(generate_velocity_plot)

# --- OBJETOS DE TEXTO PARA LAS VELOCIDADES, CRONÓMETRO DE ANIMACIÓN Y CRONÓMETRO TEÓRICO ---
velocity_text_bottom_pos = initial_bottom_pos - 7 * widget_row_spacing # Ajustado

global_text_neutron_vel = fig.text(widget_left_col, velocity_text_bottom_pos,
                                   'Neutrón: ', fontsize=10, ha='left', va='center')
global_text_carbon_vel = fig.text(widget_left_col, velocity_text_bottom_pos - 0.04, # Label for C1
                                  'Carbono 1: ', fontsize=10, ha='left', va='center')
global_text_carbon2_vel = fig.text(widget_left_col, velocity_text_bottom_pos - 0.08, # Nuevo para C2
                                   'Carbono 2: ', fontsize=10, ha='left', va='center')

global_text_timer = fig.text(widget_left_col, velocity_text_bottom_pos - 0.12, # Ajustar posición
                             'Tiempo Sim.: 0.00 s', fontsize=10, ha='left', va='center', weight='bold', color='blue')

global_text_theoretical_timer = fig.text(widget_left_col, velocity_text_bottom_pos - 0.16, # Ajustar posición
                                         'Tiempo Teórico (C1): Calculando...', fontsize=10, ha='left', va='center', weight='bold', color='green')


# --- Título General de la Figura (centrado arriba) ---
fig.suptitle('Simulación de Choque Elástico: Neutrón -> C1 -> C2', fontsize=14, weight='bold', y=0.97) # Título actualizado

ax.set_yticks([])

plt.tight_layout(rect=[0, 0, widget_left_col - 0.02, 1])

plt.show()