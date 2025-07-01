import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Variables globales para la animación que serán gestionadas por el orquestador.
# Estas se inicializarán o actualizarán desde el módulo principal.
# El módulo de animación las usará para dibujar cada frame.

# Referencias a objetos de Matplotlib (se asignarán desde el orquestador)
fig_anim = None
ax_anim = None
neutron_line, carbon1_line, carbon2_line = None, None, None
text_state_anim = None
global_text_neutron_vel_anim, global_text_carbon_vel_anim, global_text_carbon2_vel_anim = None, None, None
global_text_timer_anim, global_text_theoretical_timer_anim = None, None
current_animation_obj = None # Para almacenar el objeto FuncAnimation

# Estado de la simulación para la animación (se asignarán desde el orquestador)
# Posiciones iniciales
pos_n_initial_x, pos_n_initial_y = 0.0, 0.0
pos_c1_initial_x, pos_c1_initial_y = 0.0, 0.0
pos_c2_initial_x, pos_c2_initial_y = 0.0, 0.0 # Se actualiza dinámicamente

# Puntos de colisión (actualizados por el orquestador basado en la física)
collision_point_n_c1_x, collision_point_n_c1_y = 0.0, 0.0
collision_point_c1_c2_x, collision_point_c1_c2_y = 0.0, 0.0

# Velocidades de animación (calculadas y pasadas por el orquestador)
v_n1_anim_x, v_n1_anim_y = 0.0, 0.0
v_n2_anim_x, v_n2_anim_y = 0.0, 0.0
v_c1_after_n_anim_x, v_c1_after_n_anim_y = 0.0, 0.0
v_c1_after_c2_anim_x, v_c1_after_c2_anim_y = 0.0, 0.0
v_c2_after_c1_anim_x, v_c2_after_c1_anim_y = 0.0, 0.0

# Velocidades reales (para mostrar en texto, pasadas por el orquestador)
v_n1x_real, v_n1y_real = 0.0, 0.0
v_n2x_real, v_n2y_real = 0.0, 0.0
v_c1_initial_x_real, v_c1_initial_y_real = 0.0, 0.0 # C1 en reposo
v_c1_after_n_x_real, v_c1_after_n_y_real = 0.0, 0.0
v_c1_after_c2_x_real, v_c1_after_c2_y_real = 0.0, 0.0
v_c2_initial_x_real, v_c2_initial_y_real = 0.0, 0.0 # C2 en reposo
v_c2_after_c1_x_real, v_c2_after_c1_y_real = 0.0, 0.0

# Tiempos teóricos (para mostrar, pasados por el orquestador)
total_theoretical_time_c1, total_theoretical_time_c2 = float('inf'), float('inf')

# Frames de colisión (calculados y pasados por el orquestador)
frames_until_n_c1_collision = 0
frames_until_c1_c2_collision = 0
total_frames_anim = 1000 # Un valor por defecto, puede ser ajustado

# Constantes de la animación (pasadas o configuradas por el orquestador)
FIXED_ANIMATION_INTERVAL_MS = 20
margin_for_particle_size_anim = 0.05
x_lim_left_anim, x_lim_right_anim = 0.0, 0.0
y_lim_bottom_anim, y_lim_top_anim = 0.0, 0.0

# Listas para la gráfica de velocidad (gestionadas por el orquestador)
animation_times_list = []
neutron_velocidades_escalar_list = []
c1_velocidades_escalar_list = []
c2_velocidades_escalar_list = []


def set_animation_parameters(params):
    """
    Configura los parámetros y el estado necesarios para la animación.
    Llamado por el módulo orquestador antes de iniciar una nueva animación.
    """
    global fig_anim, ax_anim, neutron_line, carbon1_line, carbon2_line
    global text_state_anim, global_text_neutron_vel_anim, global_text_carbon_vel_anim
    global global_text_carbon2_vel_anim, global_text_timer_anim, global_text_theoretical_timer_anim

    global pos_n_initial_x, pos_n_initial_y, pos_c1_initial_x, pos_c1_initial_y, pos_c2_initial_x, pos_c2_initial_y
    global collision_point_n_c1_x, collision_point_n_c1_y, collision_point_c1_c2_x, collision_point_c1_c2_y
    global v_n1_anim_x, v_n1_anim_y, v_n2_anim_x, v_n2_anim_y
    global v_c1_after_n_anim_x, v_c1_after_n_anim_y, v_c1_after_c2_anim_x, v_c1_after_c2_anim_y
    global v_c2_after_c1_anim_x, v_c2_after_c1_anim_y
    global v_n1x_real, v_n1y_real, v_n2x_real, v_n2y_real
    global v_c1_initial_x_real, v_c1_initial_y_real, v_c1_after_n_x_real, v_c1_after_n_y_real
    global v_c1_after_c2_x_real, v_c1_after_c2_y_real, v_c2_initial_x_real, v_c2_initial_y_real
    global v_c2_after_c1_x_real, v_c2_after_c1_y_real
    global total_theoretical_time_c1, total_theoretical_time_c2
    global frames_until_n_c1_collision, frames_until_c1_c2_collision, total_frames_anim
    global FIXED_ANIMATION_INTERVAL_MS, margin_for_particle_size_anim
    global x_lim_left_anim, x_lim_right_anim, y_lim_bottom_anim, y_lim_top_anim
    global animation_times_list, neutron_velocidades_escalar_list, c1_velocidades_escalar_list, c2_velocidades_escalar_list

    # Asignar objetos de Matplotlib
    fig_anim = params['fig']
    ax_anim = params['ax']
    neutron_line = params['neutron_line']
    carbon1_line = params['carbon1_line']
    carbon2_line = params['carbon2_line']
    text_state_anim = params['text_state']
    global_text_neutron_vel_anim = params['text_neutron_vel']
    global_text_carbon_vel_anim = params['text_carbon_vel']
    global_text_carbon2_vel_anim = params['text_carbon2_vel']
    global_text_timer_anim = params['text_timer']
    global_text_theoretical_timer_anim = params['text_theoretical_timer']

    # Asignar estado y parámetros de simulación
    pos_n_initial_x = params['pos_n_initial_x']
    pos_n_initial_y = params['pos_n_initial_y']
    pos_c1_initial_x = params['pos_c1_initial_x']
    pos_c1_initial_y = params['pos_c1_initial_y']
    pos_c2_initial_x = params['pos_c2_initial_x'] # Actualizado por el orquestador
    pos_c2_initial_y = params['pos_c2_initial_y'] # Actualizado por el orquestador

    collision_point_n_c1_x = params['collision_point_n_c1_x']
    collision_point_n_c1_y = params['collision_point_n_c1_y']
    collision_point_c1_c2_x = params['collision_point_c1_c2_x'] # Actualizado
    collision_point_c1_c2_y = params['collision_point_c1_c2_y'] # Actualizado

    v_n1_anim_x = params['v_n1_anim_x']
    v_n1_anim_y = params['v_n1_anim_y']
    v_n2_anim_x = params['v_n2_anim_x']
    v_n2_anim_y = params['v_n2_anim_y']
    v_c1_after_n_anim_x = params['v_c1_after_n_anim_x']
    v_c1_after_n_anim_y = params['v_c1_after_n_anim_y']
    v_c1_after_c2_anim_x = params['v_c1_after_c2_anim_x']
    v_c1_after_c2_anim_y = params['v_c1_after_c2_anim_y']
    v_c2_after_c1_anim_x = params['v_c2_after_c1_anim_x']
    v_c2_after_c1_anim_y = params['v_c2_after_c1_anim_y']

    v_n1x_real = params['v_n1x_real']
    v_n1y_real = params['v_n1y_real']
    v_n2x_real = params['v_n2x_real']
    v_n2y_real = params['v_n2y_real']
    v_c1_initial_x_real = params['v_c1_initial_x_real']
    v_c1_initial_y_real = params['v_c1_initial_y_real']
    v_c1_after_n_x_real = params['v_c1_after_n_x_real']
    v_c1_after_n_y_real = params['v_c1_after_n_y_real']
    v_c1_after_c2_x_real = params['v_c1_after_c2_x_real']
    v_c1_after_c2_y_real = params['v_c1_after_c2_y_real']
    v_c2_initial_x_real = params['v_c2_initial_x_real']
    v_c2_initial_y_real = params['v_c2_initial_y_real']
    v_c2_after_c1_x_real = params['v_c2_after_c1_x_real']
    v_c2_after_c1_y_real = params['v_c2_after_c1_y_real']

    total_theoretical_time_c1 = params['total_theoretical_time_c1']
    total_theoretical_time_c2 = params['total_theoretical_time_c2']

    frames_until_n_c1_collision = params['frames_until_n_c1_collision']
    frames_until_c1_c2_collision = params['frames_until_c1_c2_collision']
    total_frames_anim = params.get('total_frames_anim', 1000) # Usar valor de params o default

    FIXED_ANIMATION_INTERVAL_MS = params['FIXED_ANIMATION_INTERVAL_MS']
    margin_for_particle_size_anim = params['margin_for_particle_size']
    x_lim_left_anim, x_lim_right_anim = params['x_lim']
    y_lim_bottom_anim, y_lim_top_anim = params['y_lim']

    # Referencias a listas para datos de gráfica
    animation_times_list = params['animation_times_list']
    neutron_velocidades_escalar_list = params['neutron_velocidades_escalar_list']
    c1_velocidades_escalar_list = params['c1_velocidades_escalar_list']
    c2_velocidades_escalar_list = params['c2_velocidades_escalar_list']


def init_animation():
    """
    Función de inicialización para la animación. Limpia los elementos de la animación.
    """
    neutron_line.set_data([], [])
    carbon1_line.set_data([], [])
    carbon2_line.set_data([], [])
    text_state_anim.set_text('')
    if global_text_neutron_vel_anim: global_text_neutron_vel_anim.set_text('')
    if global_text_carbon_vel_anim: global_text_carbon_vel_anim.set_text('')
    if global_text_carbon2_vel_anim: global_text_carbon2_vel_anim.set_text('')
    if global_text_timer_anim: global_text_timer_anim.set_text('Tiempo Sim.: 0.00 s')
    if global_text_theoretical_timer_anim: global_text_theoretical_timer_anim.set_text('Tiempo Teórico: Calculando...')

    # Reiniciar la posición de los objetos para la próxima animación (el orquestador debe pasar las pos iniciales correctas)
    neutron_line.set_data([pos_n_initial_x], [pos_n_initial_y])
    carbon1_line.set_data([pos_c1_initial_x], [pos_c1_initial_y])
    carbon2_line.set_data([pos_c2_initial_x], [pos_c2_initial_y])

    return (neutron_line, carbon1_line, carbon2_line, text_state_anim,
            global_text_neutron_vel_anim, global_text_carbon_vel_anim,
            global_text_carbon2_vel_anim, global_text_timer_anim, global_text_theoretical_timer_anim)

def animate_frame(frame):
    """
    Función de animación que se llama para cada frame.
    Calcula y actualiza la posición de las partículas y el texto de estado.
    """
    global current_animation_obj # Necesario para detener la animación desde adentro

    time_in_s = frame * (FIXED_ANIMATION_INTERVAL_MS / 1000.0)
    animation_times_list.append(time_in_s)

    current_pos_n_x, current_pos_n_y = 0.0, 0.0
    current_pos_c1_x, current_pos_c1_y = 0.0, 0.0
    current_pos_c2_x, current_pos_c2_y = 0.0, 0.0

    # Fase 1: Antes de la colisión Neutrón-C1
    if frame <= frames_until_n_c1_collision:
        current_pos_n_x = pos_n_initial_x + v_n1_anim_x * frame
        current_pos_n_y = pos_n_initial_y + v_n1_anim_y * frame

        current_pos_c1_x = pos_c1_initial_x
        current_pos_c1_y = pos_c1_initial_y
        current_pos_c2_x = pos_c2_initial_x
        current_pos_c2_y = pos_c2_initial_y

        # Asegurar que el neutrón no pase el punto de colisión N-C1 (ajuste visual)
        # Se mueve una distancia v_n1_anim_magnitude por frame.
        # Si la distancia restante es menor que un paso, colocarlo en el punto de colisión.
        dist_n_to_c1_sq = (current_pos_n_x - collision_point_n_c1_x)**2 + (current_pos_n_y - collision_point_n_c1_y)**2
        step_dist_sq = (v_n1_anim_x**2 + v_n1_anim_y**2) * 0.25 # umbral pequeño (medio paso)
        if dist_n_to_c1_sq < step_dist_sq and frame < frames_until_n_c1_collision : # Evitar ajuste si ya es el frame de colisión
             pass # Dejar que el cálculo normal lo acerque
        elif frame == frames_until_n_c1_collision: # En el frame exacto de colisión
            current_pos_n_x = collision_point_n_c1_x
            current_pos_n_y = collision_point_n_c1_y


        text_state_anim.set_text('Fase 1: Antes de N-C1')
        if global_text_neutron_vel_anim: global_text_neutron_vel_anim.set_text(f'Neutrón: {np.sqrt(v_n1x_real**2 + v_n1y_real**2):.2e} m/s')
        if global_text_carbon_vel_anim: global_text_carbon_vel_anim.set_text(f'C1: {np.sqrt(v_c1_initial_x_real**2 + v_c1_initial_y_real**2):.2e} m/s')
        if global_text_carbon2_vel_anim: global_text_carbon2_vel_anim.set_text(f'C2: {np.sqrt(v_c2_initial_x_real**2 + v_c2_initial_y_real**2):.2e} m/s')

        neutron_velocidades_escalar_list.append(np.sqrt(v_n1x_real**2 + v_n1y_real**2))
        c1_velocidades_escalar_list.append(np.sqrt(v_c1_initial_x_real**2 + v_c1_initial_y_real**2))
        c2_velocidades_escalar_list.append(np.sqrt(v_c2_initial_x_real**2 + v_c2_initial_y_real**2))

    # Fase 2: Entre colisión N-C1 y C1-C2
    elif frame <= frames_until_c1_c2_collision:
        frame_since_n_c1_collision = (frame - frames_until_n_c1_collision)

        current_pos_n_x = collision_point_n_c1_x + v_n2_anim_x * frame_since_n_c1_collision
        current_pos_n_y = collision_point_n_c1_y + v_n2_anim_y * frame_since_n_c1_collision

        current_pos_c1_x = pos_c1_initial_x + v_c1_after_n_anim_x * frame_since_n_c1_collision
        current_pos_c1_y = pos_c1_initial_y + v_c1_after_n_anim_y * frame_since_n_c1_collision

        current_pos_c2_x = pos_c2_initial_x
        current_pos_c2_y = pos_c2_initial_y

        # Asegurar que C1 no pase el punto de colisión C1-C2 (ajuste visual)
        if frames_until_c1_c2_collision < total_frames_anim :
            dist_c1_to_c2_sq = (current_pos_c1_x - collision_point_c1_c2_x)**2 + (current_pos_c1_y - collision_point_c1_c2_y)**2
            step_dist_c1_sq = (v_c1_after_n_anim_x**2 + v_c1_after_n_anim_y**2) * 0.25
            if dist_c1_to_c2_sq < step_dist_c1_sq and frame < frames_until_c1_c2_collision:
                pass
            elif frame == frames_until_c1_c2_collision: # En el frame exacto de colisión
                current_pos_c1_x = collision_point_c1_c2_x
                current_pos_c1_y = collision_point_c1_c2_y

        text_state_anim.set_text('Fase 2: N-C1 hecho, antes de C1-C2')
        if global_text_neutron_vel_anim: global_text_neutron_vel_anim.set_text(f'Neutrón: {np.sqrt(v_n2x_real**2 + v_n2y_real**2):.2e} m/s')
        if global_text_carbon_vel_anim: global_text_carbon_vel_anim.set_text(f'C1: {np.sqrt(v_c1_after_n_x_real**2 + v_c1_after_n_y_real**2):.2e} m/s')
        if global_text_carbon2_vel_anim: global_text_carbon2_vel_anim.set_text(f'C2: {np.sqrt(v_c2_initial_x_real**2 + v_c2_initial_y_real**2):.2e} m/s')

        neutron_velocidades_escalar_list.append(np.sqrt(v_n2x_real**2 + v_n2y_real**2))
        c1_velocidades_escalar_list.append(np.sqrt(v_c1_after_n_x_real**2 + v_c1_after_n_y_real**2))
        c2_velocidades_escalar_list.append(np.sqrt(v_c2_initial_x_real**2 + v_c2_initial_y_real**2))

    # Fase 3: Después de la colisión C1-C2
    else:
        frame_since_c1_c2_collision = (frame - frames_until_c1_c2_collision)
        frame_since_n_c1_collision_for_n = (frame - frames_until_n_c1_collision)

        current_pos_n_x = collision_point_n_c1_x + v_n2_anim_x * frame_since_n_c1_collision_for_n
        current_pos_n_y = collision_point_n_c1_y + v_n2_anim_y * frame_since_n_c1_collision_for_n

        current_pos_c1_x = collision_point_c1_c2_x + v_c1_after_c2_anim_x * frame_since_c1_c2_collision
        current_pos_c1_y = collision_point_c1_c2_y + v_c1_after_c2_anim_y * frame_since_c1_c2_collision

        current_pos_c2_x = pos_c2_initial_x + v_c2_after_c1_anim_x * frame_since_c1_c2_collision
        current_pos_c2_y = pos_c2_initial_y + v_c2_after_c1_anim_y * frame_since_c1_c2_collision

        text_state_anim.set_text('Fase 3: Después de C1-C2')
        if global_text_neutron_vel_anim: global_text_neutron_vel_anim.set_text(f'Neutrón: {np.sqrt(v_n2x_real**2 + v_n2y_real**2):.2e} m/s')
        if global_text_carbon_vel_anim: global_text_carbon_vel_anim.set_text(f'C1: {np.sqrt(v_c1_after_c2_x_real**2 + v_c1_after_c2_y_real**2):.2e} m/s')
        if global_text_carbon2_vel_anim: global_text_carbon2_vel_anim.set_text(f'C2: {np.sqrt(v_c2_after_c1_x_real**2 + v_c2_after_c1_y_real**2):.2e} m/s')

        neutron_velocidades_escalar_list.append(np.sqrt(v_n2x_real**2 + v_n2y_real**2))
        c1_velocidades_escalar_list.append(np.sqrt(v_c1_after_c2_x_real**2 + v_c1_after_c2_y_real**2))
        c2_velocidades_escalar_list.append(np.sqrt(v_c2_after_c1_x_real**2 + v_c2_after_c1_y_real**2))

    # Actualizar posiciones en el gráfico
    neutron_line.set_data([current_pos_n_x], [current_pos_n_y])
    carbon1_line.set_data([current_pos_c1_x], [current_pos_c1_y])
    carbon2_line.set_data([current_pos_c2_x], [current_pos_c2_y])

    if global_text_timer_anim: global_text_timer_anim.set_text(f'Tiempo Sim.: {time_in_s:.2f} s')

    # --- Lógica de Detención (copiada y adaptada del original, ahora en este módulo) ---
    n_is_off_screen = not (x_lim_left_anim + margin_for_particle_size_anim < current_pos_n_x < x_lim_right_anim - margin_for_particle_size_anim and \
                           y_lim_bottom_anim + margin_for_particle_size_anim < current_pos_n_y < y_lim_top_anim - margin_for_particle_size_anim)
    c1_is_off_screen = not (x_lim_left_anim + margin_for_particle_size_anim < current_pos_c1_x < x_lim_right_anim - margin_for_particle_size_anim and \
                            y_lim_bottom_anim + margin_for_particle_size_anim < current_pos_c1_y < y_lim_top_anim - margin_for_particle_size_anim)
    c2_is_off_screen = not (x_lim_left_anim + margin_for_particle_size_anim < current_pos_c2_x < x_lim_right_anim - margin_for_particle_size_anim and \
                            y_lim_bottom_anim + margin_for_particle_size_anim < current_pos_c2_y < y_lim_top_anim - margin_for_particle_size_anim)

    n_is_active = False
    if not n_is_off_screen:
        if frame < frames_until_n_c1_collision and np.sqrt(v_n1x_real**2 + v_n1y_real**2) >= 1e-9:
            n_is_active = True
        elif frame >= frames_until_n_c1_collision and np.sqrt(v_n2x_real**2 + v_n2y_real**2) >= 1e-9:
            n_is_active = True

    c1_is_active = False
    if not c1_is_off_screen:
        if frame < frames_until_n_c1_collision and np.sqrt(v_n1x_real**2 + v_n1y_real**2) >= 1e-9 :
             c1_is_active = True
        elif frame >= frames_until_c1_c2_collision and np.sqrt(v_c1_after_c2_x_real**2 + v_c1_after_c2_y_real**2) >= 1e-9:
            c1_is_active = True
        elif frame >= frames_until_n_c1_collision and frame < frames_until_c1_c2_collision and \
             np.sqrt(v_c1_after_n_x_real**2 + v_c1_after_n_y_real**2) >= 1e-9:
            c1_is_active = True

    c2_is_active = False
    if not c2_is_off_screen:
        if frame < frames_until_c1_c2_collision and frames_until_c1_c2_collision < total_frames_anim and \
           np.sqrt(v_c1_after_n_x_real**2 + v_c1_after_n_y_real**2) >= 1e-9:
            c2_is_active = True
        elif frame >= frames_until_c1_c2_collision and np.sqrt(v_c2_after_c1_x_real**2 + v_c2_after_c1_y_real**2) >= 1e-9:
            c2_is_active = True

    should_stop_animation = False
    if (n_is_off_screen or not n_is_active) and \
       (c1_is_off_screen or not c1_is_active) and \
       (c2_is_off_screen or not c2_is_active):
        should_stop_animation = True
        if n_is_off_screen and c1_is_off_screen and c2_is_off_screen:
            text_state_anim.set_text('¡Animación Terminada! Todas las partículas fuera de pantalla.')
        else:
            text_state_anim.set_text('¡Animación Terminada! Partículas estáticas o fuera de pantalla.')

    if should_stop_animation:
        if current_animation_obj is not None and current_animation_obj.event_source is not None:
            current_animation_obj.event_source.stop()
            if global_text_timer_anim: global_text_timer_anim.set_text(f'Tiempo Sim.: {time_in_s:.2f} s (Final)')
            if global_text_theoretical_timer_anim:
                if total_theoretical_time_c2 != float('inf') and frames_until_c1_c2_collision < total_frames_anim :
                    global_text_theoretical_timer_anim.set_text(f'Tiempo Teórico (C2 salida): {total_theoretical_time_c2:.2e} s')
                elif total_theoretical_time_c1 != float('inf'):
                    global_text_theoretical_timer_anim.set_text(f'Tiempo Teórico (C1 salida): {total_theoretical_time_c1:.2e} s')
                else:
                    global_text_theoretical_timer_anim.set_text('Tiempo Teórico: ∞ s')
            if fig_anim: fig_anim.canvas.draw_idle()

    return (neutron_line, carbon1_line, carbon2_line, text_state_anim,
            global_text_neutron_vel_anim, global_text_carbon_vel_anim,
            global_text_carbon2_vel_anim, global_text_timer_anim, global_text_theoretical_timer_anim)

def start_animation_loop(params):
    """
    Inicia el bucle de animación de Matplotlib.
    'params' es un diccionario que contiene todos los parámetros necesarios
    configurados por el orquestador.
    Devuelve el objeto FuncAnimation.
    """
    global current_animation_obj
    set_animation_parameters(params) # Configurar todas las variables globales del módulo

    # Detener cualquier animación anterior si existe y está corriendo
    if current_animation_obj is not None and current_animation_obj.event_source is not None:
        current_animation_obj.event_source.stop()
        current_animation_obj = None

    # Crear y asignar la nueva animación
    current_animation_obj = animation.FuncAnimation(fig_anim, animate_frame, frames=total_frames_anim,
                                                    init_func=init_animation, blit=False,
                                                    interval=FIXED_ANIMATION_INTERVAL_MS, repeat=False)
    if fig_anim: fig_anim.canvas.draw_idle()
    return current_animation_obj

def stop_current_animation_if_running():
    """Detiene la animación actual si está en curso."""
    global current_animation_obj
    if current_animation_obj is not None and current_animation_obj.event_source is not None:
        current_animation_obj.event_source.stop()
        # current_animation_obj = None # Podría ser necesario si se quiere permitir reiniciar limpiamente.
                                     # Por ahora, el orquestador maneja la creación de nuevos objetos.

# Otras funciones relacionadas con la animación podrían ir aquí,
# por ejemplo, funciones para actualizar tamaños de marcador si cambian dinámicamente, etc.
