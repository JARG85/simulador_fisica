import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, TextBox
import matplotlib
matplotlib.use('TkAgg') # Asegura que la ventana de la animación se muestre

# --- Datos del Problema (Variables Globales que se pueden modificar) ---
v_n1x_real = 2.6e7 # Velocidad inicial del neutrón
m_c = 12.0 # Masa inicial del carbono
m_n = 1.0 # La masa del neutrón se mantiene constante según el problema original

# --- Cálculos para las velocidades después del choque (se recalcularán) ---
v_c1x_real = 0.0
v_n2x_real = 0.0
v_c2x_real = 0.0

# --- Factor de escala para el tamaño visual del carbono ---
CARBON_MARKER_SCALE_FACTOR = 1.66
MIN_CARBON_MARKERSIZE = 5

# --- Función para escalar el tamaño del carbono ---
def get_carbon_markersize(mass_c):
    return max(MIN_CARBON_MARKERSIZE, mass_c * CARBON_MARKER_SCALE_FACTOR)

# --- Configuración de la Animación ---
fig = plt.figure(figsize=(12, 6))

ax = fig.add_axes([0.08, 0.1, 0.55, 0.8])

# Ajustar los límites X para dar un poco más de "espacio de salida"
ax.set_xlim(-0.1, 1.7)
x_lim_left = ax.get_xlim()[0]
x_lim_right = ax.get_xlim()[1]

ax.set_ylim(-0.5, 0.5)
ax.set_aspect('equal')

pos_n_initial = 0.1
pos_c_initial = 0.8
collision_buffer = 0.02
collision_point_x = pos_c_initial - collision_buffer

neutron, = ax.plot([], [], 'o', color='orange', markersize=15, label='Neutrón', zorder=2)
carbon, = ax.plot([], [], 'o', color='gray', markersize=get_carbon_markersize(m_c), label='Carbono', zorder=2)

text_state = ax.text(0.4, 0.99, '', transform=ax.transAxes, fontsize=12, weight='bold', ha='center', va='top')

ax.axhline(0, color='black', linewidth=0.5, zorder=1)
ax.arrow(0.0, 0, 1.4, 0, head_width=0.03, head_length=0.05, fc='black', ec='black', zorder=1)
ax.text(1.45, -0.05, 'X', fontsize=12, zorder=1)

# --- Variables globales para los objetos Text de velocidad, cronómetro de animación y cronómetro teórico ---
global_text_neutron_vel = None
global_text_carbon_vel = None
global_text_timer = None
global_text_theoretical_timer = None # <--- NUEVO: Objeto de texto para el cronómetro teórico

# --- Parámetros de Duración Visual de las Fases (en frames) ---
frames_despues_choque_fijos = 300
final_hold_frames = 100

# --- Escalado de Velocidades para la Animación Visual (se recalcularán) ---
v_n1_anim = 0.0
v_n2_anim = 0.0
v_c2_anim = 0.0

# --- Listas para almacenar datos de velocidad para la gráfica ---
animation_times = []
neutron_velocities = []
carbon_velocities = []

# --- Variables globales para la duración total y tiempo de colisión en segundos ---
t_col_graph = 0.0

# --- Variable global para la instancia de animación ---
current_animation = None

# --- CONSTANTE PARA LA SENSIBILIDAD DE LA ANIMACIÓN ---
ANIMATION_SPEED_FACTOR = 2.69e-10

# --- Intervalo de animación fijo ---
FIXED_ANIMATION_INTERVAL_MS = 20

# --- Variable para almacenar el número de frames hasta la colisión ---
frames_until_collision = 0
total_frames = 20000

# --- NUEVAS VARIABLES GLOBALES PARA TIEMPOS TEÓRICOS ---
theoretical_time_to_collision = 0.0
theoretical_time_post_collision = 0.0
total_theoretical_time = 0.0

# --- Función de Inicialización de la Animación (limpia) ---
def init():
    neutron.set_data([], [])
    carbon.set_data([], [])
    text_state.set_text('')
    if global_text_neutron_vel is not None:
        global_text_neutron_vel.set_text('')
    if global_text_carbon_vel is not None:
        global_text_carbon_vel.set_text('')
    if global_text_timer is not None:
        global_text_timer.set_text('Tiempo: 0.00 s')
    if global_text_theoretical_timer is not None: # <--- NUEVO: Limpiar el cronómetro teórico
        global_text_theoretical_timer.set_text('Tiempo Teórico: Calculando...')
    
    # Reiniciar la posición de los objetos para la próxima animación
    neutron.set_data([pos_n_initial], [0])
    carbon.set_data([pos_c_initial], [0])
    
    return neutron, carbon, text_state, global_text_neutron_vel, global_text_carbon_vel, global_text_timer, global_text_theoretical_timer # <--- NUEVO: Devolver el objeto del cronómetro teórico

# --- Función de Animación (MODIFICADA) ---
def animate(frame):
    global v_n1x_real, m_c, v_n2x_real, v_c2x_real, v_n1_anim, v_n2_anim, v_c2_anim, t_col_graph, \
           global_text_neutron_vel, global_text_carbon_vel, global_text_timer, global_text_theoretical_timer, \
           frames_until_collision, current_animation, total_theoretical_time # Añadir global_text_theoretical_timer

    time_in_s = frame * (FIXED_ANIMATION_INTERVAL_MS / 1000.0)
    animation_times.append(time_in_s)

    current_pos_n_x = 0.0
    current_pos_c_x = 0.0

    margin_for_particle_size = 0.05

    if frame <= frames_until_collision:
        current_pos_n_x = pos_n_initial + v_n1_anim * frame
        current_pos_c_x = pos_c_initial

        if current_pos_n_x >= collision_point_x:
            current_pos_n_x = collision_point_x

        text_state.set_text('Antes del choque')
        if global_text_neutron_vel is not None:
            global_text_neutron_vel.set_text(f'Neutrón: {v_n1x_real:.2e} m/s')
        if global_text_carbon_vel is not None:
            global_text_carbon_vel.set_text('Carbono: 0 m/s')

        neutron_velocities.append(v_n1x_real)
        carbon_velocities.append(v_c1x_real)

    else: # Después del choque
        frame_since_collision = (frame - frames_until_collision)

        current_pos_n_x = collision_point_x + v_n2_anim * frame_since_collision
        current_pos_c_x = pos_c_initial + v_c2_anim * frame_since_collision

        text_state.set_text('Después del choque')
        if global_text_neutron_vel is not None:
            global_text_neutron_vel.set_text(f'Neutrón: {v_n2x_real:.2e} m/s')
        if global_text_carbon_vel is not None:
            global_text_carbon_vel.set_text(f'Carbono: {v_c2x_real:.2e} m/s')

        neutron_velocities.append(v_n2x_real)
        carbon_velocities.append(v_c2x_real)

    neutron.set_data([current_pos_n_x], [0])
    carbon.set_data([current_pos_c_x], [0])

    # Actualizar el texto del cronómetro de animación
    if global_text_timer is not None:
        global_text_timer.set_text(f'Tiempo Sim.: {time_in_s:.2f} s') # Renombrado para claridad

    # LÓGICA DE DETENCIÓN AJUSTADA
    if current_pos_c_x > x_lim_right - margin_for_particle_size or \
       current_pos_c_x < x_lim_left + margin_for_particle_size:
        
        if current_animation is not None and current_animation.event_source is not None:
            if current_pos_c_x > x_lim_right - margin_for_particle_size or \
               current_pos_c_x < x_lim_left + margin_for_particle_size:
                current_animation.event_source.stop()
                text_state.set_text('¡Animación Terminada! Carbono fuera de pantalla.')
                # Asegurarse de que el tiempo final se muestre en el cronómetro de animación
                if global_text_timer is not None:
                    global_text_timer.set_text(f'Tiempo Sim.: {time_in_s:.2f} s (Final)')
                
                # <--- NUEVO: Mostrar el tiempo teórico final una vez que la animación termina ---
                if global_text_theoretical_timer is not None:
                    global_text_theoretical_timer.set_text(f'Tiempo Teórico: {total_theoretical_time:.2e} s') # Formato científico para tiempo real

                fig.canvas.draw_idle()

    return neutron, carbon, text_state, global_text_neutron_vel, global_text_carbon_vel, global_text_timer, global_text_theoretical_timer # <--- Devolver el nuevo objeto

# --- Función para generar la gráfica de velocidad ---
def generate_velocity_plot(event):
    min_len = min(len(animation_times), len(neutron_velocities), len(carbon_velocities))
    times_np = np.array(animation_times[:min_len])
    neutron_vel_np = np.array(neutron_velocities[:min_len])
    carbon_vel_np = np.array(carbon_velocities[:min_len])

    fig_vel, ax_vel = plt.subplots(figsize=(10, 6))

    ax_vel.plot(times_np, neutron_vel_np, label='Neutrón', color='blue')
    ax_vel.plot(times_np, carbon_vel_np, label='Carbono', color='red')

    ax_vel.axvline(t_col_graph, color='black', linestyle='--', label='Colisión')

    ax_vel.set_xlabel("Tiempo (s)")
    ax_vel.set_ylabel("Velocidad (m/s)")
    ax_vel.set_title("Velocidad vs. Tiempo del Choque Elástico")
    ax_vel.legend()
    ax_vel.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# --- Función para iniciar o reiniciar la animación (MODIFICADA) ---
def run_animation_logic():
    global v_n1x_real, m_c, v_n2x_real, v_c2x_real, v_n1_anim, v_n2_anim, v_c2_anim, t_col_graph, \
           current_animation, frames_until_collision, total_frames, \
           theoretical_time_to_collision, theoretical_time_post_collision, total_theoretical_time # Añadir las nuevas variables globales

    # Detener cualquier animación en curso de forma segura
    if current_animation is not None and current_animation.event_source is not None:
        current_animation.event_source.stop()
        current_animation = None

    # Limpiar los datos de la animación anterior
    animation_times.clear()
    neutron_velocities.clear()
    carbon_velocities.clear()
    
    # Reiniciar el texto del cronómetro de animación
    if global_text_timer is not None:
        global_text_timer.set_text('Tiempo Sim.: 0.00 s')
    
    # <--- NUEVO: Reiniciar el cronómetro teórico y calcularlo ---
    if global_text_theoretical_timer is not None:
        global_text_theoretical_timer.set_text('Tiempo Teórico: Calculando...')


    try:
        new_v_n1x_real = float(textbox_v_n.text)
        new_m_c = float(textbox_m_c.text)

        if new_m_c <= 0:
            raise ValueError("La masa del carbono debe ser mayor que cero.")
        if new_v_n1x_real <= 0:
             raise ValueError("La velocidad del neutrón debe ser mayor que cero.")


        v_n1x_real = new_v_n1x_real
        m_c = new_m_c

        carbon.set_markersize(get_carbon_markersize(m_c))

        v_n2x_real = ((m_n - m_c) / (m_n + m_c)) * v_n1x_real + (2 * m_c / (m_n + m_c)) * v_c1x_real
        v_c2x_real = (2 * m_n / (m_n + m_c)) * v_n1x_real + ((m_c - m_n) / (m_n + m_c)) * v_c1x_real

        v_n1_anim = v_n1x_real * ANIMATION_SPEED_FACTOR
        v_n2_anim = v_n2x_real * ANIMATION_SPEED_FACTOR
        v_c2_anim = v_c2x_real * ANIMATION_SPEED_FACTOR

        distancia_visual_al_choque = collision_point_x - pos_n_initial
        
        if v_n1_anim > 0:
            frames_until_collision = int(distancia_visual_al_choque / v_n1_anim)
        else:
            frames_until_collision = 0

        frames_until_collision = max(1, frames_until_collision) 
        
        total_frames = 500

        t_col_graph = frames_until_collision * (FIXED_ANIMATION_INTERVAL_MS / 1000.0)

        # <--- NUEVO: CÁLCULOS DEL TIEMPO TEÓRICO REAL ---
        # 1. Tiempo hasta la colisión (para el neutrón)
        # La distancia que el neutrón recorre hasta el punto de colisión es pos_c_initial - pos_n_initial
        # (Asumiendo que el carbono inicialmente está quieto en pos_c_initial)
        distance_to_collision_real_units = pos_c_initial - pos_n_initial # Esta es la distancia visual en el plot

        if v_n1x_real != 0:
            theoretical_time_to_collision = distance_to_collision_real_units / v_n1x_real
        else:
            theoretical_time_to_collision = float('inf') # Nunca colisionaría si el neutrón no se mueve

        # 2. Tiempo post-colisión (para el carbono)
        # Distancia que el carbono recorrerá desde su posición inicial (después del choque) hasta salir de pantalla
        # Consideramos el mismo margen que en la condición de parada de la animación
        margin_for_particle_size = 0.05
        distance_carbon_post_collision_real_units = (x_lim_right - margin_for_particle_size) - pos_c_initial

        if v_c2x_real > 0: # Solo si el carbono se mueve hacia la derecha (sale por la derecha)
            theoretical_time_post_collision = distance_carbon_post_collision_real_units / v_c2x_real
        elif v_c2x_real < 0: # Si el carbono se mueve hacia la izquierda
            # Podría salir por la izquierda, la distancia sería diferente
            distance_carbon_post_collision_real_units = pos_c_initial - (x_lim_left + margin_for_particle_size)
            if distance_carbon_post_collision_real_units > 0: # Solo si hay distancia real para salir por la izquierda
                 theoretical_time_post_collision = distance_carbon_post_collision_real_units / abs(v_c2x_real)
            else:
                theoretical_time_post_collision = float('inf') # O ya está fuera, o no se moverá lo suficiente
        else: # Si v_c2x_real es 0, no se moverá
            theoretical_time_post_collision = float('inf')

        # El tiempo total teórico es la suma del tiempo hasta la colisión y el tiempo post-colisión
        # Solo sumamos si ambos tiempos son finitos, de lo contrario, puede ser infinito.
        if theoretical_time_to_collision != float('inf') and theoretical_time_post_collision != float('inf'):
            total_theoretical_time = theoretical_time_to_collision + theoretical_time_post_collision
        else:
            total_theoretical_time = float('inf') # La partícula nunca saldría

        # <--- NUEVO: Mostrar el tiempo teórico calculado inmediatamente ---
        if global_text_theoretical_timer is not None:
            # Si el tiempo es infinito, mostrar "Infinito"
            if total_theoretical_time == float('inf'):
                global_text_theoretical_timer.set_text('Tiempo Teórico: ∞ s (No saldrá de pantalla)')
            else:
                global_text_theoretical_timer.set_text(f'Tiempo Teórico: {total_theoretical_time:.2e} s') # Formato científico


        # Iniciar la nueva animación
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
widget_row_spacing = 0.08
initial_bottom_pos = 0.85

# V Neutrón
fig.text(widget_left_col, initial_bottom_pos + widget_label_offset_y, 'V Neutrón (m/s):',
         fontsize=10, ha='left', va='center')
ax_v_n_textbox = fig.add_axes([widget_left_col, initial_bottom_pos, 0.25, widget_height])
textbox_v_n = TextBox(ax_v_n_textbox, '', initial=str(v_n1x_real))

# Masa Carbono
fig.text(widget_left_col, initial_bottom_pos - widget_row_spacing + widget_label_offset_y, 'Masa Carbono (u):',
         fontsize=10, ha='left', va='center')
ax_m_c_textbox = fig.add_axes([widget_left_col, initial_bottom_pos - widget_row_spacing, 0.25, widget_height])
textbox_m_c = TextBox(ax_m_c_textbox, '', initial=str(m_c))

# Botón Iniciar Animación
ax_start_button = fig.add_axes([widget_left_col, initial_bottom_pos - 2 * widget_row_spacing, 0.25, widget_height + 0.01])
start_button = Button(ax_start_button, 'Iniciar Animación')
start_button.on_clicked(on_start_button_clicked)

# Botón Reiniciar Animación
ax_restart_button = fig.add_axes([widget_left_col, initial_bottom_pos - 3 * widget_row_spacing, 0.25, widget_height + 0.01])
restart_button = Button(ax_restart_button, 'Reiniciar Animación')
restart_button.on_clicked(on_restart_button_clicked)

# Botón Mostrar Gráfica
ax_plot_button = fig.add_axes([widget_left_col, initial_bottom_pos - 4 * widget_row_spacing, 0.25, widget_height + 0.01])
plot_button = Button(ax_plot_button, 'Mostrar Gráfica')
plot_button.on_clicked(generate_velocity_plot)

# --- OBJETOS DE TEXTO PARA LAS VELOCIDADES, CRONÓMETRO DE ANIMACIÓN Y CRONÓMETRO TEÓRICO ---
velocity_text_bottom_pos = initial_bottom_pos - 5 * widget_row_spacing

global_text_neutron_vel = fig.text(widget_left_col, velocity_text_bottom_pos,
                                   'Neutrón: ', fontsize=10, ha='left', va='center')
global_text_carbon_vel = fig.text(widget_left_col, velocity_text_bottom_pos - 0.04,
                                  'Carbono: ', fontsize=10, ha='left', va='center')

global_text_timer = fig.text(widget_left_col, velocity_text_bottom_pos - 0.08,
                             'Tiempo Sim.: 0.00 s', fontsize=10, ha='left', va='center', weight='bold', color='blue')

global_text_theoretical_timer = fig.text(widget_left_col, velocity_text_bottom_pos - 0.12, # <--- NUEVO: Posición debajo del cronómetro de simulación
                                         'Tiempo Teórico: Calculando...', fontsize=10, ha='left', va='center', weight='bold', color='green')


# --- Título General de la Figura (centrado arriba) ---
fig.suptitle('Simulación de Choque Elástico entre Neutrón y Carbono', fontsize=14, weight='bold', y=0.97)

ax.set_yticks([])

plt.tight_layout(rect=[0, 0, widget_left_col - 0.02, 1])

plt.show()