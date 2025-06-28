import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, TextBox
import matplotlib
matplotlib.use('TkAgg') # Asegura que la ventana de la animación se muestre

# --- Datos del Problema (Variables Globales que se pueden modificar) ---
v_n1_magnitud_real = 2.6e7 # Magnitud de la velocidad inicial del neutrón
m_c = 12.0 # Masa inicial del carbono
m_n = 1.0 # La masa del neutrón se mantiene constante según el problema original

# --- Ángulo de la trayectoria diagonal del neutrón (en radianes) ---
# Por ejemplo, 45 grados (pi/4). El neutrón se moverá hacia arriba y a la derecha.
angulo_neutron = np.pi / 4

# --- Componentes de la velocidad inicial del neutrón ---
v_n1x_real = v_n1_magnitud_real * np.cos(angulo_neutron)
v_n1y_real = v_n1_magnitud_real * np.sin(angulo_neutron)


# --- Cálculos para las velocidades después del choque (se recalcularán para 2D) ---
v_c1x_real = 0.0
v_c1y_real = 0.0 # Carbono inicialmente en reposo
v_n2x_real = 0.0
v_n2y_real = 0.0
v_c2x_real = 0.0
v_c2y_real = 0.0

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
pos_c_initial_x = 0.8
pos_c_initial_y = pos_n_initial_y + (pos_c_initial_x - pos_n_initial_x) * np.tan(angulo_neutron) # Carbono en la trayectoria

collision_buffer = 0.02 # Un pequeño buffer para la colisión
# El punto de colisión es donde el neutrón alcanzaría la coordenada x del carbono
collision_point_x = pos_c_initial_x - collision_buffer
collision_point_y = pos_n_initial_y + (collision_point_x - pos_n_initial_x) * np.tan(angulo_neutron)


neutron, = ax.plot([], [], 'o', color='orange', markersize=15, label='Neutrón', zorder=2)
carbon, = ax.plot([], [], 'o', color='gray', markersize=get_carbon_markersize(m_c), label='Carbono', zorder=2)

text_state = ax.text(0.4, 0.99, '', transform=ax.transAxes, fontsize=12, weight='bold', ha='center', va='top')

# Ejes X e Y
ax.axhline(0, color='black', linewidth=0.5, zorder=1) # Eje X
ax.axvline(0, color='black', linewidth=0.5, zorder=1) # Eje Y
ax.arrow(0.0, 0, x_lim_right - 0.2, 0, head_width=0.03, head_length=0.05, fc='black', ec='black', zorder=1)
ax.text(x_lim_right - 0.15, -0.05, 'X', fontsize=12, zorder=1)
ax.arrow(0, 0.0, 0, y_lim_top - 0.1, head_width=0.03, head_length=0.05, fc='black', ec='black', zorder=1)
ax.text(-0.05, y_lim_top - 0.05, 'Y', fontsize=12, zorder=1)


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
    neutron.set_data([pos_n_initial_x], [pos_n_initial_y])
    carbon.set_data([pos_c_initial_x], [pos_c_initial_y])
    
    return neutron, carbon, text_state, global_text_neutron_vel, global_text_carbon_vel, global_text_timer, global_text_theoretical_timer # <--- NUEVO: Devolver el objeto del cronómetro teórico

# --- Función de Animación (MODIFICADA para trayectoria diagonal) ---
margin_for_particle_size = 0.05
def animate(frame):
    global v_n1x_real, v_n1y_real, m_c, v_n2x_real, v_n2y_real, v_c2x_real, v_c2y_real, \
           v_n1_anim_x, v_n1_anim_y, v_n2_anim_x, v_n2_anim_y, v_c2_anim_x, v_c2_anim_y, \
           t_col_graph, global_text_neutron_vel, global_text_carbon_vel, global_text_timer, \
           global_text_theoretical_timer, frames_until_collision, current_animation, total_theoretical_time

    time_in_s = frame * (FIXED_ANIMATION_INTERVAL_MS / 1000.0)
    animation_times.append(time_in_s)

    current_pos_n_x = 0.0
    current_pos_n_y = 0.0
    current_pos_c_x = 0.0
    current_pos_c_y = 0.0


    if frame <= frames_until_collision:
        # Movimiento del neutrón antes de la colisión (diagonal)
        current_pos_n_x = pos_n_initial_x + v_n1_anim_x * frame
        current_pos_n_y = pos_n_initial_y + v_n1_anim_y * frame
        
        # Carbono permanece en su posición inicial (que está en la trayectoria del neutrón)
        current_pos_c_x = pos_c_initial_x
        current_pos_c_y = pos_c_initial_y

        # Asegurar que el neutrón no pase el punto de colisión visualmente antes del frame de colisión
        if current_pos_n_x >= collision_point_x:
            current_pos_n_x = collision_point_x
        if current_pos_n_y >= collision_point_y: # Asumiendo que se mueve hacia arriba y derecha
            current_pos_n_y = collision_point_y


        text_state.set_text('Antes del choque')
        if global_text_neutron_vel is not None:
            # Mostrar magnitud de velocidad o componentes (ej. magnitud)
            v_n1_mag = np.sqrt(v_n1x_real**2 + v_n1y_real**2)
            global_text_neutron_vel.set_text(f'Neutrón: {v_n1_mag:.2e} m/s')
        if global_text_carbon_vel is not None:
            global_text_carbon_vel.set_text(f'Carbono: {np.sqrt(v_c1x_real**2 + v_c1y_real**2):.2e} m/s') # Debería ser 0

        neutron_velocities.append(np.sqrt(v_n1x_real**2 + v_n1y_real**2)) # Guardar magnitud
        carbon_velocities.append(np.sqrt(v_c1x_real**2 + v_c1y_real**2))

    else: # Después del choque
        frame_since_collision = (frame - frames_until_collision)

        # Movimiento del neutrón después de la colisión
        current_pos_n_x = collision_point_x + v_n2_anim_x * frame_since_collision
        current_pos_n_y = collision_point_y + v_n2_anim_y * frame_since_collision
        
        # Movimiento del carbono después de la colisión
        current_pos_c_x = pos_c_initial_x + v_c2_anim_x * frame_since_collision
        current_pos_c_y = pos_c_initial_y + v_c2_anim_y * frame_since_collision


        text_state.set_text('Después del choque')
        if global_text_neutron_vel is not None:
            v_n2_mag = np.sqrt(v_n2x_real**2 + v_n2y_real**2)
            global_text_neutron_vel.set_text(f'Neutrón: {v_n2_mag:.2e} m/s')
        if global_text_carbon_vel is not None:
            v_c2_mag = np.sqrt(v_c2x_real**2 + v_c2y_real**2)
            global_text_carbon_vel.set_text(f'Carbono: {v_c2_mag:.2e} m/s')

        neutron_velocities.append(np.sqrt(v_n2x_real**2 + v_n2y_real**2))
        carbon_velocities.append(np.sqrt(v_c2x_real**2 + v_c2y_real**2))

    neutron.set_data([current_pos_n_x], [current_pos_n_y])
    carbon.set_data([current_pos_c_x], [current_pos_c_y])


    # Actualizar el texto del cronómetro de animación
    if global_text_timer is not None:
        global_text_timer.set_text(f'Tiempo Sim.: {time_in_s:.2f} s') # Renombrado para claridad

    # LÓGICA DE DETENCIÓN AJUSTADA para 2D (considerar límites en X e Y)
    # Detener si el carbono o el neutrón salen de los límites del área visible
    stop_animation = False
    if not (x_lim_left + margin_for_particle_size < current_pos_c_x < x_lim_right - margin_for_particle_size and \
            y_lim_bottom + margin_for_particle_size < current_pos_c_y < y_lim_top - margin_for_particle_size):
        text_state.set_text('¡Animación Terminada! Carbono fuera de pantalla.')
        stop_animation = True
    
    if not (x_lim_left + margin_for_particle_size < current_pos_n_x < x_lim_right - margin_for_particle_size and \
            y_lim_bottom + margin_for_particle_size < current_pos_n_y < y_lim_top - margin_for_particle_size):
        # Podríamos tener un mensaje diferente si el neutrón sale primero, o un mensaje genérico.
        # Por ahora, si el neutrón sale, también detenemos.
        if not stop_animation: # Solo si el carbono no lo detuvo ya
             text_state.set_text('¡Animación Terminada! Partícula fuera de pantalla.')
        stop_animation = True

    if stop_animation:
        if current_animation is not None and current_animation.event_source is not None:
            current_animation.event_source.stop()
            # Asegurarse de que el tiempo final se muestre en el cronómetro de animación
            if global_text_timer is not None:
                global_text_timer.set_text(f'Tiempo Sim.: {time_in_s:.2f} s (Final)')
            
            if global_text_theoretical_timer is not None:
                # Actualizar el tiempo teórico final si es necesario o simplemente dejar el último calculado
                 if total_theoretical_time == float('inf'):
                    global_text_theoretical_timer.set_text('Tiempo Teórico: ∞ s (No saldría)')
                 else:
                    global_text_theoretical_timer.set_text(f'Tiempo Teórico: {total_theoretical_time:.2e} s')
            fig.canvas.draw_idle()


    return neutron, carbon, text_state, global_text_neutron_vel, global_text_carbon_vel, global_text_timer, global_text_theoretical_timer

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

# --- Función para iniciar o reiniciar la animación (MODIFICADA para 2D) ---
def run_animation_logic():
    global v_n1_magnitud_real, m_c, angulo_neutron, \
           v_n1x_real, v_n1y_real, v_c1x_real, v_c1y_real, \
           v_n2x_real, v_n2y_real, v_c2x_real, v_c2y_real, \
           v_n1_anim_x, v_n1_anim_y, v_n2_anim_x, v_n2_anim_y, v_c2_anim_x, v_c2_anim_y, \
           t_col_graph, current_animation, frames_until_collision, total_frames, \
           theoretical_time_to_collision, theoretical_time_post_collision, total_theoretical_time, \
           pos_c_initial_x, pos_c_initial_y, collision_point_x, collision_point_y # Necesitamos actualizar pos_c y collision_point si el ángulo cambia


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
        new_v_n_magnitud_real = float(textbox_v_n.text) # El textbox ahora controla la magnitud
        new_m_c = float(textbox_m_c.text)

        if new_m_c <= 0:
            raise ValueError("La masa del carbono debe ser mayor que cero.")
        if new_v_n_magnitud_real <= 0:
             raise ValueError("La magnitud de la velocidad del neutrón debe ser mayor que cero.")

        v_n1_magnitud_real = new_v_n_magnitud_real
        m_c = new_m_c

        # Recalcular componentes de velocidad inicial del neutrón con la nueva magnitud y el ángulo fijo
        v_n1x_real = v_n1_magnitud_real * np.cos(angulo_neutron)
        v_n1y_real = v_n1_magnitud_real * np.sin(angulo_neutron)
        
        # Carbono inicialmente en reposo
        v_c1x_real = 0.0
        v_c1y_real = 0.0

        carbon.set_markersize(get_carbon_markersize(m_c))

        # --- Colisión Elástica 2D ---
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

        # Aplicamos las fórmulas de choque elástico 1D para las componentes paralelas:
        v_n2_paralelo = ((m_n - m_c) / (m_n + m_c)) * v_n1_paralelo
        v_c2_paralelo = (2 * m_n / (m_n + m_c)) * v_n1_paralelo
        
        # Ahora, descomponemos estas velocidades paralelas de nuevo en componentes x e y:
        v_n2x_real = v_n2_paralelo * np.cos(angulo_neutron)
        v_n2y_real = v_n2_paralelo * np.sin(angulo_neutron)
        v_c2x_real = v_c2_paralelo * np.cos(angulo_neutron)
        v_c2y_real = v_c2_paralelo * np.sin(angulo_neutron)

        # Las componentes perpendiculares a la línea de colisión no cambian para el neutrón
        # si la interacción es solo a lo largo de esa línea (simplificación).
        # Y el carbono inicialmente no tiene velocidad perpendicular.
        # Esta es una simplificación fuerte para mantener la trayectoria fija.

        # Velocidades para la animación (componentes)
        v_n1_anim_x = v_n1x_real * ANIMATION_SPEED_FACTOR
        v_n1_anim_y = v_n1y_real * ANIMATION_SPEED_FACTOR
        v_n2_anim_x = v_n2x_real * ANIMATION_SPEED_FACTOR
        v_n2_anim_y = v_n2y_real * ANIMATION_SPEED_FACTOR
        v_c2_anim_x = v_c2x_real * ANIMATION_SPEED_FACTOR
        v_c2_anim_y = v_c2y_real * ANIMATION_SPEED_FACTOR

        # Distancia visual al choque (a lo largo del eje X, pero el tiempo se basa en la velocidad diagonal)
        distancia_visual_al_choque_x = collision_point_x - pos_n_initial_x
        distancia_visual_al_choque_y = collision_point_y - pos_n_initial_y
        distancia_visual_al_choque_total = np.sqrt(distancia_visual_al_choque_x**2 + distancia_visual_al_choque_y**2)

        v_n1_anim_magnitud = np.sqrt(v_n1_anim_x**2 + v_n1_anim_y**2)
        
        if v_n1_anim_magnitud > 0:
            frames_until_collision = int(distancia_visual_al_choque_total / v_n1_anim_magnitud)
        else:
            frames_until_collision = 0

        frames_until_collision = max(1, frames_until_collision) 
        
        total_frames = 1000 # Aumentar para dar más tiempo a la animación 2D

        t_col_graph = frames_until_collision * (FIXED_ANIMATION_INTERVAL_MS / 1000.0)

        # --- CÁLCULOS DEL TIEMPO TEÓRICO REAL para 2D ---
        # 1. Tiempo hasta la colisión (para el neutrón)
        distance_to_collision_real_units = np.sqrt((pos_c_initial_x - pos_n_initial_x)**2 + (pos_c_initial_y - pos_n_initial_y)**2)

        if v_n1_magnitud_real != 0:
            theoretical_time_to_collision = distance_to_collision_real_units / v_n1_magnitud_real
        else:
            theoretical_time_to_collision = float('inf')

        # 2. Tiempo post-colisión (para el carbono, asumiendo que se mueve y sale de pantalla)
        # Esto es más complejo en 2D. Simplificamos: tiempo hasta que la coordenada X o Y salga.
        # O mejor, tiempo hasta que la distancia desde el origen (o punto de colisión) exceda un límite.
        # Por ahora, usamos una heurística similar a la 1D, pero podría ser infinito si no sale.
        
        # Tiempo para que el carbono salga por la derecha (eje X)
        time_c_exit_x_right = float('inf')
        if v_c2x_real > 0:
            dist_c_exit_x_right = (x_lim_right - margin_for_particle_size) - pos_c_initial_x
            if dist_c_exit_x_right > 0 :
                 time_c_exit_x_right = dist_c_exit_x_right / v_c2x_real
        
        # Tiempo para que el carbono salga por la izquierda (eje X)
        time_c_exit_x_left = float('inf')
        if v_c2x_real < 0:
            dist_c_exit_x_left = pos_c_initial_x - (x_lim_left + margin_for_particle_size)
            if dist_c_exit_x_left > 0:
                time_c_exit_x_left = dist_c_exit_x_left / abs(v_c2x_real)

        # Tiempo para que el carbono salga por arriba (eje Y)
        time_c_exit_y_top = float('inf')
        if v_c2y_real > 0:
            dist_c_exit_y_top = (y_lim_top - margin_for_particle_size) - pos_c_initial_y
            if dist_c_exit_y_top > 0:
                time_c_exit_y_top = dist_c_exit_y_top / v_c2y_real
        
        # Tiempo para que el carbono salga por abajo (eje Y)
        time_c_exit_y_bottom = float('inf')
        if v_c2y_real < 0:
            dist_c_exit_y_bottom = pos_c_initial_y - (y_lim_bottom + margin_for_particle_size)
            if dist_c_exit_y_bottom > 0:
                time_c_exit_y_bottom = dist_c_exit_y_bottom / abs(v_c2y_real)

        # El tiempo post-colisión es el mínimo de estos tiempos de salida (si son finitos y positivos)
        possible_exit_times = [t for t in [time_c_exit_x_right, time_c_exit_x_left, time_c_exit_y_top, time_c_exit_y_bottom] if t > 0]
        if not possible_exit_times:
            theoretical_time_post_collision = float('inf')
        else:
            theoretical_time_post_collision = min(possible_exit_times)


        if theoretical_time_to_collision != float('inf') and theoretical_time_post_collision != float('inf'):
            total_theoretical_time = theoretical_time_to_collision + theoretical_time_post_collision
        else:
            total_theoretical_time = float('inf') 

        if global_text_theoretical_timer is not None:
            if total_theoretical_time == float('inf'):
                global_text_theoretical_timer.set_text('Tiempo Teórico: ∞ s')
            else:
                global_text_theoretical_timer.set_text(f'Tiempo Teórico: {total_theoretical_time:.2e} s')


        # Iniciar la nueva animación
        # Asegurarse de que las nuevas variables de velocidad de animación _x e _y se usan en animate()
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

# V Neutrón (Magnitud)
fig.text(widget_left_col, initial_bottom_pos + widget_label_offset_y, 'Velocidad Neutrón (m/s):',
         fontsize=10, ha='left', va='center') # Cambiado a "Velocidad Neutrón (m/s)"
ax_v_n_textbox = fig.add_axes([widget_left_col, initial_bottom_pos, 0.25, widget_height])
textbox_v_n = TextBox(ax_v_n_textbox, '', initial=str(v_n1_magnitud_real)) # Usar v_n1_magnitud_real

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