import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, TextBox # Importar TextBox
import matplotlib
matplotlib.use('TkAgg') # Asegura que la ventana de la animación se muestre

# --- Datos del Problema (Variables Globales que se pueden modificar) ---
# Inicializamos con los valores por defecto que tenías
v_n1x_real = 2.6e7
m_c = 12.0
m_n = 1.0 # La masa del neutrón se mantiene constante según el problema original

# --- Cálculos para las velocidades después del choque (se recalcularán) ---
v_c1x_real = 0.0
v_n2x_real = 0.0 # Se inicializan, se calcularán más tarde
v_c2x_real = 0.0 # Se inicializan, se calcularán más tarde

# --- Configuración de la Animación ---
fig, ax = plt.subplots(figsize=(10, 6)) # Un poco más grande para los widgets

ax.set_xlim(-0.1, 1.5)
ax.set_ylim(-0.5, 0.5)
ax.set_aspect('equal')

pos_n_initial = 0.1
pos_c_initial = 0.8

neutron, = ax.plot([], [], 'o', color='orange', markersize=15, label='Neutrón')
carbon, = ax.plot([], [], 'o', color='gray', markersize=30, label='Carbono')

text_neutron_vel = ax.text(0.05, 0.4, '', transform=ax.transAxes, fontsize=10)
text_carbon_vel = ax.text(0.6, 0.4, '', transform=ax.transAxes, fontsize=10)
text_state = ax.text(0.4, 0.9, '', transform=ax.transAxes, fontsize=12, weight='bold')

ax.axhline(0, color='black', linewidth=0.5)
ax.arrow(0.0, 0, 1.4, 0, head_width=0.03, head_length=0.05, fc='black', ec='black')
ax.text(1.45, -0.05, 'X', fontsize=12)

# --- Parámetros de Tiempo para las Fases de la Animación ---
tiempo_antes_choque = 100
tiempo_despues_choque = 300
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
total_frames = tiempo_antes_choque + tiempo_despues_choque + final_hold_frames
total_animation_duration_s = total_frames * 0.02
t_col_graph = 0.0 # Se calculará después

# --- Variable global para la instancia de animación ---
current_animation = None

# --- Función de Inicialización de la Animación (limpia) ---
def init():
    neutron.set_data([], [])
    carbon.set_data([], [])
    text_neutron_vel.set_text('')
    text_carbon_vel.set_text('')
    text_state.set_text('')
    return neutron, carbon, text_neutron_vel, text_carbon_vel, text_state

# --- Función de Animación (usa variables globales para las velocidades) ---
def animate(frame):
    global v_n1x_real, m_c, v_n2x_real, v_c2x_real, v_n1_anim, v_n2_anim, v_c2_anim, t_col_graph

    time_in_s = (frame / total_frames) * total_animation_duration_s
    animation_times.append(time_in_s)

    if frame < tiempo_antes_choque:
        current_pos_n = pos_n_initial + v_n1_anim * frame
        current_pos_c = pos_c_initial

        if current_pos_n >= pos_c_initial - 0.02:
            current_pos_n = pos_c_initial - 0.02

        text_state.set_text('Antes del choque')
        text_neutron_vel.set_text(f'Neutrón: {v_n1x_real:.2e} m/s')
        text_carbon_vel.set_text('Carbono: 0 m/s')

        neutron_velocities.append(v_n1x_real)
        carbon_velocities.append(v_c1x_real)

    elif frame < tiempo_antes_choque + tiempo_despues_choque:
        frame_since_collision = (frame - tiempo_antes_choque)

        collision_point_n = pos_c_initial - 0.02

        current_pos_n = collision_point_n + v_n2_anim * frame_since_collision
        current_pos_c = pos_c_initial + v_c2_anim * frame_since_collision

        text_state.set_text('Después del choque')
        text_neutron_vel.set_text(f'Neutrón: {v_n2x_real:.2e} m/s')
        text_carbon_vel.set_text(f'Carbono: {v_c2x_real:.2e} m/s')

        neutron_velocities.append(v_n2x_real)
        carbon_velocities.append(v_c2x_real)

    else:
        frame_since_collision_total = tiempo_despues_choque

        collision_point_n = pos_c_initial - 0.02

        current_pos_n = collision_point_n + v_n2_anim * frame_since_collision_total
        current_pos_c = pos_c_initial + v_c2_anim * frame_since_collision_total

        text_state.set_text('Después del choque (Final)')
        text_neutron_vel.set_text(f'Neutrón: {v_n2x_real:.2e} m/s')
        text_carbon_vel.set_text(f'Carbono: {v_c2x_real:.2e} m/s')

        neutron_velocities.append(v_n2x_real)
        carbon_velocities.append(v_c2x_real)

    neutron.set_data([current_pos_n], [0])
    carbon.set_data([current_pos_c], [0])

    return neutron, carbon, text_neutron_vel, text_carbon_vel, text_state

# --- Función para generar la gráfica de velocidad (sin cambios) ---
def generate_velocity_plot(event):
    # Asegurarse de que las longitudes de las listas sean iguales
    min_len = min(len(animation_times), len(neutron_velocities), len(carbon_velocities))
    times_np = np.array(animation_times[:min_len])
    neutron_vel_np = np.array(neutron_velocities[:min_len])
    carbon_vel_np = np.array(carbon_velocities[:min_len])

    fig_vel, ax_vel = plt.subplots(figsize=(10, 6))

    ax_vel.plot(times_np, neutron_vel_np, label='Neutrón', color='blue')
    ax_vel.plot(times_np, carbon_vel_np, label='Carbono', color='red')

    # Usar t_col_graph que se calcula al iniciar la animación
    ax_vel.axvline(t_col_graph, color='black', linestyle='--', label='Colisión')

    ax_vel.set_xlabel("Tiempo (s)")
    ax_vel.set_ylabel("Velocidad (m/s)")
    ax_vel.set_title("Velocidad vs. Tiempo del Choque Elástico")
    ax_vel.legend()
    ax_vel.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# --- Función para iniciar la animación con los nuevos valores ---
def start_animation(event):
    global v_n1x_real, m_c, v_n2x_real, v_c2x_real, v_n1_anim, v_n2_anim, v_c2_anim, t_col_graph, current_animation

    # Detener la animación si ya hay una en curso
    if current_animation is not None:
        current_animation.event_source.stop()
        # Opcional: limpiar los datos de la animación si es necesario
        neutron.set_data([], [])
        carbon.set_data([], [])
        fig.canvas.draw_idle() # Redibuja la figura para que se vea limpia

    # Limpiar las listas de datos para la nueva animación
    animation_times.clear()
    neutron_velocities.clear()
    carbon_velocities.clear()

    try:
        # Leer los valores de los campos de texto
        new_v_n1x_real = float(textbox_v_n.text)
        new_m_c = float(textbox_m_c.text)

        # Validar si los valores son razonables (ej. masa no negativa)
        if new_m_c <= 0:
            raise ValueError("La masa del carbono debe ser mayor que cero.")

        # Actualizar las variables globales con los nuevos valores
        v_n1x_real = new_v_n1x_real
        m_c = new_m_c

        # Recalcular las velocidades físicas reales después del choque
        v_n2x_real = ((m_n - m_c) / (m_n + m_c)) * v_n1x_real + (2 * m_c / (m_n + m_c)) * v_c1x_real
        v_c2x_real = (2 * m_n / (m_n + m_c)) * v_n1x_real + ((m_c - m_n) / (m_n + m_c)) * v_c1x_real

        # Recalcular las velocidades de animación escaladas
        distancia_visual_recorrido_n_inicial = (pos_c_initial - 0.02) - pos_n_initial
        v_n1_anim = distancia_visual_recorrido_n_inicial / tiempo_antes_choque
        v_n2_anim = v_n1_anim * (v_n2x_real / v_n1x_real) if v_n1x_real != 0 else 0
        v_c2_anim = v_n1_anim * (v_c2x_real / v_n1x_real) if v_n1x_real != 0 else 0

        # Recalcular el tiempo de colisión para la gráfica
        t_col_graph = (tiempo_antes_choque / total_frames) * total_animation_duration_s
        
        # Iniciar la nueva animación
        current_animation = animation.FuncAnimation(fig, animate, frames=total_frames,
                                                  init_func=init, blit=True, interval=20, repeat=False) # repeat=False para que no se repita
        fig.canvas.draw_idle() # Asegura que la figura se redibuje con la nueva animación

    except ValueError as e:
        print(f"Error en los valores de entrada: {e}")
        # Puedes añadir un mensaje de error visual en la figura si lo deseas
        text_state.set_text(f"ERROR: {e}")
        fig.canvas.draw_idle()


# --- Configuración de los Widgets de Entrada y Botones ---
# Crear espacio para los campos de entrada
# [left, bottom, width, height] en coordenadas de la figura
ax_v_n_textbox = plt.axes([0.05, 0.90, 0.15, 0.05]) # Posición para velocidad del neutrón
textbox_v_n = TextBox(ax_v_n_textbox, 'V Neutrón (m/s): ', initial=str(v_n1x_real))

ax_m_c_textbox = plt.axes([0.05, 0.83, 0.15, 0.05]) # Posición para masa del carbono
textbox_m_c = TextBox(ax_m_c_textbox, 'Masa Carbono (u): ', initial=str(m_c))

# Crear el botón de inicio de animación
ax_start_button = plt.axes([0.8, 0.90, 0.15, 0.05]) # Posición para el botón Iniciar
start_button = Button(ax_start_button, 'Iniciar Animación')
start_button.on_clicked(start_animation)

# Crear el botón para mostrar gráfica (posicionamiento ajustado para no colisionar)
ax_plot_button = plt.axes([0.8, 0.83, 0.15, 0.05]) # Posición para el botón Mostrar Gráfica
plot_button = Button(ax_plot_button, 'Mostrar Gráfica')
plot_button.on_clicked(generate_velocity_plot)


plt.title('Simulación de Choque Elástico entre Neutrón y Carbono')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Ocultar los ticks del eje Y para un gráfico más limpio del movimiento horizontal
ax.set_yticks([])

# Mostrar la figura principal que contiene los widgets.
# La animación no comenzará hasta que se presione el botón "Iniciar Animación".
plt.show()