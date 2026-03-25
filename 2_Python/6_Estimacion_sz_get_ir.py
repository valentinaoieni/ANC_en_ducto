import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fft import fft, ifft
from scipy.signal import chirp

def get_ir(y, x, f0min=250, f0max=19000, fs=44100, lor=0):
    """
    Estima la respuesta al impulso (IR) h de un sistema mediante deconvolución en frecuencia.
    
    Parámetros:
    y: Señal grabada (salida del sistema / wet)
    x: Señal original (entrada al sistema / dry)
    f0min, f0max: Rango de frecuencias de interés para el filtrado
    fs: Frecuencia de muestreo
    lor: Selector de canal para señales estéreo (0: Izquierdo, 1: Derecho)
    """
    
    # --- Pre-procesamiento de canales ---
    if y.ndim > 1:
        y = y[:, 1] if lor == 1 else y[:, 0]
    if x.ndim > 1:
        # Se asegura que la señal de referencia sea un vector unidimensional
        x = x.flatten() if x.ndim > 1 else x

    # --- Sincronizar longitudes (Zero Padding) ---
    md, mw = len(x), len(y)
    l = max(md, mw)
    
    x_padded = np.pad(x, (0, l - md))
    y_padded = np.pad(y, (0, l - mw))

    # --- Deconvolución en el dominio de la frecuencia ---
    Xdry = fft(x_padded)
    Xwet = fft(y_padded)

    # Normalizar para evitar división por cero (ruido fuera de banda)
    epsilon = 1e-10 
    H = Xwet / (Xdry + epsilon)

    # --- Filtrado en frecuencia (Ventana Rectangular) ---
    # Convertimos frecuencias de corte a índices de la FFT
    wn1 = int(np.ceil(f0min / fs * l))
    wn2 = int(np.floor(f0max / fs * l))
    
    ventana = np.zeros(l)
    ventana[wn1:wn2] = 1
    
    # Aplicar filtro y volver al dominio del tiempo
    H_filtered = H * ventana
    h = np.real(ifft(H_filtered))
    
    return h

# CONFIGURACIÓN DE LA MEDICIÓN Y DEL HARDWARE
fs = 44100
duracion = 5
t = np.linspace(0, duracion, int(fs * duracion))

f0 = 100    # Frecuencia inicial del barrido
f1 = 15000  # Frecuencia final del barrido

# Configuración del dispositivo de audio (Ajustar según hw o ID)
DEVICE = 5 

# 1. Generación del Log-Chirp (Excitación del sistema)
x_mono = 0.5 * chirp(t, f0=f0, f1=f1, t1=duracion, method='logarithmic')

# 2. Preparación de señal estéreo (Enviar señal por canal derecho)
x_stereo = np.zeros((len(x_mono), 2))
x_stereo[:, 1] = x_mono 

# 3. Ejecución de la medición: Reproducción y Grabación simultánea
print(f"Iniciando grabación en dispositivo {DEVICE}...")
grabacion = sd.playrec(
    x_stereo, 
    fs, 
    channels=2, 
    device=DEVICE, 
    blocking=True, 
    latency='low'
)
print("Grabación finalizada.")

# 4. Cálculo de la Respuesta al Impulso (IR)
# Se utiliza el canal 0 de la grabación para procesar

ir = get_ir(grabacion, x_mono, f0min=f0, f0max=f1, fs=fs)
max_idx = np.argmax(np.abs(ir))

# Ploteo de los resultados obtenidos

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title(r"Respuesta al Impulso Estimación Camino Secundario", fontsize=22)
plt.axvline(max_idx, color='r', linestyle='--', label=f'Pico en {max_idx}')
plt.xlim([0, 3000])
plt.xlabel('Muestras [adim.]', fontsize=18)
plt.ylabel('Amplitud [u.r.]', fontsize=18)
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)
freqs = np.fft.rfftfreq(len(ir), 1/fs)
mag = 20 * np.log10(np.abs(np.fft.rfft(ir)) + 1e-6)
plt.semilogx(freqs, mag)
plt.xlim([f0, f1])
plt.title("Respuesta en Frecuencia", fontsize=22)
plt.xlabel("Frecuencia [Hz]", fontsize=18)
plt.ylabel("Magnitud [dB]", fontsize=18)
plt.grid(True, which="both")
plt.tight_layout()
plt.show()
print(np.argmax(ir))
