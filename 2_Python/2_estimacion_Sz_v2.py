import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import time

"""
Script de ESTIMACIÓN DEL CAMINO SECUNDARIO

ANC en configuración Feedforward, que incluye estimación del camino secundario offline (remitirse a otro script)
y estimación del camino primario online mediante algoritmo LMS
Autores:
- Suárez, Alejo
- Oieni Orozco, Valentina

Se define una clase de objeto ANC_Feedforward, que contiene también una función, esta es el algoritmo FxLMS
Esta funcion se ejecuta cada vez que se llena el buffer de entrada con las muestras del sonido captado.
La librería sounddevice gestiona esto, y llama a la función "callback" cada vez que esto sucede. Dentro de
esta función callback está contenido el método de computar por FxLMS, la señal de control y la adaptación
del filtro del camino primario

---PARÁMETROS
Fs = Frecuencia de muestreo. Puede ser 44100 o 48000 ya que la biblioteca sound device no admite otras.

BLOCKSIZE = Número de frames que se procesan luego de que se llene un buffer de esta capacidad
            En nuestra aplicación el número de frames a procesar simultáneamente es definido y es 2048
            de forma tal de asegurar que el número de datos captados en el buffer sean coincidentes con
            el número de taps de los filtros. Esto la librería lo recomienda solo cuando es necesario
            setear un número fijo de frames, por procesamiento.
            Para asignación automática y dinámica por la librería, de la cantidad de frames por bloque,
            setear BLOCKSIZE = 0. Esto hace que trabaje con la mínima cantidad posible de frames y esto
            hace teóricamente que se evite overflow a la entrada y underflow a la salida. Es decir, que
            se pierdan datos.
            IMPORTANTE: si el sistema no anda, alternar con varias ejecutaciones de codigo entre BLOCKSIZE = 0
            y BLOCKSIZE = 2048, que no ande o que haya overflow está dado muchas veces por falta de memoria y
            otros procesos en ejecución. Dejar solo el proceso de python

LATENCY  =  Se le indica a la librería de ajustar el procesado de forma tal que exista la menor latencia 
            posible y esto se logra con el valor LATENCY = 'low'

DEVICE   =  Se le indica a la librería el dispositivo a utilizar, para poder identificarlos, se debe 
            ejecutar en otro script previo el comando print(sd.query_devices()), de forma completa:
            import sounddevice as sd
            print(sd.query_devices())
            Esto para identificar por ID (número) el dispositivo de la placa USB Dual Pre Project Series

N_TAPS   =  Longitud o tamaño del filtro FIR tanto del camino secundario S(z), como del camino primario P(z)
            Se destaca que coincide con el parámetro BLOCKSIZE, por lo explicado anteriormente

MU       =  Step Size o paso del algoritmo. Mientras más grande sea, más rápido converge, pero existe un valor
            límite para el cual el sistema se torna inestable. Por ello se lo llama también Factor de convergencia
            IMPORTANTE:
            En el LMS Feedforward debe ser un valor muy pequeño, en el orden de 1e-5 o 1e-6, la convergencia es
            lenta pero estable

DURACION_ESTIMACION = Segundos de duración del experimento
            
IMPORTANTE: Antes de realizar el experimento, comprobar que:
-el parlante actuador esté en el canal 0 
-el parlante de ruido esté en el canal 1
-que el micrófono de error, esté en el canal 0
-que el micrófono de ruido, esté en el canal 1

"""

# --- CONFIGURACIÓN ---
FS = 44100          # Frecuencia de muestreo. Puede ser 44100 o 48000 ya que la biblioteca sound device no admite otras.
BLOCKSIZE = 0       # Número de frames que se procesan luego de que se llene un buffer de esta capacidad. 
N_TAPS = 2048       # Sugerencia de latencia para el driver
MU = 0.01           # Bajamos MU para mayor estabilidad en el mundo real
EPS = 1e-3          # Elección de cota de error
duracion = 30
DEVICE = 5 #'hw:2,0'
flag = True

print(sd.query_devices())
input("Presione [Enter] para continuar")

class RobustEstimator:
    def __init__(self, n_taps, mu, eps):
        self.w = np.zeros(n_taps)
        self.x_buffer = np.zeros(n_taps)
        self.mu = mu
        self.eps = eps
        self.error_history = []

    def process_block(self, x_block, d_block):
        n_samples = len(x_block)
        y_block = np.zeros(n_samples)

        for i in range(n_samples):
            # 1. Shifteo 
            self.x_buffer = np.roll(self.x_buffer, 1)
            self.x_buffer[0] = x_block[i]
            
            # 2. Predicción
            y_n = np.dot(self.w, self.x_buffer)
            e_n = d_block[i] - y_n
            
            # 3. NLMS: Normalización por energía del buffer
            norm_x = np.dot(self.x_buffer, self.x_buffer) + self.eps
            
            # 4. Actualización 
            self.w += (self.mu / norm_x) * e_n * self.x_buffer
            
            y_block[i] = y_n
            self.error_history.append(e_n)
            
        return y_block

# --- INICIALIZACIÓN ---
noise = 0.5 * np.random.normal(0, 0.2, FS * duracion).astype('float32') # 20 segundos
est = RobustEstimator(N_TAPS, MU, EPS)
idx = 0

def callback(indata, outdata, frames, time_info, status):
    global idx
    if status:
        print(status)
    
    if idx + frames > len(noise):
        outdata.fill(0)
        return

    # Asgurarse de que tanto para la entrada como la salida el mapeo sea correcto
    d_chunk = indata[:, 0] 
    x_chunk = noise[idx : idx + frames]
    
    _ = est.process_block(x_chunk, d_chunk)
    
    outdata[:, 1] = x_chunk # Sale por el parlante actuador
    outdata[:, 0] = np.zeros(len(x_chunk)) # Sale por el parlante actuador
    idx += frames

# --- EJECUCIÓN ---
try:
    # Usamos latency='high' para evitar cortes de audio en Python
    with sd.Stream(samplerate=FS, blocksize=BLOCKSIZE, channels=1, 
                   callback=callback, latency='low', device=DEVICE):
        print("Procesando... Observa si aparecen errores de Underflow/Overflow.")
        time.sleep(15)
except Exception as e:
    print(f"Error: {e}")

# --- GRÁFICOS Y ANÁLISIS ---
max_idx = np.argmax(np.abs(est.w))
print(f"\nPico detectado en el tap: {max_idx}")
print(f"Latencia estimada: {(max_idx/FS)*1000:.2f} ms")

plt.figure(figsize=(12, 6))
plt.subplot(2,1,1)
plt.stem(est.w)
plt.axvline(max_idx, color='r', linestyle='--', label=f'Pico en {max_idx}')
plt.title("Respuesta al Impulso (Filtro W)") 
plt.legend()

plt.subplot(2,1,2)
plt.plot(est.error_history)
plt.title("Historial del Error")
plt.tight_layout()
plt.show()

# Se guardan los coeficientes para utilizarlos luego

file_path = r'/home/alejo/Documentos/final_acustica/s_hat_coefs.npy'

np.savetxt(file_path, est.w, delimiter=',')

# Se realiza el ploteo de la respuesta en frecuencia obtenida a partir de la respuesta impulsiva
NFFT = 8192
H = np.fft.rfft(est.w, NFFT)
freq = np.fft.rfftfreq(NFFT, 1/FS)

# Magnitud en dB
mag = np.abs(H)
H_db = 20 * np.log10(mag / np.max(mag) + 1e-6) 

plt.figure(figsize=(12, 6))
plt.semilogx(freq, H_db)

plt.title("Respuesta en Frecuencia Estimación (NLMS)", fontsize=18)
plt.xlabel("Frecuencia [Hz]", fontsize=14)
plt.ylabel("Magnitud [dB]", fontsize=14)

plt.xlim(100, 15000) # f0 y f1 del código anterior
plt.ylim([-60, 5])      # Un rango de 60dB suele ser estándar para ver claridad
plt.grid(True, which="both")

plt.tight_layout()
plt.show()