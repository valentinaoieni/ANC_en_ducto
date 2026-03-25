import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import time

"""
Estimación del camino secundario offline (para ANC en configuración Feedforward remitirse a otro script)
Autores:
- Suárez, Alejo
- Oieni Orozco, Valentina

Se define una clase de objeto RobustEstimator, que contiene también una función, esta es el algoritmo FxLMS
Esta funcion se ejecuta cada vez que se llena el buffer de entrada con las muestras del sonido captado.
La librería sounddevice gestiona esto, y llama a la función "callback" cada vez que esto sucede. Dentro de
esta función callback está contenido el método de computar por FxLMS, la señal de control y la adaptación
del camino secundario

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
            En la estimación del camino secundario puede ser un valor no tan pequeño, en el orden de 1e-1

DURACION_ESTIMACION = Segundos de duración del experimento
            
IMPORTANTE: Antes de realizar el experimento, comprobar que:
-el parlante actuador esté en el canal 0 
-el parlante de ruido esté en el canal 1
-que el micrófono de error, esté en el canal 0
-que el micrófono de ruido, esté en el canal 1

"""

# --- CONFIGURACIÓN ---
FS = 44100
BLOCKSIZE = 0   # Aumentamos un poco el bloque para dar respiro a la CPU
N_TAPS = 2048    # Aumentamos a 5120 para que los 64ms (3072) queden cómodos al medio
MU = 0.01           # Bajamos MU para mayor estabilidad en el mundo real
EPS = 1e-3
duracion = 30
DEVICE = 5 #'hw:2,0'
flag = True

print(sd.query_devices())

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
            # 1. Shifteo eficiente (hacia la izquierda para mantener orden temporal)
            self.x_buffer = np.roll(self.x_buffer, 1)
            self.x_buffer[0] = x_block[i]
            
            # 2. Predicción
            y_n = np.dot(self.w, self.x_buffer)
            e_n = d_block[i] - y_n
            
            # 3. NLMS: Normalización por energía del buffer
            norm_x = np.dot(self.x_buffer, self.x_buffer) + self.eps
            
            # 4. Actualización (El corazón del aprendizaje)
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

    # IMPORTANTE: Antes de realizar el experimento, comprobar que:
    # -el parlante actuador esté en el canal 0 
    # -el parlante de ruido esté en el canal 1
    # -que el micrófono de error, esté en el canal 0
    # -que el micrófono de ruido, esté en el canal 1

    d_chunk = indata[:, 1]  # tomar micrófono de ruido (de referencia de la fuente)
    x_chunk = noise[idx : idx + frames]
    
    _ = est.process_block(x_chunk, d_chunk)
    # print(len(d_chunk))
    
    outdata[:, 0] = x_chunk # Sale por el parlante actuador
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
# plt.plot(20 * np.log10(np.abs(est.error_history) + 1e-7))
plt.plot(est.error_history)
plt.title("Historial del Error")
plt.tight_layout()
plt.show()

file_path = r'/home/alejo/Documentos/final_acustica/feedback_path_coefs.npy'

np.savetxt(file_path, est.w, delimiter=',')