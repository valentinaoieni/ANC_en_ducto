import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import lfilter

"""
Script de ANC FEEDFORWARD

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

# --- CONFIGURACIÓN PARAMETRIZADA ---
FS = 44100              # Frecuencia de muestreo. Puede ser 44100 o 48000 ya que la biblioteca sound device no admite otras.
BLOCKSIZE = 2048        # Número de frames que se procesan luego de que se llene un buffer de esta capacidad. 
LATENCY = 'low'         # Sugerencia de latencia para el driver
DEVICE = 5# 'hw:2,0'
#print(sd.query_devices())

# Parámetros del Filtro Adaptativo (LMS)
N_TAPS = 2048                 # Longitud de la respuesta al impulso a estimar
MU = 0.000_0005               # Factor de convergencia (Step size)

DURACION_ESTIMACION = 70 # Segundos de duración del experimento

# leer coeff Sz
# Este resultado es previamente guardado de la estimación del camino secundario
# Cambiar por la ruta correspondiente
coeff_path = r'/home/alejo/Documentos/final_acustica/s_hat_coefs.npy'

feedback = r'/home/alejo/Documentos/final_acustica/feedback_path_coefs.npy'

# TONO
f_tono = 500

class ANC_feedforward:
    def __init__(self, n_taps, mu, h_sec):
        self.w = np.zeros(n_taps, dtype='float32')
        self.x_buffer = np.zeros(n_taps, dtype='float32')
        self.xf_buffer = np.zeros(n_taps, dtype='float32')
        self.mu = mu
        
        # Camino secundario estimado
        self.h_sec = h_sec

        
        # Estado del filtro del camino secundario
        self.sec_state = np.zeros(len(h_sec)-1, dtype='float32')


    
    def process_block(self, d_block, e_block):

        # Filtrar bloque por el camino secundario estimado
        x_filtered, self.sec_state = lfilter(
            self.h_sec, 
            1, 
            d_block, 
            zi=self.sec_state
        )
        
        anti_noise = np.zeros(len(d_block), dtype='float32')

        for i in range(len(d_block)):
            # Actualizar buffer de referencia
            self.x_buffer = np.roll(self.x_buffer, 1)
            self.x_buffer[0] = d_block[i]

            # Actualizar buffer de referencia FILTRADA por  S'(z)
            self.xf_buffer = np.roll(self.xf_buffer, 1)
            self.xf_buffer[0] = x_filtered[i]

            # Calcular salida
            y_n = np.dot(self.w, self.x_buffer)
            anti_noise[i] = y_n

            # Actualizar pesos (LMS con xf_buffer) - Elegir si hacer LMS normalizado o no
            norm = np.dot(self.xf_buffer, self.xf_buffer) + 1e-6
            # self.w += (self.mu / norm) * e_block[i] * self.xf_buffer
            self.w += self.mu * e_block[i] * self.xf_buffer
        return anti_noise

# Pre-generar el tono para no gastar CPU en la callback

t = np.linspace(0, DURACION_ESTIMACION, int(FS * DURACION_ESTIMACION), endpoint=False)
tono = 0.8 * np.sin(2 * np.pi * f_tono * t).astype('float32')
error = np.zeros(len(t))
source_noise = np.zeros(len(t))
anti_noise_u = np.zeros(len(t))

# Cargar coeficientes estimados del camino secundario     
# h = np.load('s_hat_coefs.npy')
h = np.loadtxt(coeff_path, delimiter=',')
f = np.loadtxt(feedback, delimiter=',')

idx = 0
estimator = ANC_feedforward(N_TAPS, MU, h, f)

def callback(indata, outdata, frames, time, status):
    global idx
    if status:
        print(f"Status: {status}") # Aquí verás si desaparece el overflow
    
    # Tomar segmento de ruido pre-generado
    x_n = tono[idx : idx + frames]
    outdata[:, 0] = x_n # Salida Canal 1 0 - LEFT / 1 - RIGHT
    
    # Entrada Mic Error
    d_n = tono[idx : idx + frames] # cambiar por indata[:, 1] si se desea tomar la entrada real del microfono de referencia
    e_n = indata[:, 0]
    
    # Computo de señal de control
    control_signal = estimator.process_block(d_n, e_n)

    # Actualización de variables y guardado histórico para ploteo
    outdata[:, 1] = control_signal
    error[idx:idx+frames] = e_n
    source_noise[idx:idx+frames] = d_n
    anti_noise_u[idx:idx+frames] = control_signal
    idx += frames
    

# ... resto del código de ejecución y plot ...

# --- EJECUCIÓN ---
print(f"Iniciando estimación del camino secundario por {DURACION_ESTIMACION}s...")
try:
    with sd.Stream(samplerate=FS,
                   blocksize=BLOCKSIZE,
                   device=DEVICE, # Cambiar por ID de Scarlett si es necesario
                #    channels=(2, 2), # In/Out
                   channels=(2, 2), # In/Out 1 == (1, 1), 2 = (2, 2)
                   dtype='float32',
                   latency=LATENCY,
                   callback=callback):
        sd.sleep(DURACION_ESTIMACION * 1000)
except Exception as e:
    print(f"Error: {e}")

# --- VISUALIZACIÓN DE RESULTADOS ---
print("Estimación finalizada. Generando gráficos...")

plt.figure(figsize=(12, 5))
plt.subplot(3, 1, 1)
plt.plot(t, error)
plt.title('Error ANC FeedForward', fontsize=22)
plt.xlabel('Tiempo', fontsize=18)
plt.ylabel('Amplitud', fontsize=18)
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(t, anti_noise_u, label='Accion Control')
plt.title('Acción de Control', fontsize=22)
plt.xlabel('Tiempo', fontsize=18)
plt.ylabel('Amplitud', fontsize=18)
plt.grid(True)
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(t, source_noise, label='Fuente de Ruido')
plt.title('Fuente de Ruido', fontsize=22)
plt.xlabel('Tiempo', fontsize=18)
plt.ylabel('Amplitud', fontsize=18)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
k = np.arange(0, len(estimator.w))
plt.figure(figsize=(12, 5))
plt.title('Coeficientes Filtro')
plt.plot(k, estimator.w)
plt.grid(True)
plt.show()