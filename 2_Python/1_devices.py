import sounddevice as sd

"""
Se debe conocer los dispositivos que estan reconociendo los controladores de sonido
Buscar el que corresponda a la Interfaz Dual Pre y el número de ID se utilizará en 
los otros códigos
"""

print(sd.query_devices())