import numpy as np
import wave
import os

# Frequências das 10 notas (Hz)
notas = [261.63, 293.66, 329.63, 349.23, 392.00, 
         440.00, 493.88, 523.25, 587.33, 659.25]  
# (C4, D4, E4, F4, G4, A4, B4, C5, D5, E5)

def gerar_onda(freq, duracao=0.5, sample_rate=44100):
    t = np.linspace(0, duracao, int(sample_rate*duracao), False)
    onda = 0.5 * np.sin(2 * np.pi * freq * t)
    onda = np.int16(onda * 32767)
    return onda

for i, freq in enumerate(notas, start=1):
    onda = gerar_onda(freq)
    filename = os.path.join("sons", f"som{i}.wav")
    with wave.open(filename, "w") as f:
        f.setnchannels(1)  # mono
        f.setsampwidth(2)  # 16 bits
        f.setframerate(44100)
        f.writeframes(onda.tobytes())
    print(f"{filename} gerado!")
