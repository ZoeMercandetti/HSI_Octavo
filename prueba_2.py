# =========================
# HMI para procesamiento de señales de audio
# =========================

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from pydub import AudioSegment
from scipy.io.wavfile import write as wav_write


# =========================
# Carga de audio desde archivo
# =========================
def cargar_audio(file):
    ext = os.path.splitext(file)[1].lower()

    if ext == ".wav":
        sr, data = wavfile.read(file)
    else:
        audio = AudioSegment.from_file(file).set_channels(1)
        sr = audio.frame_rate
        data = np.array(audio.get_array_of_samples())

    data = data.astype(np.float32)

    if len(data) < 20:
        reps = int(np.ceil(20 / len(data)))
        data = np.tile(data, reps)

    tiempo = np.linspace(0, len(data)/sr, len(data))
    plt.figure(figsize=(10, 3))
    plt.plot(tiempo, data, color='hotpink')
    plt.title("Señal Original")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.tight_layout()
    temp_img_path = "/tmp/original_signal.png"
    plt.savefig(temp_img_path)
    plt.close()

    return sr, data.tolist(), temp_img_path, data.tolist()


# =========================
# Aplicación de filtros con parámetros configurables
# =========================
def aplicar_filtro(data, sr, tipo, f1, f2, orden):
    data = np.array(data)

    if tipo == "pasa-bajas":
        b, a = butter(orden, f1 / (0.5 * sr), btype='low')
    elif tipo == "pasa-altas":
        b, a = butter(orden, f1 / (0.5 * sr), btype='high')
    elif tipo == "pasa-banda":
        b, a = butter(orden, [f1 / (0.5 * sr), f2 / (0.5 * sr)], btype='band')
    else:
        return None, data.tolist()

    filtrada = filtfilt(b, a, data)

    tiempo = np.linspace(0, len(data)/sr, len(data))
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(tiempo, data, color='hotpink')
    plt.title("Señal Original")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")

    plt.subplot(2, 1, 2)
    plt.plot(tiempo, filtrada, color='purple')
    plt.title(f"Señal Filtrada ({tipo})")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")

    plt.tight_layout()
    temp_img_path = "/tmp/comparacion.png"
    plt.savefig(temp_img_path)
    plt.close()

    return temp_img_path, filtrada.tolist()


# =========================
# Transformada de Fourier - 4 gráficas en 1 imagen
# =========================
def aplicar_fft(filtrada, sr, original_data):
    if original_data is None:
        original_data = filtrada
    

    filtrada = np.array(filtrada)
    original = np.array(original_data)

    # Normalizamos ambas señales
    original = original / np.max(np.abs(original))
    filtrada = filtrada / np.max(np.abs(filtrada))

    N_orig = len(original)
    N_filt = len(filtrada)
    tiempo_orig = np.linspace(0, N_orig / sr, N_orig)
    tiempo_filt = np.linspace(0, N_filt / sr, N_filt)

    fft_original = np.fft.rfft(original)
    freqs_orig = np.fft.rfftfreq(N_orig, 1 / sr)
    mag_original = np.abs(fft_original)

    fft_filtrada = np.fft.rfft(filtrada)
    freqs_filt = np.fft.rfftfreq(N_filt, 1 / sr)
    mag_filtrada = np.abs(fft_filtrada)

    # Graficamos todo junto
    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(tiempo_orig, original, color='hotpink')
    plt.title("Señal Original")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")

    plt.subplot(4, 1, 2)
    plt.plot(freqs_orig, mag_original, color='blue')
    plt.title("Transformada de Fourier - Señal Original")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")

    plt.subplot(4, 1, 3)
    plt.plot(tiempo_filt, filtrada, color='purple')
    plt.title("Señal Filtrada")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")

    plt.subplot(4, 1, 4)
    plt.plot(freqs_filt, mag_filtrada, color='darkorange')
    plt.title("Transformada de Fourier - Señal Filtrada")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")

    plt.tight_layout()
    temp_fft_path = "/tmp/fft_completa_4graficas.png"
    plt.savefig(temp_fft_path)
    plt.close()

    return temp_fft_path


# =========================
# Exportar audio como archivo .wav
# =========================
def guardar_audio(data, sr):
    data = np.array(data)
    data = np.clip(data, -32768, 32767).astype(np.int16)
    export_path = "/tmp/audio_filtrado.wav"
    wav_write(export_path, sr, data)
    return export_path


# =========================
# Interfaz Gradio completa
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## 🎧 Procesador de Audio - Filtros & FFT")

    with gr.Row():
        audio_input = gr.Audio(label="📂 Cargar archivo de audio", type="filepath")
        load_btn = gr.Button("Cargar Audio")

    signal_plot = gr.Image(label="📈 Gráfica")

    with gr.Row():
        filter_type = gr.Dropdown(["pasa-bajas", "pasa-altas", "pasa-banda"], label="Tipo de filtro", value="pasa-bajas")
        freq1_slider = gr.Slider(100, 5000, value=1000, step=100, label="Frecuencia Corte 1 (Hz)")
        freq2_slider = gr.Slider(500, 6000, value=1500, step=100, label="Frecuencia Corte 2 (Hz, solo pasa-banda)")
        order_slider = gr.Slider(1, 10, value=5, step=1, label="Orden del Filtro")

    with gr.Row():
        apply_btn = gr.Button("Aplicar Filtro")
        fft_btn = gr.Button("Aplicar Transformada")
        export_btn = gr.Button("Guardar Resultado")

    export_output = gr.File(label="🔊 Archivo Exportado (.wav)")

    # Estados compartidos
    sr_state = gr.State()
    data_state = gr.State()
    original_data_state = gr.State()

    # Conexiones
    load_btn.click(fn=cargar_audio, inputs=audio_input, outputs=[sr_state, data_state, signal_plot, original_data_state])
    apply_btn.click(fn=aplicar_filtro, inputs=[data_state, sr_state, filter_type, freq1_slider, freq2_slider, order_slider], outputs=[signal_plot, data_state])
    fft_btn.click(fn=aplicar_fft, inputs=[data_state, sr_state, original_data_state], outputs=signal_plot)
    export_btn.click(fn=guardar_audio, inputs=[data_state, sr_state], outputs=export_output)

# Lanza la interfaz
demo.launch()