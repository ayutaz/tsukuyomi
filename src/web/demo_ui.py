"""
Tsukuyomi Demo UI - Real-time web interface for TTS demonstration

Features:
- Real-time synthesis
- Voice morphing controls
- Waveform visualization
- Download capabilities
"""

import io
from typing import List, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import requests
import soundfile as sf

# API configuration
API_URL = "http://localhost:8000"

# Available options
EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "fear", "disgust"]
STYLES = [
    "normal",
    "energetic",
    "calm",
    "dramatic",
    "whisper",
    "shout",
    "formal",
    "casual",
]
SPEAKERS = {
    "Default Female": 0,
    "Male Voice 1": 1,
    "Female Voice 2": 2,
    "Male Voice 2": 3,
    "Child Voice": 4,
    "Elder Male": 5,
    "Elder Female": 6,
    "Anime Female": 7,
    "Anime Male": 8,
    "Narrator": 9,
}


def synthesize_speech(
    text: str,
    speaker: str,
    emotion: str,
    style: str,
    speed: float,
    pitch: float,
    energy: float,
) -> Tuple[int, np.ndarray]:
    """Synthesize speech from text."""
    try:
        response = requests.post(
            f"{API_URL}/synthesize",
            json={
                "text": text,
                "speaker_id": SPEAKERS[speaker],
                "emotion": emotion,
                "style": style,
                "speed": speed,
                "pitch_shift": pitch,
                "energy": energy,
                "output_format": "wav",
            },
        )

        if response.status_code == 200:
            # Convert response to audio
            audio_data = response.content
            audio, sr = sf.read(io.BytesIO(audio_data))
            return sr, audio
        else:
            return None, None

    except Exception as e:
        print(f"Error: {e}")
        return None, None


def morph_voices(
    text: str,
    speakers: List[str],
    weights: str,
    emotions: List[str],
    emotion_weights: str,
    speed: float,
) -> Tuple[int, np.ndarray]:
    """Morph multiple voices."""
    try:
        # Parse weights
        speaker_weights = [float(w.strip()) for w in weights.split(",")]
        emotion_weights_list = None

        if emotions and emotion_weights:
            emotion_weights_list = [
                float(w.strip()) for w in emotion_weights.split(",")
            ]

        # Get speaker IDs
        speaker_ids = [SPEAKERS[s] for s in speakers]

        response = requests.post(
            f"{API_URL}/morph",
            json={
                "text": text,
                "speaker_ids": speaker_ids,
                "speaker_weights": speaker_weights,
                "emotion_ids": emotions if emotions else None,
                "emotion_weights": emotion_weights_list,
                "speed": speed,
                "output_format": "wav",
            },
        )

        if response.status_code == 200:
            audio_data = response.content
            audio, sr = sf.read(io.BytesIO(audio_data))
            return sr, audio
        else:
            return None, None

    except Exception as e:
        print(f"Error: {e}")
        return None, None


def create_waveform_plot(audio: np.ndarray, sr: int) -> plt.Figure:
    """Create waveform visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # Waveform
    time = np.arange(len(audio)) / sr
    ax1.plot(time, audio, color="blue", alpha=0.7)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Waveform")
    ax1.grid(True, alpha=0.3)

    # Spectrogram
    from scipy import signal

    f, t, Sxx = signal.spectrogram(audio, sr, nperseg=1024)
    ax2.pcolormesh(
        t,
        f[:8000],
        10 * np.log10(Sxx[: len(f[f < 8000])]),
        shading="gouraud",
        cmap="viridis",
    )
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Spectrogram")

    plt.tight_layout()
    return fig


def build_demo():
    """Build Gradio demo interface."""

    with gr.Blocks(title="Tsukuyomi TTS Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🌙 Tsukuyomi TTS Demo
            
            Ultimate Japanese Text-to-Speech System with voice morphing capabilities.
            """
        )

        with gr.Tabs():
            # Basic Synthesis Tab
            with gr.TabItem("🎤 Basic Synthesis"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(
                            label="Text to synthesize",
                            placeholder="月読は最高峰の音声合成システムです",
                            lines=3,
                        )

                    with gr.Column(scale=1):
                        synthesize_btn = gr.Button("Synthesize", variant="primary")

                with gr.Row():
                    speaker_dropdown = gr.Dropdown(
                        choices=list(SPEAKERS.keys()),
                        value="Default Female",
                        label="Speaker",
                    )
                    emotion_dropdown = gr.Dropdown(
                        choices=EMOTIONS, value="neutral", label="Emotion"
                    )
                    style_dropdown = gr.Dropdown(
                        choices=STYLES, value="normal", label="Style"
                    )

                with gr.Row():
                    speed_slider = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed"
                    )
                    pitch_slider = gr.Slider(
                        minimum=-12,
                        maximum=12,
                        value=0,
                        step=1,
                        label="Pitch (semitones)",
                    )
                    energy_slider = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Energy"
                    )

                with gr.Row():
                    audio_output = gr.Audio(label="Generated Audio", type="numpy")

                with gr.Row():
                    waveform_plot = gr.Plot(label="Waveform Analysis")

                # Event handlers
                def on_synthesize(text, speaker, emotion, style, speed, pitch, energy):
                    sr, audio = synthesize_speech(
                        text, speaker, emotion, style, speed, pitch, energy
                    )
                    if audio is not None:
                        fig = create_waveform_plot(audio, sr)
                        return (sr, audio), fig
                    return None, None

                synthesize_btn.click(
                    fn=on_synthesize,
                    inputs=[
                        text_input,
                        speaker_dropdown,
                        emotion_dropdown,
                        style_dropdown,
                        speed_slider,
                        pitch_slider,
                        energy_slider,
                    ],
                    outputs=[audio_output, waveform_plot],
                )

            # Voice Morphing Tab
            with gr.TabItem("🎭 Voice Morphing"):
                with gr.Row():
                    morph_text = gr.Textbox(
                        label="Text to synthesize",
                        placeholder="複数の声をブレンドします",
                        lines=3,
                    )

                with gr.Row():
                    speakers_select = gr.CheckboxGroup(
                        choices=list(SPEAKERS.keys()),
                        value=["Default Female", "Male Voice 1"],
                        label="Select Speakers to Morph",
                    )

                with gr.Row():
                    weights_input = gr.Textbox(
                        label="Speaker Weights (comma-separated)",
                        placeholder="0.7, 0.3",
                        value="0.5, 0.5",
                    )

                with gr.Row():
                    emotions_select = gr.CheckboxGroup(
                        choices=EMOTIONS, label="Select Emotions to Morph (optional)"
                    )

                with gr.Row():
                    emotion_weights_input = gr.Textbox(
                        label="Emotion Weights (comma-separated)",
                        placeholder="0.6, 0.4",
                    )

                with gr.Row():
                    morph_speed = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed"
                    )
                    morph_btn = gr.Button("Morph Voices", variant="primary")

                with gr.Row():
                    morph_audio = gr.Audio(label="Morphed Audio", type="numpy")

                with gr.Row():
                    morph_plot = gr.Plot(label="Morphed Waveform")

                # Morphing handler
                def on_morph(text, speakers, weights, emotions, emotion_weights, speed):
                    sr, audio = morph_voices(
                        text, speakers, weights, emotions, emotion_weights, speed
                    )
                    if audio is not None:
                        fig = create_waveform_plot(audio, sr)
                        return (sr, audio), fig
                    return None, None

                morph_btn.click(
                    fn=on_morph,
                    inputs=[
                        morph_text,
                        speakers_select,
                        weights_input,
                        emotions_select,
                        emotion_weights_input,
                        morph_speed,
                    ],
                    outputs=[morph_audio, morph_plot],
                )

            # Examples Tab
            with gr.TabItem("📚 Examples"):
                gr.Markdown(
                    """
                    ## Example Texts
                    
                    ### Basic Japanese
                    - こんにちは、月読です。今日はいい天気ですね。
                    - 人工知能による音声合成技術は、日々進化しています。
                    - お電話ありがとうございます。ただいま担当者が不在です。
                    
                    ### Emotional Expressions
                    - わあ！すごく嬉しいです！ (happy)
                    - どうして...そんなことを言うの？ (sad)
                    - もう！いい加減にしてください！ (angry)
                    
                    ### Different Styles
                    - 本日は晴天なり、マイクのテスト中です。 (formal)
                    - ねえねえ、聞いて聞いて！ (casual)
                    - ひそひそ...これは秘密の話なんだけど... (whisper)
                    
                    ### Voice Morphing Examples
                    - Try blending "Default Female" (70%) with "Male Voice 1" (30%)
                    - Mix emotions: "happy" (60%) with "excited" (40%)
                    - Create unique character voices by morphing multiple speakers
                    """
                )

            # Settings Tab
            with gr.TabItem("⚙️ Settings"):
                gr.Markdown(
                    """
                    ## API Settings
                    
                    Current API endpoint: `http://localhost:8000`
                    
                    ## Audio Settings
                    - Sample Rate: 48,000 Hz
                    - Bit Depth: 16-bit
                    - Format: WAV
                    
                    ## Performance
                    - RTF: < 0.05 (20x faster than real-time)
                    - Latency: ~60ms
                    - GPU: CUDA enabled
                    """
                )

                with gr.Row():
                    api_status = gr.Textbox(
                        label="API Status", value="Checking...", interactive=False
                    )
                    check_btn = gr.Button("Check Connection")

                def check_api():
                    try:
                        response = requests.get(f"{API_URL}/health")
                        if response.status_code == 200:
                            data = response.json()
                            return f"✅ Connected - Device: {data['device']}"
                        else:
                            return "❌ Connection failed"
                    except:
                        return "❌ API server not running"

                check_btn.click(fn=check_api, outputs=api_status)
                demo.load(fn=check_api, outputs=api_status)

        gr.Markdown(
            """
            ---
            
            <div style="text-align: center;">
                <p>🌙 Tsukuyomi TTS - Ultimate Japanese Text-to-Speech System</p>
                <p>Made with ❤️ for the Japanese TTS community</p>
            </div>
            """
        )

    return demo


def main():
    """Launch the demo UI."""
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()
