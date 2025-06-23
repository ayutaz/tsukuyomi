#!/usr/bin/env python3
"""Streamlit Web UI for Tsukuyomi TTS"""

import streamlit as st
import torch
import numpy as np
import soundfile as sf
import io
import base64
from pathlib import Path
import json
import time
from typing import Dict, List, Optional

# Set page config
st.set_page_config(
    page_title="Tsukuyomi TTS - 月読",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .audio-player {
        margin: 1rem 0;
        padding: 1rem;
        background-color: #f0f0f0;
        border-radius: 10px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path: Optional[str] = None):
    """Load TTS model with caching"""
    try:
        from src.tsukuyomi_tts import TsukuyomiTTS
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TsukuyomiTTS(device=device)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            model.load_checkpoint(checkpoint_path)
            
        return model
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {e}")
        return None


def generate_audio(
    model,
    text: str,
    speaker_id: int = 0,
    emotion: str = "neutral",
    speed: float = 1.0,
    pitch_shift: float = 0.0,
    emotion_intensity: float = 1.0
) -> Optional[np.ndarray]:
    """Generate audio from text"""
    try:
        with st.spinner("音声を生成中..."):
            start_time = time.time()
            
            audio = model.synthesize(
                text=text,
                speaker_id=speaker_id,
                emotion=emotion,
                speed=speed,
                pitch_shift=pitch_shift,
                emotion_intensity=emotion_intensity
            )
            
            generation_time = time.time() - start_time
            
        return audio, generation_time
    except Exception as e:
        st.error(f"音声生成エラー: {e}")
        return None, 0


def audio_to_base64(audio: np.ndarray, sample_rate: int = 24000) -> str:
    """Convert audio array to base64 string for HTML audio player"""
    # Convert to WAV in memory
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)
    
    # Convert to base64
    audio_base64 = base64.b64encode(buffer.read()).decode()
    return f"data:audio/wav;base64,{audio_base64}"


def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🌙 Tsukuyomi TTS")
        st.markdown("### 高品質日本語音声合成システム")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # Model selection
        st.subheader("モデル設定")
        model_path = st.text_input(
            "モデルパス",
            value="checkpoints/best_model.pt",
            help="学習済みモデルのパス"
        )
        
        # Speaker settings
        st.subheader("話者設定")
        speaker_id = st.selectbox(
            "話者",
            options=list(range(10)),
            format_func=lambda x: f"話者 {x+1}"
        )
        
        # Emotion settings
        st.subheader("感情設定")
        emotion = st.selectbox(
            "感情",
            options=[
                "neutral", "happy", "sad", "angry", "fearful",
                "surprised", "disgusted", "excited", "calm", "confident"
            ],
            format_func=lambda x: {
                "neutral": "😐 中立",
                "happy": "😊 喜び",
                "sad": "😢 悲しみ",
                "angry": "😠 怒り",
                "fearful": "😨 恐れ",
                "surprised": "😲 驚き",
                "disgusted": "🤢 嫌悪",
                "excited": "🤗 興奮",
                "calm": "😌 穏やか",
                "confident": "😎 自信"
            }.get(x, x)
        )
        
        emotion_intensity = st.slider(
            "感情の強さ",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1
        )
        
        # Voice settings
        st.subheader("音声設定")
        speed = st.slider(
            "話速",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            format="%.1fx"
        )
        
        pitch_shift = st.slider(
            "ピッチシフト",
            min_value=-12.0,
            max_value=12.0,
            value=0.0,
            step=0.5,
            format="%.1f半音"
        )
        
        # Advanced settings
        with st.expander("詳細設定"):
            noise_scale = st.slider(
                "ノイズスケール",
                min_value=0.0,
                max_value=1.0,
                value=0.667,
                step=0.01
            )
            
            length_scale = st.slider(
                "長さスケール",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
    
    # Main content
    st.header("📝 テキスト入力")
    
    # Text input tabs
    tab1, tab2, tab3 = st.tabs(["シンプル入力", "バッチ処理", "SSML入力"])
    
    with tab1:
        text_input = st.text_area(
            "合成したいテキストを入力してください",
            value="月読は、最先端の日本語音声合成システムです。",
            height=100,
            max_chars=500
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            generate_button = st.button("🎵 音声生成", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ クリア", use_container_width=True)
            
    with tab2:
        batch_text = st.text_area(
            "複数のテキスト（1行1文）",
            value="こんにちは、月読です。\n今日はいい天気ですね。\n音声合成を楽しんでください。",
            height=200
        )
        batch_generate = st.button("🎵 バッチ生成", type="primary")
        
    with tab3:
        ssml_input = st.text_area(
            "SSML形式での入力",
            value='<speak>こんにちは。<break time="0.5s"/>月読です。</speak>',
            height=150
        )
        ssml_generate = st.button("🎵 SSML生成", type="primary")
    
    # Load model
    model = load_model(model_path if Path(model_path).exists() else None)
    
    # Generate audio
    if generate_button and text_input and model:
        audio, gen_time = generate_audio(
            model,
            text_input,
            speaker_id,
            emotion,
            speed,
            pitch_shift,
            emotion_intensity
        )
        
        if audio is not None:
            # Display results
            st.header("🎧 生成結果")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("生成時間", f"{gen_time:.2f}秒")
            with col2:
                st.metric("音声長", f"{len(audio)/24000:.2f}秒")
            with col3:
                st.metric("RTF", f"{gen_time/(len(audio)/24000):.3f}")
            with col4:
                st.metric("サンプルレート", "24kHz")
            
            # Audio player
            st.markdown('<div class="audio-player">', unsafe_allow_html=True)
            st.audio(audio, format='audio/wav', sample_rate=24000)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download button
            buffer = io.BytesIO()
            sf.write(buffer, audio, 24000, format='WAV')
            buffer.seek(0)
            
            st.download_button(
                label="📥 音声をダウンロード",
                data=buffer,
                file_name=f"tsukuyomi_{int(time.time())}.wav",
                mime="audio/wav"
            )
            
            # Waveform visualization
            with st.expander("波形表示"):
                st.line_chart(audio[:1000])  # Show first 1000 samples
                
            # Store in session state
            if 'history' not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                'text': text_input,
                'speaker': speaker_id,
                'emotion': emotion,
                'audio': audio,
                'time': time.time()
            })
    
    # Clear button
    if clear_button:
        st.session_state.clear()
        st.experimental_rerun()
    
    # History
    if 'history' in st.session_state and st.session_state.history:
        st.header("📜 生成履歴")
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"履歴 {i+1}: {item['text'][:30]}..."):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"話者: {item['speaker']+1}, 感情: {item['emotion']}")
                    st.audio(item['audio'], format='audio/wav', sample_rate=24000)
                with col2:
                    buffer = io.BytesIO()
                    sf.write(buffer, item['audio'], 24000, format='WAV')
                    buffer.seek(0)
                    st.download_button(
                        "📥 保存",
                        data=buffer,
                        file_name=f"history_{i+1}.wav",
                        mime="audio/wav",
                        key=f"download_{i}"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Made with ❤️ by Tsukuyomi Team | "
        "[GitHub](https://github.com/ayutaz/tsukuyomi) | "
        "[Documentation](https://github.com/ayutaz/tsukuyomi/wiki)"
    )


if __name__ == "__main__":
    main()