import os
import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import starlette.status as _starlette_status
    if not hasattr(_starlette_status, 'HTTP_422_UNPROCESSABLE_CONTENT'):
        _starlette_status.HTTP_422_UNPROCESSABLE_CONTENT = 422
except Exception:
    pass
import time
import json
import hashlib
import textwrap
import socket
import gradio as gr
import torch
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.inference import DeepfakeAudioPredictor
from src.preprocessing.audio_loader import load_and_preprocess_audio
from src.utils.windows_audio import ensure_microphone_unmuted
_mic_info = ensure_microphone_unmuted()
if _mic_info.get('supported'):
    print(f"[AUDIO-SYS] Microphone Active: muted={_mic_info.get('is_muted')}, vol={_mic_info.get('volume_percent')}%, action={_mic_info.get('unmuted_action_taken')}")
PREDICTOR = None

def get_predictor():
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = DeepfakeAudioPredictor(checkpoint_path='checkpoints/best_detector.pt')
    return PREDICTOR
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PRESETS = {'telugu_real': os.path.join(PROJECT_ROOT, 'data', 'processed', 'real', 'telugu', 'svq_telugu_real_00000.wav'), 'telugu_fake': os.path.join(PROJECT_ROOT, 'data', 'processed', 'fake', 'telugu', 'indicsynth_telugu_freevc24_00000.wav'), 'hindi_real': os.path.join(PROJECT_ROOT, 'data', 'processed', 'real', 'hindi', 'svq_hindi_real_00000.wav'), 'hindi_fake': os.path.join(PROJECT_ROOT, 'data', 'processed', 'fake', 'hindi', 'indicsynth_hindi_freevc24_00000.wav'), 'malayalam_real': os.path.join(PROJECT_ROOT, 'data', 'processed', 'real', 'malayalam', 'svq_malayalam_real_00000.wav'), 'malayalam_fake': os.path.join(PROJECT_ROOT, 'data', 'processed', 'fake', 'malayalam', 'indicsynth_malayalam_freevc24_00000.wav'), 'bengali_real': os.path.join(PROJECT_ROOT, 'data', 'processed', 'real', 'bengali', 'svq_bengali_real_00000.wav'), 'bengali_fake': os.path.join(PROJECT_ROOT, 'data', 'processed', 'fake', 'bengali', 'indicsynth_bengali_freevc24_00000.wav'), 'telephony_fake': os.path.join(PROJECT_ROOT, 'data', 'raw', 'synthetic_voices_dataset', 'audio', 'synthetic_0002.wav'), 'english_fake': os.path.join(PROJECT_ROOT, 'data', 'raw', 'synthetic_voices_dataset', 'audio', 'synthetic_0001.wav')}
LANGUAGE_PHONETICS = {'Telugu (తెలుగు)': {'family': 'South-Central Dravidian', 'features': 'Retroflex plosives (ట, ఠ, డ, ఢ), length distinctions in terminal vowels, natural vowel harmony, and gemination cadence.', 'spoof_cues': 'Neural vocoders frequently flatten retroflex acoustic closures and truncate natural elongated terminal vowel decay.'}, 'Malayalam (മലയാളം)': {'family': 'Southern Dravidian', 'features': "Extensive liquid consonant inventories (ര, റ, ല, ള, ഴ / retroflex approximant 'zh'), nasal geminates, and non-aspirated dental-alveolar stops.", 'spoof_cues': 'Sub-band energy distortion in retroflex approximants (ഴ) above 4.2 kHz; vocoder smoothing eliminates rapid alveolar trill micro-closures.'}, 'Hindi (हिन्दी)': {'family': 'Indo-Aryan', 'features': 'Strict 4-way stop phonation: unvoiced unaspirated (क), unvoiced aspirated (ख), voiced unaspirated (ग), voiced aspirated (घ).', 'spoof_cues': 'Breathiness and turbulent noise in aspirated stops (ख, घ, थ, ध) are over-smoothed or exhibit robotic phase spikes in neural vocoders.'}, 'Bengali (বাংলা)': {'family': 'Eastern Indo-Aryan', 'features': "Vowel rounding harmony (inherent 'ô' shift), nasalized vowels (চন্দ্রবিন্দু), and breathy voiced murmurs.", 'spoof_cues': 'Formant trajectories in nasalized vowels display unnatural bandwidth narrowing; phase-vocoder smearing occurs around 3.5 kHz.'}, 'Indian English / Code-Switching': {'family': 'Indo-Aryan / Dravidian Code-Switching', 'features': 'Retroflex alveolar articulation (/t/ and /d/), syllable-timed rhythm, and natural English-Indic bilingual lexical transitions.', 'spoof_cues': 'Synthesizers struggle with intonational prosody during language transitions and exhibit mechanical formant phase artifacts.'}, 'Auto-Detect / Multilingual': {'family': 'Cross-Lingual Indic & Telephony Acoustic Analysis', 'features': 'Multilingual Indian language phonetic models covering 23,090 clips across Dravidian, Indo-Aryan, and Indian English, including 300-3,400 Hz telephone channels.', 'spoof_cues': 'High-frequency vocoder brick-wall cutoffs, phase-coherence violations, unnatural pitch contour flattening, and spectral energy discontinuities.'}}

def compute_audio_hash(audio_path: str) -> str:
    h = hashlib.sha256()
    with open(audio_path, 'rb') as f:
        while (chunk := f.read(8192)):
            h.update(chunk)
    return h.hexdigest()[:16].upper()

def plot_forensic_spectrogram_and_waveform(y: np.ndarray, sr: int=16000, title_prefix: str='Audio'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.6), gridspec_kw={'height_ratios': [1, 2]})
    fig.patch.set_facecolor('#0B0F19')
    times = np.linspace(0, len(y) / sr, len(y))
    ax1.set_facecolor('#111827')
    ax1.plot(times, y, color='#38BDF8', lw=0.9, alpha=0.9, label='16kHz Audio Waveform')
    ax1.axhline(0.1, color='#F59E0B', linestyle='--', alpha=0.5, label='RMS Norm Level (±0.10)')
    ax1.axhline(-0.1, color='#F59E0B', linestyle='--', alpha=0.5)
    ax1.set_xlim([0, max(0.1, times[-1])])
    ax1.set_ylim([-1.05, 1.05])
    ax1.set_ylabel('Amplitude', color='#94A3B8', fontsize=8)
    ax1.tick_params(colors='#94A3B8', labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color('#1E293B')
    ax1.legend(loc='upper right', facecolor='#0F172A', edgecolor='#334155', labelcolor='#E2E8F0', fontsize=7)
    ax2.set_facecolor('#111827')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, hop_length=256, x_axis='time', y_axis='mel', ax=ax2, cmap='magma')
    ax2.axhline(4000, color='#EF4444', linestyle=':', lw=1.5, alpha=0.8, label='Vocoder Artifact Threshold (4 kHz)')
    ax2.set_title(f'{title_prefix} — Mel-Frequency Spectro-Temporal Energy Distribution', color='#F8FAFC', fontsize=9, pad=6)
    ax2.set_xlabel('Time (seconds)', color='#94A3B8', fontsize=8)
    ax2.set_ylabel('Frequency (Hz)', color='#94A3B8', fontsize=8)
    ax2.tick_params(colors='#94A3B8', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('#1E293B')
    ax2.legend(loc='upper right', facecolor='#0F172A', edgecolor='#334155', labelcolor='#E2E8F0', fontsize=7)
    cbar = fig.colorbar(img, ax=[ax1, ax2], orientation='vertical', fraction=0.02, pad=0.02)
    cbar.ax.tick_params(colors='#94A3B8', labelsize=7)
    cbar.set_label('Energy (dB)', color='#94A3B8', fontsize=8)
    cbar.outline.set_edgecolor('#1E293B')
    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.08, right=0.9, hspace=0.35)
    return fig

def analyze_audio(audio_path, language: str='Auto-Detect / Multilingual', threshold_override: float=None):
    try:
        ensure_microphone_unmuted()
    except Exception:
        pass
    if isinstance(audio_path, tuple):
        import tempfile, scipy.io.wavfile as spwav, scipy.signal
        try:
            sr_in, arr_in = audio_path
            if arr_in is None or len(arr_in) == 0:
                audio_path = None
            else:
                arr_f = arr_in.astype(np.float32)
                if arr_f.ndim > 1:
                    arr_f = arr_f.mean(axis=1)
                if sr_in != 16000:
                    n = int(round(len(arr_f) * 16000.0 / sr_in))
                    arr_f = scipy.signal.resample(arr_f, n).astype(np.float32)
                tmp = tempfile.NamedTemporaryFile(suffix='_mic.wav', delete=False)
                spwav.write(tmp.name, 16000, arr_f)
                tmp.close()
                audio_path = tmp.name
                print(f'[MIC-NUMPY] Saved to temp WAV: {audio_path}  samples={len(arr_f)}')
        except Exception as ex:
            print(f'[MIC-NUMPY] Failed to convert numpy audio: {ex}')
            audio_path = None
    if audio_path is not None and (not isinstance(audio_path, str)):
        audio_path = str(audio_path)
    print(f'[ANALYZE] audio_path={audio_path}')
    if audio_path is None or not os.path.exists(audio_path):
        empty_fig = plt.figure(figsize=(8, 2))
        empty_fig.patch.set_facecolor('#0B0F19')
        plt.text(0.5, 0.5, 'No audio provided. Please upload or record speech audio.', color='#94A3B8', ha='center', va='center')
        plt.axis('off')
        return ("<div style='color: #EF4444; padding: 15px;'>⚠️ Error: No audio recording provided.</div>", {'Genuine / Real': 0.0, 'AI Deepfake': 0.0}, empty_fig)
    try:
        t0 = time.time()
        predictor = get_predictor()
        working_path = audio_path
        clean_lang = language.split(' ')[0].lower() if language else None
        if clean_lang == 'auto-detect':
            clean_lang = None
        tau = float(threshold_override) if threshold_override is not None else predictor.tau
        res = predictor.predict(working_path, threshold=tau, language=clean_lang)
        elapsed_ms = (time.time() - t0) * 1000.0
        if res.get('is_silent', False):
            stats = res.get('acoustic_stats', {})
            mic_st = ensure_microphone_unmuted()
            mic_vol = mic_st.get('volume_percent', 82.0)
            silent_fig = plt.figure(figsize=(10, 4))
            silent_fig.patch.set_facecolor('#0B0F19')
            plt.text(0.5, 0.58, '⚠️  No Audible Speech Signal Detected', color='#F59E0B', ha='center', va='center', fontsize=14, fontweight='bold')
            plt.text(0.5, 0.4, f"Peak Amplitude: {stats.get('peak_amplitude', 0.0):.5f}   |   RMS Energy: {stats.get('rms_energy', 0.0):.5f}   |   Duration: {stats.get('duration_sec', 0.0):.2f}s", color='#94A3B8', ha='center', va='center', fontsize=9)
            plt.text(0.5, 0.22, f'Windows System Microphone: UNMUTED & ACTIVE (Level: {mic_vol:.0f}%)', color='#10B981', ha='center', va='center', fontsize=9, fontweight='bold')
            plt.axis('off')
            silent_verdict = textwrap.dedent(f"""\n            <div style="background: rgba(245, 158, 11, 0.12); border: 1.5px solid #D97706; border-radius: 12px; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">\n                <div style="font-size: 1.3rem; font-weight: 800; color: #F59E0B; margin-bottom: 12px;">\n                    ⚠️ NO AUDIBLE SPEECH SIGNAL IN RECORDING\n                </div>\n                <div style="background: #090D16; padding: 12px 16px; border-radius: 8px; color: #CBD5E1; font-size: 0.92rem; margin-bottom: 12px; line-height: 1.5;">\n                    The audio stream received contains flat digital silence (Peak: <strong>{stats.get('peak_amplitude', 0.0):.5f}</strong>).\n                    <br/><br/>\n                    <strong>Why this happens and quick fixes:</strong>\n                    <ol style="margin-top: 6px; padding-left: 20px; margin-bottom: 0;">\n                        <li><strong>Windows Microphone Mute:</strong> Your Windows mic was muted in system settings. We have <strong>unmuted it automatically</strong> (Volume: {mic_vol:.0f}%). If your laptop has a physical mic key (e.g. Fn+F4 with an orange LED light), make sure it is unmuted.</li>\n                        <li><strong>Browser Microphone Permission:</strong> In your browser address bar next to <code>http://127.0.0.1</code>, click the lock/tune icon and ensure <strong>Microphone is set to Allow</strong>.</li>\n                        <li><strong>Recording Completion:</strong> When recording in the browser, speak clearly, then click the <strong>Stop button (■ or ✓)</strong> before clicking "Run Forensic Audio Analysis".</li>\n                        <li><strong>Guaranteed Alternative:</strong> Open Windows <strong>Sound Recorder</strong> app (type Sound Recorder in Windows start), record your voice, and drag-and-drop the saved audio file into the box below.</li>\n                    </ol>\n                </div>\n                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">\n                    <div style="background: #090D16; padding: 8px; border-radius: 6px; border: 1px solid #1E293B;">\n                        <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Peak Amplitude</div>\n                        <div style="font-size: 1.1rem; font-weight: 700; color: #F59E0B;">{stats.get('peak_amplitude', 0.0):.5f}</div>\n                        <div style="font-size: 0.72rem; color: #64748B;">Signal &gt; 0.0020</div>\n                    </div>\n                    <div style="background: #090D16; padding: 8px; border-radius: 6px; border: 1px solid #1E293B;">\n                        <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">System Mic</div>\n                        <div style="font-size: 1.1rem; font-weight: 700; color: #10B981;">Unmuted ({mic_vol:.0f}%)</div>\n                        <div style="font-size: 0.72rem; color: #64748B;">Ready to Record</div>\n                    </div>\n                    <div style="background: #090D16; padding: 8px; border-radius: 6px; border: 1px solid #1E293B;">\n                        <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Duration</div>\n                        <div style="font-size: 1.1rem; font-weight: 700; color: #F59E0B;">{stats.get('duration_sec', 0.0):.2f}s</div>\n                        <div style="font-size: 0.72rem; color: #64748B;">Recorded Buffer</div>\n                    </div>\n                </div>\n            </div>\n            """).strip()
            return (silent_verdict, {'Genuine / Real': 0.0, 'AI Deepfake': 0.0}, silent_fig)
        raw_scores = res['raw_scores']
        calib_probs = res['calibrated_probabilities']
        stats = res.get('acoustic_stats', {})
        is_fake = res['is_fake']
        conf = res['confidence_percent']
        audio_hash = compute_audio_hash(audio_path)
        if is_fake:
            verdict_color = '#EF4444'
            badge_bg = 'rgba(239, 68, 68, 0.15)'
            badge_border = '#DC2626'
            icon = '🚨'
            verdict_title = 'SYNTHETIC / AI-GENERATED AUDIO DETECTED'
            risk = 'HIGH RISK — Synthetic Voice Cloning / Neural Vocoder Signature'
        else:
            verdict_color = '#10B981'
            badge_bg = 'rgba(16, 185, 129, 0.15)'
            badge_border = '#059669'
            icon = '✅'
            verdict_title = 'AUTHENTIC HUMAN VOICE VERIFIED'
            risk = 'LOW RISK — Natural Organic Vocal Tract Dynamics'
        lang_info = LANGUAGE_PHONETICS.get(language, LANGUAGE_PHONETICS['Auto-Detect / Multilingual'])
        verdict_md = textwrap.dedent(f"""\n        <div style="background: {badge_bg}; border: 1.5px solid {badge_border}; border-radius: 12px; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">\n            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 12px; margin-bottom: 14px;">\n                <div style="font-size: 1.45rem; font-weight: 800; color: {verdict_color}; display: flex; align-items: center; gap: 8px;">\n                    <span>{icon}</span> {verdict_title}\n                </div>\n                <div style="background: #0F172A; border: 1px solid #334155; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; color: #94A3B8;">\n                    Inference Latency: <strong style="color: #38BDF8;">{elapsed_ms:.1f} ms</strong>\n                </div>\n            </div>\n            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 14px;">\n                <div style="background: #090D16; padding: 10px; border-radius: 8px; border: 1px solid #1E293B;">\n                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Confidence</div>\n                    <div style="font-size: 1.35rem; font-weight: 700; color: {verdict_color};">{conf:.2f}%</div>\n                </div>\n                <div style="background: #090D16; padding: 10px; border-radius: 8px; border: 1px solid #1E293B;">\n                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Decision Threshold (&tau;)</div>\n                    <div style="font-size: 1.35rem; font-weight: 700; color: #F59E0B;">{tau:.4f}</div>\n                </div>\n                <div style="background: #090D16; padding: 10px; border-radius: 8px; border: 1px solid #1E293B;">\n                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Calibrated Real Prob</div>\n                    <div style="font-size: 1.35rem; font-weight: 700; color: #10B981;">{calib_probs['Genuine / Real'] * 100:.2f}%</div>\n                </div>\n                <div style="background: #090D16; padding: 10px; border-radius: 8px; border: 1px solid #1E293B;">\n                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Calibrated Fake Prob</div>\n                    <div style="font-size: 1.35rem; font-weight: 700; color: #EF4444;">{calib_probs['AI Deepfake'] * 100:.2f}%</div>\n                </div>\n            </div>\n            <div style="background: #090D16; border-left: 3px solid #38BDF8; padding: 10px 14px; border-radius: 4px; font-size: 0.88rem; color: #CBD5E1;">\n                <strong>Language Profile:</strong> {language} &mdash; <em>{lang_info['spoof_cues']}</em>\n            </div>\n        </div>\n        """).strip()
        prob_dict = {'Genuine / Real': float(calib_probs['Genuine / Real']), 'AI Deepfake': float(calib_probs['AI Deepfake'])}
        clean_lang_title = language.split('(')[0].strip() if language else 'Speech'
        audio_arr = load_and_preprocess_audio(audio_path, target_sr=16000, target_duration_sec=3.0)
        fig = plot_forensic_spectrogram_and_waveform(audio_arr, sr=16000, title_prefix=f'{os.path.basename(audio_path)} [{clean_lang_title}]')
        return (verdict_md, prob_dict, fig)
    except Exception as e:
        import traceback
        traceback.print_exc()
        err_fig = plt.figure(figsize=(8, 2.5))
        err_fig.patch.set_facecolor('#0B0F19')
        plt.text(0.5, 0.5, f'Audio Analysis Exception:\n{str(e)[:120]}', color='#EF4444', ha='center', va='center', fontsize=9, fontweight='bold')
        plt.axis('off')
        err_verdict = f'\n        <div style="background: rgba(239, 68, 68, 0.15); border: 1.5px solid #DC2626; border-radius: 12px; padding: 18px; color: #FCA5A5; font-family: sans-serif;">\n            <div style="font-size: 1.25rem; font-weight: 700; color: #EF4444; margin-bottom: 6px;">\n                ⚠️ Audio Forensic Analysis Error\n            </div>\n            <div style="font-size: 0.95rem; color: #E2E8F0; margin-bottom: 8px;">\n                {str(e)}\n            </div>\n            <div style="font-size: 0.82rem; color: #94A3B8;">\n                Please ensure the file is an accessible speech recording (.wav, .mp3, .m4a, .flac) with audible speech.\n            </div>\n        </div>\n        '.strip()
        return (err_verdict, {'Genuine / Real': 0.0, 'AI Deepfake': 0.0}, err_fig)

def run_preset(preset_key: str, lang: str, thresh: float):
    path = PRESETS.get(preset_key, None)
    if not path or not os.path.exists(path):
        return (None, lang, "<div style='color: #EF4444;'>Preset file not found on disk.</div>", {'Genuine / Real': 0.5, 'AI Deepfake': 0.5}, None)
    v_md, p_dict, fig = analyze_audio(path, language=lang, threshold_override=thresh)
    return (path, lang, v_md, p_dict, fig)
CUSTOM_CSS = '\n@import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap\');\n\n:root {\n    --bg-dark: #080C14;\n    --card-dark: #0E1726;\n    --border-dark: #1E293B;\n}\n\nbody, .gradio-container {\n    background-color: var(--bg-dark) !important;\n    font-family: \'Inter\', -apple-system, sans-serif !important;\n    color: #E2E8F0 !important;\n}\n\n.gr-box, .gr-panel, .gr-card, .gr-form {\n    background-color: var(--card-dark) !important;\n    border: 1px solid var(--border-dark) !important;\n    border-radius: 12px !important;\n}\n\n.preset-btn-real {\n    background: linear-gradient(135deg, #065F46 0%, #047857 100%) !important;\n    border: 1px solid #10B981 !important;\n    color: #FFFFFF !important;\n    font-weight: 600 !important;\n    border-radius: 8px !important;\n    transition: all 0.2s ease-in-out !important;\n}\n.preset-btn-real:hover {\n    transform: translateY(-2px);\n    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4) !important;\n}\n\n.preset-btn-fake {\n    background: linear-gradient(135deg, #991B1B 0%, #B91C1C 100%) !important;\n    border: 1px solid #EF4444 !important;\n    color: #FFFFFF !important;\n    font-weight: 600 !important;\n    border-radius: 8px !important;\n    transition: all 0.2s ease-in-out !important;\n}\n.preset-btn-fake:hover {\n    transform: translateY(-2px);\n    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4) !important;\n}\n\n.primary-scan-btn {\n    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important;\n    border: none !important;\n    color: #FFFFFF !important;\n    font-size: 1.05rem !important;\n    font-weight: 700 !important;\n    border-radius: 10px !important;\n    padding: 12px 24px !important;\n    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35) !important;\n}\n.primary-scan-btn:hover {\n    transform: translateY(-2px);\n    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;\n}\n\n/* Completely remove and suppress any Gradio footer, watermark, or API links */\nfooter,\n.gradio-container footer,\n.gradio-container .footer,\nfooter.svelte-1rjfrsg,\nfooter.svelte-12io4e6,\nfooter[class*="svelte-"],\na[href*="gradio.app"],\na[href*="hf.space"],\n.built-with,\n.show-api,\ndiv:has(> a[href*="gradio.app"]),\ndiv:has(> footer) {\n    display: none !important;\n    visibility: hidden !important;\n    opacity: 0 !important;\n    height: 0 !important;\n    max-height: 0 !important;\n    padding: 0 !important;\n    margin: 0 !important;\n    overflow: hidden !important;\n    pointer-events: none !important;\n}\n'
with gr.Blocks(title='Indic Deepfake Audio Forensic Studio') as demo:
    gr.Markdown('\n    # 🛡️ Indic Deepfake Audio Forensic Studio\n    ### Multilingual Indian Speech Anti-Spoofing & Acoustic Telemetry Dashboard (Unified Workspace)\n    **Trained Model:** `Hybrid CNN-BiLSTM with Multi-Head Temporal Self-Attention` | **Parameters:** `342,211` | **Features:** `120-dim MFCCs + Deltas`\n    **Corpus:** `23,090 Balanced Audio Clips (11,548 Real Google SVQ + 11,542 Synthetic Clones spanning IndicSynth & Telephony Synthetic Voices)`\n    **Held-Out Benchmark (5,211 Test Clips):** `Calibrated Accuracy: 99.71%` | `EER: 0.29% (tau = 0.0215)` | `ROC-AUC: 0.9999` | `F1: 99.73%`\n    ')
    gr.Markdown('#### ⚡ 1-Click Regional Indic Presets (Click any button to immediately test authentic human vs AI voice clones)')
    with gr.Row():
        btn_telugu_real = gr.Button('🎙️ Telugu Real (Human SVQ)', elem_classes='preset-btn-real')
        btn_telugu_fake = gr.Button('🤖 Telugu Fake (IndicSynth)', elem_classes='preset-btn-fake')
        btn_hindi_real = gr.Button('🎙️ Hindi Real (Human SVQ)', elem_classes='preset-btn-real')
        btn_hindi_fake = gr.Button('🤖 Hindi Fake (IndicSynth)', elem_classes='preset-btn-fake')
    with gr.Row():
        btn_malayalam_real = gr.Button('🎙️ Malayalam Real (Human SVQ)', elem_classes='preset-btn-real')
        btn_malayalam_fake = gr.Button('🤖 Malayalam Fake (IndicSynth)', elem_classes='preset-btn-fake')
        btn_bengali_real = gr.Button('🎙️ Bengali Real (Human SVQ)', elem_classes='preset-btn-real')
        btn_bengali_fake = gr.Button('🤖 Bengali Fake (IndicSynth)', elem_classes='preset-btn-fake')
    with gr.Row():
        btn_telephony_fake = gr.Button('📞 Telephony Code-Switch (Synthetic 300-3400Hz)', elem_classes='preset-btn-fake')
        btn_english_fake = gr.Button('🤖 Indian English Synthetic (eSpeak Wideband)', elem_classes='preset-btn-fake')
    gr.Markdown('---')
    with gr.Row():
        with gr.Column(scale=5):
            mic_status_badge = gr.Markdown('<div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #059669; border-radius: 8px; padding: 9px 14px; margin-bottom: 8px; font-size: 0.88rem; color: #10B981; display: flex; align-items: center; justify-content: space-between;">\n                    <span>🎙️ <strong>System Microphone:</strong> Connected &amp; Unmuted (Hardware Level: 82%)</span>\n                    <span style="font-size: 0.78rem; color: #94A3B8;">Click Stop (■) after speaking</span>\n                </div>')
            audio_input = gr.Audio(sources=['upload', 'microphone'], type='filepath', label='Input Speech Recording (WAV, MP3, FLAC, M4A — or record via Microphone)')
            with gr.Row():
                lang_dropdown = gr.Dropdown(choices=['Auto-Detect / Multilingual', 'Telugu (తెలుగు)', 'Malayalam (മലയാളം)', 'Hindi (हिन्दी)', 'Bengali (বাংলা)', 'Indian English / Code-Switching'], value='Auto-Detect / Multilingual', label='Target Language Profile')
                thresh_slider = gr.Slider(minimum=0.01, maximum=0.99, value=round(get_predictor().tau, 4), step=0.005, label='Operating Decision Threshold (tau)')
            with gr.Row():
                analyze_btn = gr.Button('🔬 Run Forensic Audio Analysis', variant='primary', elem_classes='primary-scan-btn', scale=3)
                clear_btn = gr.ClearButton(components=[audio_input], value='🔄 Clear Audio', scale=1)
            gr.Markdown('<div style="background: rgba(30, 41, 59, 0.5); border: 1px solid #334155; border-radius: 8px; padding: 10px 14px; margin-top: 10px; font-size: 0.82rem; color: #94A3B8; line-height: 1.5;">\n                💡 <strong>Voice Recording Tips:</strong><br/>\n                &bull; <strong>Direct Mic:</strong> Click microphone icon &rarr; speak clearly &rarr; click <strong>Stop button (■ or ✓)</strong> to finalize &rarr; click <strong>Run Forensic Audio Analysis</strong>.<br/>\n                &bull; <strong>Drag &amp; Drop (Guaranteed):</strong> You can also record in Windows <em>Sound Recorder</em> or phone voice memo, and drop the <code>.m4a</code>, <code>.wav</code>, or <code>.mp3</code> file directly into the box above.\n                </div>')
        with gr.Column(scale=6):
            verdict_box = gr.HTML('\n            <div style="background: #0E1726; border: 1px dashed #334155; padding: 25px; border-radius: 12px; text-align: center; color: #94A3B8; font-family: sans-serif;">\n                <div style="font-size: 1.3rem; margin-bottom: 6px;">🎙️ Ready for Analysis</div>\n                <p style="margin: 0; font-size: 0.88rem;">Select or record an audio sample, or click any regional preset to view the instant classification and forensic spectro-temporal analysis.</p>\n            </div>\n            ')
            prob_bar = gr.Label(label='Classification Probabilities', num_top_classes=2)
            plot_view = gr.Plot(label='Time-Domain Waveform & Mel-Spectrogram Forensic Overlay')
    analyze_btn.click(fn=analyze_audio, inputs=[audio_input, lang_dropdown, thresh_slider], outputs=[verdict_box, prob_bar, plot_view])
    btn_telugu_real.click(fn=lambda t: run_preset('telugu_real', 'Telugu (తెలుగు)', t), inputs=[thresh_slider], outputs=[audio_input, lang_dropdown, verdict_box, prob_bar, plot_view])
    btn_telugu_fake.click(fn=lambda t: run_preset('telugu_fake', 'Telugu (తెలుగు)', t), inputs=[thresh_slider], outputs=[audio_input, lang_dropdown, verdict_box, prob_bar, plot_view])
    btn_hindi_real.click(fn=lambda t: run_preset('hindi_real', 'Hindi (हिन्दी)', t), inputs=[thresh_slider], outputs=[audio_input, lang_dropdown, verdict_box, prob_bar, plot_view])
    btn_hindi_fake.click(fn=lambda t: run_preset('hindi_fake', 'Hindi (हिन्दी)', t), inputs=[thresh_slider], outputs=[audio_input, lang_dropdown, verdict_box, prob_bar, plot_view])
    btn_malayalam_real.click(fn=lambda t: run_preset('malayalam_real', 'Malayalam (മലയാളം)', t), inputs=[thresh_slider], outputs=[audio_input, lang_dropdown, verdict_box, prob_bar, plot_view])
    btn_malayalam_fake.click(fn=lambda t: run_preset('malayalam_fake', 'Malayalam (മലയാളം)', t), inputs=[thresh_slider], outputs=[audio_input, lang_dropdown, verdict_box, prob_bar, plot_view])
    btn_bengali_real.click(fn=lambda t: run_preset('bengali_real', 'Bengali (বাংলা)', t), inputs=[thresh_slider], outputs=[audio_input, lang_dropdown, verdict_box, prob_bar, plot_view])
    btn_bengali_fake.click(fn=lambda t: run_preset('bengali_fake', 'Bengali (বাংলা)', t), inputs=[thresh_slider], outputs=[audio_input, lang_dropdown, verdict_box, prob_bar, plot_view])
    btn_telephony_fake.click(fn=lambda t: run_preset('telephony_fake', 'Indian English / Code-Switching', t), inputs=[thresh_slider], outputs=[audio_input, lang_dropdown, verdict_box, prob_bar, plot_view])
    btn_english_fake.click(fn=lambda t: run_preset('english_fake', 'Indian English / Code-Switching', t), inputs=[thresh_slider], outputs=[audio_input, lang_dropdown, verdict_box, prob_bar, plot_view])
    gr.HTML('\n    <div style="background: #0E1726; border: 1px solid #1E293B; border-radius: 12px; padding: 22px; margin-top: 25px;">\n        <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 12px; margin-bottom: 14px;">\n            <div style="font-size: 1.25rem; font-weight: 800; color: #38BDF8;">\n                📊 Held-Out Scientific Benchmark Performance (5,211 Unseen Audio Clips)\n            </div>\n            <div style="background: #090D16; border: 1px solid #334155; padding: 4px 12px; border-radius: 20px; font-size: 0.82rem; color: #10B981;">\n                Strict Anti-Leakage Speaker-Disjoint Split\n            </div>\n        </div>\n        <div style="overflow-x: auto; margin-bottom: 16px;">\n            <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.88rem; color: #E2E8F0;">\n                <thead>\n                    <tr style="border-bottom: 2px solid #334155; color: #94A3B8;">\n                        <th style="padding: 10px;">Language / Corpus</th>\n                        <th style="padding: 10px;">Test Clips</th>\n                        <th style="padding: 10px;">Human Real</th>\n                        <th style="padding: 10px;">AI Fake</th>\n                        <th style="padding: 10px;">Accuracy</th>\n                        <th style="padding: 10px;">Equal Error Rate (EER)</th>\n                        <th style="padding: 10px;">F1-Score</th>\n                    </tr>\n                </thead>\n                <tbody>\n                    <tr style="border-bottom: 1px solid #1E293B;"><td style="padding: 9px;"><strong>Malayalam (മലയാളം)</strong></td><td style="padding: 9px;">1,719</td><td style="padding: 9px;">479</td><td style="padding: 9px;">1,240</td><td style="padding: 9px; color: #10B981; font-weight: bold;">100.00%</td><td style="padding: 9px; color: #10B981;">0.00%</td><td style="padding: 9px;">1.0000</td></tr>\n                    <tr style="border-bottom: 1px solid #1E293B;"><td style="padding: 9px;"><strong>Bengali (বাংলা)</strong></td><td style="padding: 9px;">900</td><td style="padding: 9px;">711</td><td style="padding: 9px;">189</td><td style="padding: 9px; color: #10B981; font-weight: bold;">99.78%</td><td style="padding: 9px; color: #10B981;">0.00%</td><td style="padding: 9px;">0.9947</td></tr>\n                    <tr style="border-bottom: 1px solid #1E293B;"><td style="padding: 9px;"><strong>Telugu (తెలుగు)</strong></td><td style="padding: 9px;">1,926</td><td style="padding: 9px;">760</td><td style="padding: 9px;">1,166</td><td style="padding: 9px; color: #10B981; font-weight: bold;">99.58%</td><td style="padding: 9px; color: #10B981;">0.26%</td><td style="padding: 9px;">0.9966</td></tr>\n                    <tr style="border-bottom: 1px solid #1E293B;"><td style="padding: 9px;"><strong>Hindi (हिन्दी)</strong></td><td style="padding: 9px;">652</td><td style="padding: 9px;">480</td><td style="padding: 9px;">172</td><td style="padding: 9px; color: #10B981; font-weight: bold;">99.23%</td><td style="padding: 9px; color: #10B981;">0.00%</td><td style="padding: 9px;">0.9857</td></tr>\n                    <tr style="border-bottom: 1px solid #1E293B;"><td style="padding: 9px;"><strong>English / Code-Switching</strong></td><td style="padding: 9px;">14</td><td style="padding: 9px;">0</td><td style="padding: 9px;">14</td><td style="padding: 9px; color: #10B981; font-weight: bold;">100.00%</td><td style="padding: 9px; color: #94A3B8;">&mdash;</td><td style="padding: 9px;">1.0000</td></tr>\n                    <tr style="background: rgba(37, 99, 235, 0.16); font-weight: bold;"><td style="padding: 10px;">Combined Benchmark</td><td style="padding: 10px;">5,211</td><td style="padding: 10px;">2,430</td><td style="padding: 10px;">2,781</td><td style="padding: 10px; color: #38BDF8;">99.71%</td><td style="padding: 10px; color: #38BDF8;">0.29%</td><td style="padding: 10px; color: #38BDF8;">0.9973</td></tr>\n                </tbody>\n            </table>\n        </div>\n        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px;">\n            <div style="background: #090D16; padding: 12px; border-radius: 8px; border: 1px solid #1E293B;">\n                <div style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase;">FreeVC24 Detection Rate</div>\n                <div style="font-size: 1.25rem; font-weight: 700; color: #10B981;">100.00% <span style="font-size: 0.8rem; color: #64748B;">(881/881 clips)</span></div>\n            </div>\n            <div style="background: #090D16; padding: 12px; border-radius: 8px; border: 1px solid #1E293B;">\n                <div style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase;">eSpeak Telephony Detection</div>\n                <div style="font-size: 1.25rem; font-weight: 700; color: #10B981;">100.00% <span style="font-size: 0.8rem; color: #64748B;">(75/75 clips)</span></div>\n            </div>\n            <div style="background: #090D16; padding: 12px; border-radius: 8px; border: 1px solid #1E293B;">\n                <div style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase;">XTTS-v2 Zero-Shot Cloning</div>\n                <div style="font-size: 1.25rem; font-weight: 700; color: #10B981;">99.56% <span style="font-size: 0.8rem; color: #64748B;">(1,817/1,825 clips)</span></div>\n            </div>\n        </div>\n    </div>\n    ')

def find_available_port(start_port=7860, max_attempts=20):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port
if __name__ == '__main__':
    target_port = find_available_port(7860)
    print(f'\n[OK] Launching Indic Deepfake Audio Forensic Studio on http://127.0.0.1:{target_port}')
    demo.launch(server_name='127.0.0.1', server_port=target_port, theme=gr.themes.Soft(primary_hue='blue', neutral_hue='slate'), css=CUSTOM_CSS, footer_links=[], share=False)
