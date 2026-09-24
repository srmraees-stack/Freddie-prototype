from flask import Flask, request, jsonify, send_from_directory
from groq import Groq
import subprocess
import uuid
import os
import time
import wave
from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from the .env file next to app.py

app = Flask(__name__)
AUDIO_FOLDER = "audio_replies"
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# ======================================================
# CONFIG
# ======================================================
# Real Freddie API (not working properly yet, so Groq is used instead).
# FREDDIE_URL = "http://20.118.34.182:5088/api/robot/message"

# Set your key in the shell:  export GROQ_API_KEY="gsk_..."
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
GROQ_MODEL = "openai/gpt-oss-20b"

# Piper neural voice (download with: python -m piper.download_voices --download-dir voices en_US-lessac-medium)
VOICE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices", "en_US-lessac-medium.onnx")
try:
    from piper import PiperVoice
    piper_voice = PiperVoice.load(VOICE_MODEL)
except Exception as e:
    print(f"[tts] Piper unavailable, falling back to espeak: {e}")
    piper_voice = None

EMOTIONS = {"friendly", "concerned", "encouraging", "thinking"}
EMOTION_TO_ACTION = {
    "friendly": "nod",
    "concerned": "tilt",
    "encouraging": "nod",
    "thinking": "look_up",
}
FALLBACK_REPLY = "Sorry, I lost my train of thought for a moment. Could you say that again?"

MAX_HISTORY_TURNS = 6        # user+assistant pairs remembered per user
AUDIO_MAX_AGE_SECONDS = 600  # old reply files are deleted after this

SYSTEM_PROMPT = (
    "You are Freddie, a warm and encouraging AI life coach. "
    "Reply in 2-3 short spoken sentences, plain text, no markdown. "
    "You are a coach, not a therapist or doctor. If the user mentions "
    "self-harm, suicide, or being in danger, respond with care, encourage them "
    "to reach out to a trusted person or a local crisis line or emergency "
    "services, and use the concerned tone. "
    "After your reply, on a new line, write EMOTION: followed by one word "
    "describing your tone (friendly, concerned, encouraging, or thinking)."
)

# user_id -> list of {"role": ..., "content": ...}
histories = {}


# ======================================================
# STEP A — SPEECH-TO-TEXT (the "ears")
# ======================================================
def speech_to_text(text_input):
    # TODO: replace with real audio-to-text once the mic is wired up.
    # For now, /talk receives already-typed text, so just pass it through.
    return text_input


# ======================================================
# STEP B — THE "BRAIN" (Groq stand-in for Freddie)
# ======================================================
def parse_reply(full_text):
    """Split model output into (reply, emotion), tolerating messy formatting."""
    reply_part, _, emotion_part = (full_text or "").rpartition("EMOTION:")
    if not reply_part and not emotion_part.strip():
        return "", "friendly"
    if not reply_part:  # no EMOTION: marker present
        return emotion_part.strip(), "friendly"

    words = emotion_part.strip().lower().split()
    emotion = words[0].strip(".,!:;\"'*") if words else "friendly"
    if emotion not in EMOTIONS:
        emotion = "friendly"
    return reply_part.strip(), emotion


def call_freddie_local(user_text, user_id=14):
    history = histories.setdefault(user_id, [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-MAX_HISTORY_TURNS * 2:]
    messages.append({"role": "user", "content": user_text})

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            timeout=20,
        )
        reply, emotion = parse_reply(response.choices[0].message.content)
    except Exception as e:
        print(f"[groq error]   {e}")
        return {"reply": FALLBACK_REPLY, "emotion": "concerned", "action": "tilt"}

    if not reply:
        return {"reply": FALLBACK_REPLY, "emotion": "concerned", "action": "tilt"}

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    del history[:-MAX_HISTORY_TURNS * 2]

    return {
        "reply": reply,
        "emotion": emotion,
        "action": EMOTION_TO_ACTION[emotion],
    }


# ======================================================
# STEP C — CLEAN THE REPLY FOR SPEAKING
# ======================================================
def clean_for_speech(text):
    for symbol in ["**", "*", "#", "_", "`"]:
        text = text.replace(symbol, "")
    return text.strip()


# ======================================================
# STEP D — TEXT-TO-SPEECH (the "mouth")
# Piper (natural neural voice, offline), with espeak as a fallback
# ======================================================
def cleanup_old_audio():
    now = time.time()
    for name in os.listdir(AUDIO_FOLDER):
        path = os.path.join(AUDIO_FOLDER, name)
        try:
            if name.endswith(".wav") and now - os.path.getmtime(path) > AUDIO_MAX_AGE_SECONDS:
                os.remove(path)
        except OSError:
            pass


def _piper_tts(reply_text, filename):
    with wave.open(filename, "wb") as wav_file:
        piper_voice.synthesize_wav(reply_text, wav_file)


def _espeak_tts(reply_text, filename):
    # "--" stops text starting with "-" from being read as an option.
    subprocess.run(
        ["espeak", "-s", "150", "-v", "en-us+f3", "-w", filename, "--", reply_text],
        check=True,
        timeout=15,
    )


def text_to_speech(reply_text):
    """Write reply_text to a unique WAV file. Returns the path, or None on failure."""
    cleanup_old_audio()
    filename = os.path.join(AUDIO_FOLDER, f"reply_{uuid.uuid4().hex}.wav")
    engines = [_piper_tts] if piper_voice else []
    engines.append(_espeak_tts)
    for engine in engines:
        try:
            engine(reply_text, filename)
            return filename
        except Exception as e:
            print(f"[tts error]    {engine.__name__}: {e}")
    return None


# ======================================================
# THE ONE ENDPOINT
# ======================================================
@app.route("/talk", methods=["POST"])
def talk():
    data = request.get_json(silent=True) or {}
    user_text = str(data.get("text", "")).strip()
    user_id = data.get("user_id", 14)

    if not user_text:
        return jsonify({"error": "missing 'text'"}), 400

    heard = speech_to_text(user_text)
    print(f"[heard]        {heard}")

    freddie_response = call_freddie_local(heard, user_id=user_id)
    reply = freddie_response["reply"]
    emotion = freddie_response["emotion"]
    action = freddie_response["action"]

    print(f"[freddie says] {reply}")
    print(f"[emotion]      {emotion}, [action] {action}")

    clean_reply = clean_for_speech(reply)
    audio_path = text_to_speech(clean_reply)

    return jsonify({
        "reply": clean_reply,
        "emotion": emotion,
        "action": action,
        "audio_url": f"/audio/{os.path.basename(audio_path)}" if audio_path else None,
    })


@app.route("/audio/<path:filename>")
def get_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename, mimetype="audio/wav")


if __name__ == "__main__":
    # debug=False: Flask's debugger allows remote code execution on 0.0.0.0
    app.run(host="0.0.0.0", port=5000, debug=False)
