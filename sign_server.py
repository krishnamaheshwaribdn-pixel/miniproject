import io
import json
import os
import threading
import uuid
import urllib.error
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
APP_DATA_DIR = Path(
    os.environ.get(
        "CHATBOT_DATA_DIR",
        os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"),
    )
) / "ChatbotGesture"
WEBCAM_DATASET_DIR = APP_DATA_DIR / "webcam-dataset"
TRAINED_MODEL_PATH = Path(
    os.environ.get(
        "CHATBOT_SIGN_MODEL",
        BASE_DIR / "models" / "sign_model.keras",
    )
)
TRAINED_LABELS_PATH = TRAINED_MODEL_PATH.with_suffix(".labels.json")
YOLO_HAND_MODEL_PATH = Path(
    os.environ.get(
        "CHATBOT_YOLO_MODEL",
        BASE_DIR / "models" / "yolo11n_hand_best.pt",
    )
)
MAX_FRAME_BYTES = 2 * 1024 * 1024
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
CHAT_SYSTEM_PROMPT = (
    'You are Chatbot, an advanced AI assistant: precise, helpful, and concise. '
    'Address the user as "Sir" occasionally. Solve math with clear numbered '
    "steps. Use plain text and avoid markdown formatting."
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FRAME_BYTES
model = None


class SignModel:
    def __init__(self):
        import tensorflow as tensorflow
        import numpy as np
        from ultralytics import YOLO

        tf = tensorflow
        self.tf = tf
        self.np = np
        if not TRAINED_MODEL_PATH.is_file():
            raise FileNotFoundError(
                "The trained ISL A-Z model is missing: "
                f"{TRAINED_MODEL_PATH}. Train it with train_sign_model.py "
                "using the ISL dataset, then restart the service."
            )
        if not TRAINED_LABELS_PATH.is_file():
            raise FileNotFoundError(
                f"Trained ISL model labels are missing: {TRAINED_LABELS_PATH}"
            )
        self.labels = json.loads(
            TRAINED_LABELS_PATH.read_text(encoding="utf-8")
        )
        if self.labels != list("abcdefghijklmnopqrstuvwxyz"):
            raise RuntimeError(
                "The configured sign model is not labeled with lowercase ISL "
                "letters a-z."
            )
        self.trained_model = tf.keras.models.load_model(
            TRAINED_MODEL_PATH,
            compile=False,
        )
        if self.trained_model.output_shape[-1] != len(self.labels):
            raise RuntimeError(
                "Trained model output count does not match its ISL labels file."
            )
        if not YOLO_HAND_MODEL_PATH.is_file():
            raise FileNotFoundError(
                "The YOLO11n hand detector is missing: "
                f"{YOLO_HAND_MODEL_PATH}. Place the trained hand detector there "
                "before starting the service."
            )
        self.hand_detector = YOLO(str(YOLO_HAND_MODEL_PATH))
        if self.hand_detector.task != "detect":
            raise RuntimeError("The configured YOLO hand model must be a detector.")
        self.lock = threading.Lock()
        print(
            f"Loaded YOLO11n hand detector from {YOLO_HAND_MODEL_PATH} and "
            f"ISL A-Z classifier from {TRAINED_MODEL_PATH}."
        )

    def predict(self, image_bytes):
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")

        image_array = self.np.asarray(image)
        width, height = image.size
        scale = min(width / 640, height / 480)
        side = round(400 * scale)
        left = (width - side) // 2
        top = (height - side) // 2
        if side < 1 or left + side > width or top + side > height:
            raise ValueError(
                f"Camera frame is too small for the sign input region: {width}x{height}."
            )

        with self.lock:
            detections = self.hand_detector.predict(
                source=self.np.ascontiguousarray(image_array[:, :, ::-1]),
                imgsz=640,
                conf=0.35,
                verbose=False,
            )[0]

        detected_hands = []
        if detections.boxes is not None:
            for box in detections.boxes:
                class_index = int(box.cls[0].item())
                if detections.names[class_index] != "hand":
                    continue
                coordinates = box.xyxyn[0].tolist()
                center_x = (coordinates[0] + coordinates[2]) * width / 2
                center_y = (coordinates[1] + coordinates[3]) * height / 2
                if not (
                    left <= center_x <= left + side
                    and top <= center_y <= top + side
                ):
                    continue
                detected_hands.append(
                    {
                        "box": tuple(float(value) for value in coordinates),
                        "confidence": float(box.conf[0].item()),
                    }
                )
        detected_hands.sort(
            key=lambda hand: hand["confidence"],
            reverse=True,
        )
        if not detected_hands:
            return {
                "label": "nothing",
                "confidence": 1.0,
                "alternatives": [],
                "hand_detected": False,
                "hands": [],
            }

        sign_image = image.crop((left, top, left + side, top + side))
        sign_image = sign_image.resize((200, 200), Image.Resampling.LANCZOS)
        image_buffer = io.BytesIO()
        sign_image.save(image_buffer, format="JPEG", quality=95)
        image_bytes = image_buffer.getvalue()

        image = self.tf.io.decode_jpeg(image_bytes, channels=3)
        image = self.tf.image.resize(image, (160, 160))
        with self.lock:
            scores = self.trained_model(
                self.tf.expand_dims(image, axis=0),
                training=False,
            ).numpy()[0]
        prediction = self._format_prediction(scores)
        prediction["hand_detected"] = True
        prediction["hands"] = detected_hands
        return prediction

    def _format_prediction(self, scores):
        if len(scores) != len(self.labels):
            raise RuntimeError(
                f"Model returned {len(scores)} scores for {len(self.labels)} labels."
            )
        best_indices = scores.argsort()[-3:][::-1]
        best_index = int(best_indices[0])
        return {
            "label": self.labels[best_index],
            "confidence": float(scores[best_index]),
            "hand_detected": True,
            "alternatives": [
                {
                    "label": self.labels[int(index)],
                    "confidence": float(scores[index]),
                }
                for index in best_indices
            ],
        }


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "chatbot.html")


@app.get("/api/status")
def status():
    return jsonify(
        ready=model is not None,
        error=None if model is not None else "Sign model has not loaded.",
    )


@app.post("/api/predict")
def predict():
    if request.mimetype != "image/jpeg":
        return jsonify(error="Send a JPEG camera frame."), 415
    image_bytes = request.get_data(cache=False)
    if not image_bytes:
        return jsonify(error="The camera frame is empty."), 400
    if model is None:
        return jsonify(error="Sign model is not loaded. Check the server output."), 503

    try:
        return jsonify(model.predict(image_bytes))
    except Exception:
        app.logger.exception("Sign frame prediction failed")
        return jsonify(error="Sign prediction failed; check the server log."), 500


@app.get("/api/samples")
def webcam_sample_counts():
    counts = {
        label: sum(
            1
            for path in (WEBCAM_DATASET_DIR / label).glob("*.jpg")
        )
        for label in "abcdefghijklmnopqrstuvwxyz"
    }
    return jsonify(counts=counts, total=sum(counts.values()))


@app.post("/api/collect")
def collect_webcam_sample():
    label = request.form.get("label", "")
    image_file = request.files.get("image")
    if len(label) != 1 or label not in "abcdefghijklmnopqrstuvwxyz":
        return jsonify(error="Choose a letter from a to z."), 400
    if image_file is None or image_file.mimetype != "image/jpeg":
        return jsonify(error="Send a JPEG webcam image."), 415

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify(error="The webcam image is empty."), 400

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.format != "JPEG" or min(source.size) < 160:
                return jsonify(error="The sample must be a JPEG at least 160x160."), 400
            source.verify()
    except (OSError, ValueError):
        return jsonify(error="The webcam sample is not a valid JPEG image."), 400

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            sample = source.convert("RGB")
            sample_directory = WEBCAM_DATASET_DIR / label
            sample_directory.mkdir(parents=True, exist_ok=True)
            sample_path = sample_directory / f"{uuid.uuid4().hex}.jpg"
            sample.save(sample_path, format="JPEG", quality=92)
    except OSError:
        app.logger.exception("Could not save webcam training sample.")
        return jsonify(error="Could not save the webcam sample; check the server log."), 500

    sample_count = sum(1 for _ in sample_directory.glob("*.jpg"))
    return jsonify(label=label, count=sample_count, total_bytes=sample_path.stat().st_size), 201


@app.post("/api/chat")
def chat():
    if not request.is_json:
        return jsonify(error="Send the chat messages as JSON."), 415

    body = request.get_json(silent=True)
    if body is None:
        return jsonify(error="The request body is not valid JSON."), 400

    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list) or not messages or len(messages) > 20:
        return jsonify(error="Send between 1 and 20 recent chat messages."), 400

    validated_messages = []
    for index, message in enumerate(messages):
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
            or len(message["content"]) > 4000
        ):
            return jsonify(error="A chat message is invalid or too long."), 400
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected_role:
            return jsonify(error="Chat roles must alternate, starting with the user."), 400
        validated_messages.append(
            {"role": message["role"], "content": message["content"]}
        )

    if validated_messages[-1]["role"] != "user":
        return jsonify(error="The latest chat message must be from the user."), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify(
            error=(
                "AI is not configured. Set the ANTHROPIC_API_KEY environment "
                "variable, then restart Chatbot."
            )
        ), 503

    payload = json.dumps(
        {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 1000,
            "system": CHAT_SYSTEM_PROMPT,
            "messages": validated_messages,
        }
    ).encode("utf-8")
    provider_request = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(provider_request, timeout=60) as response:
            provider_data = json.loads(response.read())
    except urllib.error.HTTPError as error:
        app.logger.error("Anthropic chat request failed with HTTP %s.", error.code)
        if error.code == 401:
            message = "Anthropic rejected the API key. Check ANTHROPIC_API_KEY."
        elif error.code == 429:
            message = "The AI service is busy or rate-limited. Try again shortly."
        else:
            message = f"The AI service returned HTTP {error.code}. Check the server log."
        return jsonify(error=message), 502
    except urllib.error.URLError as error:
        app.logger.error("Could not reach Anthropic: %s", error.reason)
        return jsonify(
            error="Could not reach the AI service. Check your internet connection."
        ), 502
    except (json.JSONDecodeError, UnicodeDecodeError):
        app.logger.exception("Anthropic returned an invalid chat response.")
        return jsonify(error="The AI service returned an invalid response."), 502

    if (
        not isinstance(provider_data, dict)
        or not isinstance(provider_data.get("content"), list)
    ):
        app.logger.error("Anthropic chat response has an unexpected structure.")
        return jsonify(error="The AI service returned an invalid response."), 502

    reply = "\n".join(
        block["text"]
        for block in provider_data.get("content", [])
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ).strip()
    if not reply:
        app.logger.error("Anthropic chat response did not contain text.")
        return jsonify(error="The AI service returned an empty response."), 502

    return jsonify(reply=reply)


def load_model():
    global model
    if model is None:
        try:
            model = SignModel()
        except Exception:
            app.logger.exception("Sign model could not load.")


load_model()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        threaded=True,
    )
