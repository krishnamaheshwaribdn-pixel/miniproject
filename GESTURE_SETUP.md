# Chatbot ISL gesture typing

The gesture panel uses a YOLO11n hand detector (`%LOCALAPPDATA%\ChatbotGesture\yolo11n_hand_best.pt`) to locate hands, followed by a MobileNetV2 classifier trained on the [Indian Sign Language image dataset](https://github.com/ayesha-hannure/Indian-Sign-Language-dataset). The ISL dataset has 26 letter classes but no hand-box annotations, so it trains the classifier, not the YOLO detector. The YOLO hand detector is a separate model. The trained letter classifier and captured webcam samples are stored under `%LOCALAPPDATA%\ChatbotGesture\`.

The classifier uses a centered 400-by-400 camera region (scaled proportionally from a 640-by-480 frame), resized to 160-by-160. The green square marks the classifier input; yellow boxes show YOLO hand detections. Keep the whole sign inside the green box. YOLO gates classification when it cannot find a hand. Lighting, hand orientation, camera quality, sign framing, and signer differences still affect accuracy; perfect recognition cannot be guaranteed.

## Enable the AI chat

Chatbot sends conversation requests through the local Flask server to Anthropic's Claude API. The API key is read from the server's `ANTHROPIC_API_KEY` environment variable and is never sent to the browser or stored in this project. Create an API key in the Anthropic Console, then set it in PowerShell:

```powershell
setx ANTHROPIC_API_KEY "your-anthropic-api-key"
```

Do not commit or share the key. Close and restart Chatbot after setting it; Windows startup shortcuts only receive updated user environment variables after signing out and back in. API usage may incur charges according to your Anthropic account.

## Automatic Windows startup

The `start_chatbot.ps1` launcher starts `sign_server.py` when Windows signs in, waits for the model service, then opens Chatbot. On first browser use, allow camera access; later launches will start the camera automatically when browser permission remains granted. To stop automatic startup, remove the **Chatbot Gesture Control** shortcut from your Windows Startup folder.

To install the startup shortcut, run this once in PowerShell from this folder:

```powershell
$startup = [Environment]::GetFolderPath('Startup')
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path $startup 'Chatbot Gesture Control.lnk'))
$shortcut.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Minimized -File `"$((Resolve-Path '.\start_chatbot.ps1').Path)`""
$shortcut.WorkingDirectory = (Get-Location).Path
$shortcut.Save()
```

## Manual start

Use Python 3.13, then run these commands from this folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python sign_server.py
```

Open `http://127.0.0.1:5000` and allow camera access. Keep your whole hand inside the green input box, with fingers unobstructed, palm facing the camera, and a plain background. A sign is typed only when the prediction is confident and consistent across several frames; uncertain predictions are ignored. Each different sign types once; return to neutral before repeating the same sign. The ISL model recognizes letters a-z; use the keyboard for spaces and deletion.

The model supports 26 still-image letter classes, not full sign-language translation; letters requiring motion may not be reliable. Camera access works on localhost; the browser must be able to reach the local Flask service.

## Retrain for your camera

No classifier can guarantee perfect recognition for every camera, background, lighting condition, and signer. To adapt the MobileNetV2 classifier to your webcam:

1. Start Chatbot and click **START CAMERA**.
2. Select a letter, make that ISL hand sign, and click **CAPTURE**. Capture at least 30 varied samples per letter, changing hand position or lighting slightly. Keep the hand inside the green box. Samples are saved locally in `%LOCALAPPDATA%\ChatbotGesture\webcam-dataset\`; they are not uploaded.
3. After capturing all 26 letters, train using both the downloaded ISL images and your webcam samples:

```powershell
$dataset = Join-Path $env:LOCALAPPDATA 'ChatbotGesture\isl-dataset'
$webcam = Join-Path $env:LOCALAPPDATA 'ChatbotGesture\webcam-dataset'
.\.venv\Scripts\python.exe .\train_sign_model.py --dataset-dir $dataset --webcam-dir $webcam --head-epochs 8 --fine-tune-epochs 3
```

The trainer mixes webcam and ISL images, reserves 20% of webcam samples for a separate webcam validation score, and saves both metrics and the model under `%LOCALAPPDATA%\ChatbotGesture\`. At least five webcam images per letter are required; 30 or more varied images per letter are recommended. CPU-only training can take a long time. Restart `sign_server.py` after training to load the updated classifier. The YOLO detector is not retrained by these images because they have no bounding-box labels.
