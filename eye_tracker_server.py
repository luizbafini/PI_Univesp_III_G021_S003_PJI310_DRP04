import asyncio
import websockets
import cv2
import mediapipe as mp
import time

# Inicializar a câmera e o MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Controle de fixação de olhar
gaze_start_time = None
fixation_threshold = 1.5  # Segundos olhando fixamente para considerar seleção

# Função para enviar comando ao navegador
async def track_and_send(websocket):
    global gaze_start_time

    last_position = None

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape

            # Coordenadas dos olhos (íris esquerda e direita)
            left_iris = [face_landmarks.landmark[474], face_landmarks.landmark[475], face_landmarks.landmark[476], face_landmarks.landmark[477]]
            right_iris = [face_landmarks.landmark[469], face_landmarks.landmark[470], face_landmarks.landmark[471], face_landmarks.landmark[472]]

            left_center_x = int(sum(p.x for p in left_iris) / 4 * w)
            left_center_y = int(sum(p.y for p in left_iris) / 4 * h)

            right_center_x = int(sum(p.x for p in right_iris) / 4 * w)
            right_center_y = int(sum(p.y for p in right_iris) / 4 * h)

            avg_x = (left_center_x + right_center_x) // 2
            avg_y = (left_center_y + right_center_y) // 2

            current_position = (avg_x, avg_y)
            await websocket.send(f"{avg_x},{avg_y}")

            # Checa fixação de olhar
            if last_position and abs(current_position[0] - last_position[0]) < 20 and abs(current_position[1] - last_position[1]) < 20:
                if gaze_start_time is None:
                    gaze_start_time = time.time()
                elif time.time() - gaze_start_time >= fixation_threshold:
                    await websocket.send("select")
                    print("Seleção confirmada por fixação de olhar!")
                    gaze_start_time = None
            else:
                gaze_start_time = None

            last_position = current_position

        cv2.imshow("Eye Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        await asyncio.sleep(0.05)  # Pequena pausa para suavizar o envio

async def main():
    async with websockets.serve(track_and_send, "localhost", 8765):
        print("Servidor WebSocket iniciado em ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
    cap.release()
    cv2.destroyAllWindows()
