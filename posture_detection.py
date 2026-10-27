import cv2
import mediapipe as mp
import serial
import time

# ESP32 serial connection
PORT = "COM6"
BAUD_RATE = 115200

try:
    esp32 = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("ESP32 connected successfully!")
except serial.SerialException as error:
    print("ESP32 connection failed:", error)
    raise SystemExit

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Webcam could not be opened.")
    esp32.close()
    raise SystemExit

last_command = None
last_warning_time = 0
warning_hold_seconds = 2.0

try:
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            status = "BODY NOT DETECTED"
            warning = False

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                left = landmarks[
                    mp_pose.PoseLandmark.LEFT_SHOULDER
                ]
                right = landmarks[
                    mp_pose.PoseLandmark.RIGHT_SHOULDER
                ]

                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                if min(
                    nose.visibility,
                    left.visibility,
                    right.visibility
                ) > 0.5:

                    shoulder_x = (left.x + right.x) / 2
                    shoulder_y = (left.y + right.y) / 2
                    vertical_gap = shoulder_y - nose.y

                    if vertical_gap > 0.01:
                        ratio = abs(nose.x - shoulder_x) / vertical_gap

                        # Preliminary heuristic, not validated posture detection
                        warning = ratio > 0.25

                        if warning:
                            last_warning_time = time.time()
                            status = "POSTURE CHANGE - CHECK POSITION"
                        else:
                            status = "POSITION CHECK - NO WARNING"

                        cv2.putText(
                            frame,
                            f"Head/shoulder ratio: {ratio:.2f}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )

            # Hold a warning briefly to avoid rapid switching
            if time.time() - last_warning_time < warning_hold_seconds:
                command = "1"
            else:
                command = "0"

            if command != last_command:
                esp32.write(command.encode())
                last_command = command

            color = (0, 0, 255) if command == "1" else (0, 200, 0)

            cv2.putText(
                frame,
                status,
                (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            cv2.imshow("Posture Monitoring System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    # Turn off LED and buzzer when closing the program
    try:
        esp32.write(b"0")
    except serial.SerialException:
        pass

    camera.release()
    cv2.destroyAllWindows()
    esp32.close()
    print("Webcam and ESP32 connection closed.")

