# --- imports ---
import argparse
import serial
from serial.tools import list_ports   
import sys
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import threading

# --- variables ---
playerLies = 0
rounds = 5

BAUD_RATE = 115200
MODEL_PATH ='gesture_recognizer.task'

BaseOptions = mp_python.BaseOptions
GestureRecognizerOptions = mp_vision.GestureRecognizerOptions
GestureRecognizer = mp_vision.GestureRecognizer

base_options = BaseOptions(model_asset_path=MODEL_PATH)
options = GestureRecognizerOptions(base_options=base_options)
recognizer = GestureRecognizer.create_from_options(options)

MPImage = mp.Image

# --- main function ---
# sets up and runs the game
def main(port):
    print("Main")
    global rounds
    current_round = rounds

    # --- RPS Mapping ---
    GESTURE_TO_MOVE = {
    "Closed_Fist": "rock",
    "Open_Palm": "paper",
    "Victory": "scissors",
    }
    MOVES = ["rock", "paper", "scissors"]

    # --- web camera & gesture smoothing ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    pred_buffer = []      # last N predictions for smoothing
    BUFFER_SIZE = 15
    current_move = None   # last smoothed move
    computer_text = ""

    
    ser = serial.Serial(port, 115200, timeout = 1.0)
    while current_round > 0:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Convert BGR -> RGB for MediaPipe
        rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = recognizer.recognize(mp_image)

        frame = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
        
        label_text = "No gesture"
        move_text = "Move: ?"

        if result.gestures:
            top = result.gestures[0][0]   # best candidate
            label = top.category_name
            score = top.score

            # Keep short history of labels
            pred_buffer.append(label)
            if len(pred_buffer) > BUFFER_SIZE:
                pred_buffer.pop(0)

            # Most frequent label in buffer
            labels, counts = np.unique(pred_buffer, return_counts=True)
            best_label = labels[np.argmax(counts)]

            current_move = GESTURE_TO_MOVE.get(best_label, None)

            label_text = f"Gesture: {best_label} ({score:.2f})"
            if current_move is not None:
                move_text = f"Move: {current_move}"
            else:
                move_text = "Move: (unmapped gesture)"
        else:
            # If no gesture, slowly clear buffer
            if pred_buffer:
                pred_buffer.pop(0)
            if not pred_buffer:
                current_move = None

        # --- Draw UI ---
        cv2.putText(frame, label_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, move_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, "START: play  |  Q: quit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
      
        
        
        cv2.imshow("RPS with gesture_recognizer.task", frame)

        key = cv2.waitKey(30) & 0xFF

        # --- quit on Q and show current result---
        if key == ord('q'):
            quit(cap)

        # listen for arduino input to start a match
        if (checkForStartButton(ser, frame, computer_text)== True):
            print("Start pressed")
         
          
            if current_move in MOVES:
                player_move = current_move
                computer_move = np.random.choice(MOVES)
                winner = decideWinner(player_move, computer_move)
                
                computer_text = f"Computer: {computer_move}"
                playerSelfReport(winner, ser, computer_text, frame)



                print("\n--- ROUND ---")
                print("Player:   ", player_move)
                print("Computer: ", computer_move)
                print("Winner:   ", winner)

                current_round -= 1

            else:
                print("\nNo valid move detected. Show a clear gesture and try again.")
     
    gameEnd(cap)
    

# --- sub functions ---

# calls setup functions and then main function
def gameSetup():
    print("game setting up...")
    port = portSetup()
    main(port)

# detect which port the arduino is using
def choosePort(provided_port=None):
    if provided_port:
        return provided_port
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found. Connect the ESP32 and try again.")
        sys.exit(1)
    if len(ports) == 1:
        return ports[0].device
    # prefer common USB-to-UART descriptors if multiple ports exist
    for p in ports:
        desc = (p.description or "").lower()
        if any(x in desc for x in ("silicon", "cp210", "ch340", "ftdi", "usb-serial")):
            return p.device
    # fallback: return first
    return ports[0].device

# set up serial port communication
def portSetup():
    parser = argparse.ArgumentParser(description="Send 'Hello ESP32!' to an ESP32 over serial.")
    parser.add_argument("-p", "--port", help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    parser.add_argument("-b", "--baud", type=int, default=BAUD_RATE, help="Baud rate (default: 115200)")
    parser.add_argument("-t", "--timeout", type=float, default=1.0, help="Read timeout in seconds")
    
    args = parser.parse_args()
    port = choosePort(args.port)

    return port

# determnines the outcome of a match
def decideWinner(player, computer):
    if player == computer:
        print("draw")
        return "draw"
    if (player == "rock" and computer == "scissors") or \
       (player == "paper" and computer == "rock") or \
       (player == "scissors" and computer == "paper"):
        print("win")
        return "player"
    print("loss")
    return "computer"

# check if the player has pressed the start button
def checkForStartButton(ser, frame, computer_text):
    start = False
    if ser.in_waiting > 0:
        data = ser.readline().decode('utf-8').strip()
        if data == "START":
            start = True
                      

    return start


# check what the player self-reported and determine if its a lie
def playerSelfReport(result, ser, computer_text, frame):
    reported = False
    while reported == False:
        display = frame.copy()
        cv2.putText(display, computer_text, (145, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(display,
                    "Input if you won, lost or the match ended in a draw",
                    (25, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("RPS with gesture_recognizer.task", display)
        cv2.waitKey(1)
        
        if ser.in_waiting > 0:
            global playerLies
            data = ser.readline().decode('utf-8').strip()
            if data == "WIN":
                print("You have reported a victory")
                if result != "player":
                    playerLies += 1
                reported = True
            elif data == "LOSS":
                print("You have reported a loss")
                if result != "computer":
                    playerLies += 1
                reported = True
            elif data == "DRAW":
                print("You have reported a draw")
                if result != "draw":
                    playerLies += 1
                reported = True


# game end after set number of rounds
def gameEnd(cap):
    print("GameEnd")
    playerReliability()
    quit(cap)

# determine & display which percentage of the player reports where false
def playerReliability():
    trust = 100 - (playerLies*100/rounds)
    round(trust, 1)
    lieDetection = "You told the truth " + str(trust) + "% of the time"
    print(lieDetection)

# close the game
def quit(cap):
    print("Game shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

# --- start game ---
gameSetup()
