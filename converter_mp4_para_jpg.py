import cv2
import os


video_path = 'manga_rosa.mp4'
saida = 'DATASET/ROSA'  


os.makedirs(saida, exist_ok=True)


cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error: Video não aberto.")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Vídeo processado.")
        break

    frame_filename = os.path.join(saida, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    
    print(f"Saved: {frame_filename}")
    frame_count += 1


cap.release()
print(f"Total de frames: {frame_count}")