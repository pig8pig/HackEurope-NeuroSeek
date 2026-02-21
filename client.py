import cv2
import requests

# Your live Cloudflare URL pointing to the A10G
API_URL = "https://modified-reaching-edwards-sheer.trycloudflare.com/analyze-pose"

# Boot up your laptop's webcam
cap = cv2.VideoCapture(0)

print("Connecting to Red Hat OpenShift Cluster...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Compress the frame so it survives the hackathon Wi-Fi
    small_frame = cv2.resize(frame, (640, 480))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
    _, img_encoded = cv2.imencode('.jpg', small_frame, encode_param)
    
    try:
        # 2. Fire the frame through the tunnel to your GPU
        response = requests.post(API_URL, files={"file": img_encoded.tobytes()})
        data = response.json()
        
        # 3. Read the GPU's mind
        if data['status'] == 'success':
            nose_x, nose_y = data['keypoints'][0]
            print(f"Nose detected at -> X: {nose_x:.2f} | Y: {nose_y:.2f}")
        else:
            print("No person detected.")
            
    except Exception as e:
        print(f"Network lag or server unavailable... {e}")

    # Show the live feed on your screen
    cv2.imshow('NeuroSeek Client Feed', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()