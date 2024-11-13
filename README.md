# license-verification-pc
this is the pc version of license verification where the user should scan the qr which is present behind their license and then facial recognition is used to compare the image in the qr data with the person in front of camera

    import cv2
    from pyzbar.pyzbar import decode
    import requests
    from bs4 import BeautifulSoup
    from PIL import Image
    import base64
    from io import BytesIO
    import face_recognition
    import time

    def qr_capture():
    capture = cv2.VideoCapture(0)
    url_opened = False
    qr_data = None
    while True:
        success, frame = capture.read()
        if not success:
            brea
        for qr_code in decode(frame):
            qr_data = qr_code.data.decode('utf-8')
            if not url_opened:
                url_opened = True
            break
        cv2.imshow("QR CODE SCANNER", frame)
        if cv2.waitKey(1) & url_opened:
            break
    capture.release()
    cv2.destroyAllWindows()
    return qr_data

    def img_download(url, compare):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_links = [img['src'] for img in soup.find_all('img') if 'src' in img.attrs]
    text_content = soup.get_text()
    if compare.lower() in text_content.lower():
        for link in img_links:
            if 'base64' in link:
                image_data = link.split(",")[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                image_path = "decoded_image.jpg"
                image.save(image_path)
                return image_path
    return None

    def capture_zoomed_face():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    taking_picture = False
    start_time = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if not taking_picture:
                    taking_picture = True
                    start_time = time.time()
        if taking_picture:
            elapsed_time = time.time() - start_time
            if elapsed_time >= 2:
                for (x, y, w, h) in faces:
                    face_crop = frame[y:y+h, x:x+w]
                    zoomed_face = cv2.resize(face_crop, (300, 300))
                    zoomed_filename = 'zoomed_face.jpg'
                    cv2.imwrite(zoomed_filename, zoomed_face)
                break
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    def find_face_encodings(image_path):
    image = cv2.imread(image_path)
    face_enc = face_recognition.face_encodings(image)
    return face_enc[0] if face_enc else None

    def face_comparison(image_path):
    image_1 = find_face_encodings(image_path)
    image_2 = find_face_encodings('zoomed_face.jpg')
    if image_1 is None or image_2 is None:
        print("Could not find a face in one of the images.")
        return False
    is_same = face_recognition.compare_faces([image_1], image_2)[0]
    if is_same:
        distance = face_recognition.face_distance([image_1], image_2)[0]
        accuracy = 100 - round(distance * 100)
        print("The images are the same")
        print(f"Accuracy Level: {accuracy}%")
    else:
        print("The images are not the same")
    return is_same

    # Main execution
    qr_data = qr_capture()
    compare_text = "lmv"

    if qr_data:
    image_path = img_download(qr_data, compare_text)
    if image_path:
        capture_zoomed_face()
        match = face_comparison(image_path)
        if match:
            print("You can drive the car")
        else:
            print("Not eligible")
    else:
        print("Image not found or text did not match.")
    else:
    print("QR code could not be scanned.")
