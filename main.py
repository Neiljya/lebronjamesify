import cv2
from PIL import Image

img_path = "image/sample.jpg"
overlay_path = "overlay/jamesify.jpg"

def detect_faces(img_path):
    # Load the input image and convert it from BGR to RGB
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb

def overlay_img(img_path, overlay_path):
    overlay_image = Image.open(overlay_path)

    # Using OpenCV's pre-trained model for face detection
    face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

    # Detect faces
    image_rgb = detect_faces(img_path)

    faces = face_cascade.detectMultiScale(image_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

    pil_image = Image.fromarray(image_rgb)

    for (x, y, w, h) in faces:
        # Resize the overlay image to fit the face
        resized_overlay_image = overlay_image.resize((w*3,h*3), Image.Resampling.LANCZOS)

        # Overlay
        pil_image.paste(resized_overlay_image,((x-w)+5, (y-h)+5), resized_overlay_image)

    # Saving the result
    pil_image.save('result.png')
    pil_image.show()


overlay_img(img_path, overlay_path)

