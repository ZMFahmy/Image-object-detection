from customtkinter import *
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

model = YOLO("my_YOLO_model.pt")
file_path = ""
image = None
image_label = None


def load_default_image():
    global image, image_label
    image = CTkImage(light_image=Image.open("Dark-grey.png"), dark_image=Image.open("Dark-grey.png"), size=(750, 400))
    image_label = CTkLabel(app, text="No image selected", image=image)
    image_label.place(relx=0.5, rely=0.4, anchor="center")


def load_image():
    global file_path, image, image_label
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    loaded_image = Image.open(file_path)
    if loaded_image.width < 750:
        load_default_image()
        image = CTkImage(light_image=loaded_image, dark_image=loaded_image, size=(loaded_image.width, 400))
    else:
        image = CTkImage(light_image=loaded_image, dark_image=loaded_image, size=(750, 400))
    image_label = CTkLabel(app, text="", image=image)
    image_label.place(relx=0.5, rely=0.4, anchor="center")


def make_inference():
    global file_path, image, image_label
    img = Image.open(file_path)

    # Convert PIL Image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Run the model on the image
    results = model(opencv_image)

    # Extract information from the results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_names = result.names  # Class names dictionary

        # Draw the bounding boxes and labels on the image
        for box, cls_id, confidence in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[int(cls_id)]
            label = f"{class_name} {confidence:.2f}"

            # Draw the rectangle
            cv2.rectangle(opencv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Put the label near the bounding box
            cv2.putText(opencv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Convert back to RGB for saving with Pillow
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(rgb_image)

    if pil_image.width < 750:
        load_default_image()
        image = CTkImage(light_image=pil_image, dark_image=pil_image, size=(pil_image.width, 400))
    else:
        image = CTkImage(light_image=pil_image, dark_image=pil_image, size=(750, 400))
    image_label = CTkLabel(app, text="", image=image)
    image_label.place(relx=0.5, rely=0.4, anchor="center")
    enumerate_categories(results)


def enumerate_categories(results):
    # Extract the detected class indices
    detected_class_indices = results[0].boxes.cls

    # Extract the class names using the model's class names
    detected_class_names = [model.names[int(idx)] for idx in detected_class_indices]

    category_map = {}
    for obj in detected_class_names:
        if obj in category_map:
            category_map[obj] += 1
        else:
            category_map[obj] = 1

    print("Objects present in image:")
    for key in category_map:
        print(key + " - count: " + str(category_map[key]))


app = CTk()
app.geometry("800x600")

set_appearance_mode("dark")

detect_btn = CTkButton(master=app, text="Detect objects", corner_radius=32, command=make_inference)
detect_btn.place(relx=0.3, rely=0.8, anchor="center")

upload_btn = CTkButton(master=app, text="Upload image", corner_radius=32, command=load_image)
upload_btn.place(relx=0.7, rely=0.8, anchor="center")

load_default_image()

app.mainloop()
