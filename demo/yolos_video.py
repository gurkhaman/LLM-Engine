from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(device)
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")


def process_frame(frame):
    # frame = cv2.resize(frame, (640, 360))  # downscaling the frame

    # Convert the frame (numpy array) to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Prepare the image for the model
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    # Perform object detection
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the outputs to extract bounding boxes and labels
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=0.9, target_sizes=target_sizes
    )[0]

    # Draw the bounding boxes on the original frame
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        cv2.rectangle(
            frame,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}",
            (int(box[0]), int(box[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return frame


def video_obj_det(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    processed_frames = []

    print("Video captured, starting frame processing...")

    frame_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally skip frames to speed up processing
        # if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 2 == 0:  # Process every 2nd frame
        print(f"Processing frame {frame_count}")
        processed_frame = process_frame(frame)
        processed_frames.append(processed_frame)
        frame_count += 1

    cap.release()

    if not processed_frames:
        print("Error: No frames processed.")
        return

    # Get frame size from the first processed frame
    height, width, _ = processed_frames[0].shape
    output_video = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height)
    )

    for frame in processed_frames:
        output_video.write(frame)

    output_video.release()
    print(f"Processed video saved to {output_video_path}")


if __name__ == "__main__":
    input_video = "/workspaces/composition-blueprint-engine/demo/SDI_DEMO_SOURCE.mp4"
    output_video = "/workspaces/composition-blueprint-engine/demo/obj_det_video.mp4"
    print("Starting video processing...")
    video_obj_det(input_video, output_video)
