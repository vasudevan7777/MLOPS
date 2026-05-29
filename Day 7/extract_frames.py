import cv2
import os

# Video folders
video_folders = {
    "low": r"D:\SMART COACH\video\low",
    "medium": r"D:\SMART COACH\video\medium",
    "high": r"D:\SMART COACH\video\high"
}

# Output folders
output_folders = {
    "low": r"D:\SMART COACH\video\extracted_frames",
    "medium": r"D:\SMART COACH\video\extracted_frames\medium",
    "high": r"D:\SMART COACH\video\extracted_frames\high"
}

# Loop through each category
for category in video_folders:

    video_path = video_folders[category]
    output_path = output_folders[category]

    os.makedirs(output_path, exist_ok=True)

    # Read all videos
    for video_file in os.listdir(video_path):

        full_video_path = os.path.join(
            video_path,
            video_file
        )

        cap = cv2.VideoCapture(full_video_path)

        frame_count = 0
        saved_count = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            # Save every 10th frame
            if frame_count % 10 == 0:

                frame_name = (
                    f"{category}_{saved_count}.jpg"
                )

                frame_path = os.path.join(
                    output_path,
                    frame_name
                )

                cv2.imwrite(frame_path, frame)

                saved_count += 1

            frame_count += 1

        cap.release()

print("Frame Extraction Completed")