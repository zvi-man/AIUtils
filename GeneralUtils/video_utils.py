import cv2
import pandas as pd
import random


class VideoUtils:
    @classmethod
    def add_bounding_boxes_to_video(cls, video_input_path: str,
                                    video_output_path: str,
                                    csv_path: str,
                                    frame_col: str = "frame",
                                    width_col: str = "w",
                                    length_col: str = "h",
                                    top_l_x_col: str = "x",
                                    top_l_y_col: str = "y",
                                    text_col: str = "text") -> None:

        # Load the video file
        video = cv2.VideoCapture(video_input_path)

        # Read the DataFrame with rectangle and frame location information
        df = pd.read_csv(csv_path)

        # Get the total number of frames in the video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get the frame width and height
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get the frame rate of the video
        fps = video.get(cv2.CAP_PROP_FPS)

        # Define the codec and create a video writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

        # Set the font and font scale of the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5

        # Loop through the frames of the video
        for i in range(total_frames):
            # Read the next frame
            ret, frame = video.read()

            # Check if the DataFrame contains information for the current frame
            if i in df[frame_col].values:
                # Get the rectangle information for the current frame
                row = df[df[frame_col] == i]
                x, y, w, h = int(row[top_l_x_col]), \
                             int(row[top_l_y_col]), \
                             int(row[width_col]), \
                             int(row[length_col])
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                # Draw the rectangle on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)

                text = row[text_col]
                textsize = cv2.getTextSize(text, font, scale, 1)[0]
                # Draw the text on the frame
                cv2.putText(frame, text, (x + int(w / 2) - int(textsize[0] / 2), y - 10),
                            font, scale, color, 1, cv2.LINE_AA)

            # Write the modified frame to the output video
            out.write(frame)

        # Release the video and writer objects
        video.release()
        out.release()
