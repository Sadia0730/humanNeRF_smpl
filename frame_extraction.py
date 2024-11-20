
import cv2
import os

class FrameExtractor:
    def __init__(self, video_path, output_folder):
        self.video_path = video_path
        self.output_folder = output_folder

        # Open the video file
        self.cap = cv2.VideoCapture(self.video_path)

        # Create the output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Get the video name without extension
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]

        # Calculate the total number of frames in the video
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Counter for frame numbering
        self.frame_count = 0

        # Calculate interval to skip frames to extract only 600 frames
        self.frame_interval = max(1, self.total_frames // 600)

    def extract_frames(self):
        # Read and save frames
        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Check if the current frame is one of the frames to be saved
            if self.frame_count % self.frame_interval == 0:
                # Construct the frame filename
                frame_filename = f"{self.video_name}_frame_{self.frame_count:04d}.png"

                # Save the frame to the output folder
                frame_path = os.path.join(self.output_folder, frame_filename)
                cv2.imwrite(frame_path, frame)

                # If we have saved 600 frames, stop
                if self.frame_count // self.frame_interval >= 600:
                    break

            self.frame_count += 1

    def release_video_capture(self):
        # Release the video capture object
        self.cap.release()

if __name__ == "__main__":
    # Example usage
    video_path = r"/home/sadia/Downloads/CoreView_313/"
    output_folder = r"D:\Sadia\Sadia\PaperHub\AnimatedHumanExp\dataset\wild\monocular\images_600"

    # Create an instance of FrameExtractor
    frame_extractor = FrameExtractor(video_path, output_folder)

    # Call the extract_frames method
    frame_extractor.extract_frames()

    # Call the release_video_capture method to release resources
    frame_extractor.release_video_capture()

    print("Frames extracted and saved successfully.")