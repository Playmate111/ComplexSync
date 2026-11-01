import os
import cv2
import ffmpeg
import torch
import numpy as np
from tqdm import tqdm


def get_file_list(data_dir, suffix="", max_num=None):
    file_list = []

    for dirpath, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(suffix):
                file_list.append(os.path.join(dirpath, filename))
                if max_num is not None and len(file_list) >= max_num:
                    break
        if max_num is not None and len(file_list) >= max_num:
            break

    try:
        file_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    except Exception as e:
        file_list = sorted(file_list)

    return file_list


def load_video(video_path):
    all_frames = []
    video_cap = cv2.VideoCapture(video_path)

    while True:
        _, f = video_cap.read()
        if f is None:
            break
        all_frames.append(f[:, :, ::-1])

    return all_frames


def crop_resize_video_frames(
    frames,
    height,
    width,
    insightface,
    y_up_ratio=0.5,
    y_down_ratio=1.5,
):
    org_h, org_w = frames[0].shape[0], frames[0].shape[1]

    crop_rects = []
    crop_frames = []
    pre_rect = None
    for frame in tqdm(frames):
        det_res = insightface.get(frame)
        if len(det_res) < 1:
            if pre_rect is not None:
                rect = pre_rect
            else:
                return None
        else:
            face_dict = det_res[0]
            rect = face_dict['bbox']

        x_min, y_min, x_max, y_max = rect[:4]
        face_w = x_max - x_min
        face_h = y_max - y_min

        crop_y_min = max(0, int(y_min - face_h * y_up_ratio))
        crop_y_max = min(org_h, int(y_max + face_h * y_down_ratio))
        crop_h = (crop_y_max - crop_y_min)
        crop_w = int(crop_h / height * width)
        center_x = (x_max + x_min) / 2
        crop_x_min = max(0, int(center_x - crop_w / 2))
        crop_x_max = min(org_w, int(center_x + crop_w / 2))
        crop_frames.append(
            cv2.resize(
                frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max],
                (width, height),
                interpolation=cv2.INTER_CUBIC
            )
        )
        crop_rects.append([crop_x_min, crop_y_min, crop_x_max, crop_y_max])

    return crop_frames, crop_rects


def image_to_tensor(image):
    image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1)

    return image


class VideoWriter:
    def __init__(self, output_path, height, width, fps, audio_stream, loglevel='fatal'):
        input_args = {
            'format': 'rawvideo',
            'pix_fmt': 'rgb24',
            's': '{}x{}'.format(width, height),
            'framerate': fps,
        }
        output_args = {
            'pix_fmt': 'yuv420p',
            'vcodec': 'libx264',
            'crf': 18,
            'preset': 'slower',
            'profile:v': 'high',
            'color_range': 'tv',
            'colorspace': 'bt709',
            'color_primaries': 'bt709',
            'color_trc': 'bt709',
            'shortest': None
        }
        if height % 2 != 0 or width % 2 != 0:
            output_args['vf'] = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
        self.process = None
        if audio_stream is None:
            self.process = (
                ffmpeg
                .input('pipe:', **input_args)
                .output(output_path, **output_args)
                .global_args('-loglevel', loglevel)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
        else:
            if isinstance(audio_stream, str):
                assert os.path.exists(audio_stream)
                audio_stream = ffmpeg.input(audio_stream)

            self.process = (
                ffmpeg
                .input('pipe:', **input_args)
                .output(audio_stream, output_path, **output_args)
                .global_args('-loglevel', loglevel)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

    @classmethod
    def from_video(cls, video_reader, output_path):
        if video_reader.has_audio:
            return cls(
                output_path,
                video_reader.height,
                video_reader.width,
                video_reader.fps,
                video_reader.streams.audio
            )
        else:
            return cls(
                output_path,
                video_reader.height,
                video_reader.width,
                video_reader.fps,
                None
            )

    def write(self, frame):
        self.process.stdin.write(frame.astype(np.uint8).tobytes())

    def close(self, timeout=None):
        self.process.stdin.close()
        self.process.wait(timeout=timeout)
        if self.process.poll() is None:
            self.process.kill()

    def __del__(self):
        self.close()


def create_masks_from_landmarks_box(landmark_list, img_shape, nose_index=28, dtype="uint8", box_expand=0.0):
    height, width = img_shape[:2]
    num_frames = landmark_list.shape[0]

    # Initialize the masks array
    masks = np.zeros((num_frames, height, width), dtype=dtype)

    if 0 <= box_expand < 1:
        box_expand = int(box_expand * width)

    for i in range(num_frames):
        # Get the landmarks for the current frame
        landmarks = landmark_list[i]

        # Get the y-coordinate of the nose landmark
        nose_point_h = landmarks[nose_index, 1]
        cut_h = nose_point_h

        # Find the leftmost and rightmost landmarks
        far_left_index = np.argmin(landmarks[:, 0])
        far_right_index = np.argmax(landmarks[:, 0])

        # Define the points for the mask contour
        left_up_point = np.array([landmarks[far_left_index][0], cut_h - box_expand], dtype=np.int32)
        left_down_point = np.array([landmarks[far_left_index][0], height - 60], dtype=np.int32)
        right_up_point = np.array([landmarks[far_right_index][0], cut_h - box_expand], dtype=np.int32)
        right_down_point = np.array([landmarks[far_right_index][0], height - 60], dtype=np.int32)

        # Define the contour
        contour = np.array([[left_up_point, left_down_point, right_down_point, right_up_point]])

        # Draw the contour on the mask
        cv2.drawContours(masks[i], [contour], -1, color=(1), thickness=cv2.FILLED)

    return torch.from_numpy(masks)
