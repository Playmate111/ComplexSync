# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Face detection / alignment helpers used at inference time."""

import cv2
import numpy as np
from skimage import transform as trans

from deep_sort.detection import Detection

REF_112 = np.array([[38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.7299, 92.2041]], dtype=np.float32)
REF_256 = REF_112 * 2 + 16


def face_detection(frame, insightface_app):
    h, w = frame.shape[:2]

    det_res = insightface_app.get(frame)
    if len(det_res) == 0:
        return None

    face_dict = det_res[0]
    rect = face_dict['bbox']
    lms_3d_68 = face_dict['landmark_3d_68']

    x_min, y_min, x_max, y_max = rect[:4]
    face_w = x_max - x_min
    face_h = y_max - y_min

    crop_x_min = max(0, int(x_min - face_w * 1))
    crop_x_max = min(w, int(x_max + face_w * 1))
    crop_y_min = max(0, int(y_min - face_h * 1))
    crop_y_max = min(h, int(y_max + face_h * 1))

    lms_68 = lms_3d_68[:, :2]
    lms_68[:, 0] -= crop_x_min
    lms_68[:, 1] -= crop_y_min

    face = frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max, :]
    crop_rect = [crop_x_min, crop_y_min, crop_x_max, crop_y_max]

    return face, lms_68, crop_rect


def align_and_crop_face(org_img, lm68, align_M, src_size, tgt_size):
    M = align_M.copy()

    align_kpt = M[:2, :] @ np.concatenate([lm68.transpose(1, 0), np.ones((1, 68))], axis=0)
    align_kpt = align_kpt.transpose(1, 0).astype(int)

    center_point = align_kpt[27, :]
    center_x, center_y = center_point[0], center_point[1]
    max_x, max_y, min_x, min_y = \
        align_kpt[:, 0].max(), align_kpt[:, 1].max(), align_kpt[:, 0].min(), align_kpt[:, 1].min()

    width = max_x - min_x

    min_x = int(max(center_x - 0.9 * width, 0))
    max_x = int(min(center_x + 0.9 * width, src_size[1]))
    min_y = int(max(center_y - 0.7 * width, 0))
    max_y = int(min(center_y + 1.1 * width, src_size[0]))

    crop_index = [min_x, min_y, max_x, max_y]

    M[0, 2] -= crop_index[0]
    M[1, 2] -= crop_index[1]
    resize_M = np.array(
        [
            [float(tgt_size[0]) / (crop_index[2] - crop_index[0]), 0, 0],
            [0, float(tgt_size[1]) / (crop_index[3] - crop_index[1]), 0],
            [0, 0, 1]
        ]
    )
    M = resize_M @ M
    res_img = cv2.warpAffine(org_img, M[:2, :], tgt_size, borderValue=0.0)

    align_kpt = M[:2, :] @ np.concatenate([lm68.transpose(1, 0), np.ones((1, 68))], axis=0)
    align_kpt = align_kpt.transpose(1, 0).astype(int)

    return res_img, align_kpt, M


def align_with_landmarks(image, landmarks):
    image_size = np.array(image.shape[:2][::-1])
    ref = REF_256 / 256 * image_size
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, ref)

    M = tform.params
    M[0:2, 2] += image_size * 0.5

    return M


def crop_and_align_face(frames, insightface, size=512):
    align_faces = []
    align_lm68s = []
    crop_rects = []
    Ms = []
    face_flags = []
    for idx, frame in enumerate(frames):
        frame = frame[..., ::-1]
        det_res = face_detection(frame, insightface)
        if det_res is None:
            if idx == 0:
                return None
            else:
                face_flags.append(0)
                align_faces.append(align_faces[-1])
                align_lm68s.append(align_lm68s[-1])
                crop_rects.append(crop_rects[-1])
                Ms.append(Ms[-1])
        else:
            face_flags.append(1)
            crop_face, lms_68, crop_rect = det_res
            face_h, face_w = crop_face.shape[:2]
            align_M = align_with_landmarks(crop_face, lms_68[[38, 43, 30, 48, 54]].copy())
            align_face, align_lm68, M = align_and_crop_face(
                crop_face, lms_68, align_M, (face_h * 2, face_w * 2), (size, size)
            )
            align_face = align_face[..., ::-1]

            align_faces.append(align_face)
            align_lm68s.append(align_lm68)
            crop_rects.append(crop_rect)
            Ms.append(M[:2])

    align_lm68s = np.stack(align_lm68s)

    return align_faces, align_lm68s, crop_rects, Ms, face_flags


def detect_and_track_face(frames, insightface, tracker, size=512, first_crop=False):
    crop_faces = []
    crop_lm68s = []
    crop_rects = []
    Ms = []

    pad_num = 4
    frames = frames[:1] * pad_num + frames

    for idx, frame in enumerate(frames):
        frame = frame[..., ::-1]
        org_h, org_w = frame.shape[:2]

        if (first_crop and idx <= pad_num) or (not first_crop):
            det_res = insightface.get(frame)
            if len(det_res) == 0:
                return None

            detections = []
            face = det_res[0]
            bbox = face.bbox.astype(np.int32)
            lm68 = face.landmark_3d_68
            conf = face.det_score
            feature = face.normed_embedding

            min_x, min_y, max_x, max_y = bbox
            w, h = max_x - min_x, max_x - min_x
            bbox_deepsort = [min_x, min_y, w, h]

            detections.append(Detection(bbox_deepsort, conf, feature))
            tracker.predict()
            tracker.update(detections)

            if idx < pad_num:
                continue

            track = tracker.tracks[0]
            if not track.is_confirmed() or track.time_since_update > 1:
                crop_rect = [0, 0, 10, 10]
            else:
                crop_rect = track.to_tlbr().astype(int)

            min_x, min_y, max_x, max_y = crop_rect

            width = max_x - min_x
            center_x = (max_x + min_x) / 2
            center_y = (max_y + min_y) / 2
            y_up_ratio, y_down_ratio = 0.3, 1.0
            min_y = int(min_y - width * y_up_ratio)
            max_y = int(max_y + width * y_down_ratio)
            crop_w = crop_h = (max_y - min_y)
            min_x = int(center_x - crop_h / 2)
            max_x = int(center_x + crop_h / 2)

            crop_rect = [max(0, min_x), max(0, min_y), min(org_w, max_x), min(org_h, max_y)]

            ref_pts = np.array([[0, 0], [crop_w, 0], [0, crop_h], [crop_w, crop_h]], dtype=np.float32)
            src_pts = np.array([[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]], dtype=np.float32)
            tform = trans.SimilarityTransform()
            tform.estimate(src_pts, ref_pts)
            M = tform.params

            resize_M = np.array(
                [
                    [float(size) / crop_w, 0, 0],
                    [0, float(size) / crop_h, 0],
                    [0, 0, 1]
                ]
            )
            M = resize_M @ M

        crop_face = cv2.warpAffine(frame, M[:2, :], (size, size), borderValue=0.0)
        crop_face = crop_face[..., ::-1].astype(np.uint8)

        crop_lm68 = M[:2, :] @ np.concatenate([lm68[:, :2].transpose(1, 0), np.ones((1, 68))], axis=0)
        crop_lm68 = crop_lm68.transpose(1, 0).astype(int)

        crop_faces.append(crop_face)
        crop_lm68s.append(crop_lm68)
        Ms.append(M)
        crop_rects.append(crop_rect)

    crop_lm68s = np.stack(crop_lm68s)

    return crop_faces, crop_lm68s, crop_rects, Ms, [i for i in range(len(crop_faces))]
