import inspect
import math
import os
from typing import Callable, List, Optional, Union

import numpy as np
import torch

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_video, get_scheduler_timesteps, one_step_sampling
from util.util import VideoWriter
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..utils.face_utils import detect_and_track_face, crop_and_align_face
from ..whisper.audio2feature import Audio2Feature
import tqdm
from insightface.app import FaceAnalysis
from deep_sort import nn_matching
from deep_sort.tracker import Tracker

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        insightface_root: Optional[str] = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

        self.insightface_app = FaceAnalysis(
            name="buffalo_l",
            root=insightface_root or os.environ.get("INSIGHTFACE_ROOT", os.path.expanduser("~/.insightface")),
            providers=[('CUDAExecutionProvider', {'device_id': '0'})]
        )
        self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        batch_size = 16
        decoded_latents = []
        for idx in range(0, len(latents), batch_size):
            decoded_latents.append(
                self.vae.decode(latents[idx:idx + batch_size]).sample
            )
        decoded_latents = torch.cat(decoded_latents)
        # decoded_latents += torch.tensor([[[[0.0009]], [[0.0241]], [[0.0045]]]]).to(
        #     device=decoded_latents.device, dtype=decoded_latents.dtype
        # )

        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            1,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )  # (b, c, f, h, w)
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        batch_size = 16
        masked_image_latents = []
        for idx in range(0, len(masked_image), batch_size):
            masked_image_latents.append(
                self.vae.encode(masked_image[idx:idx + batch_size]).latent_dist.sample(generator=generator)
            )
        masked_image_latents = torch.cat(masked_image_latents)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        batch_size = 16
        image_latents = []
        for idx in range(0, len(images), batch_size):
            image_latents.append(
                self.vae.encode(images[idx:idx + batch_size]).latent_dist.sample(generator=generator)
            )
        image_latents = torch.cat(image_latents)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray, tracking=False):
        if len(whisper_chunks) > len(video_frames):
            if tracking:
                metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, 100)
                tracker = Tracker(metric)
                det_res = detect_and_track_face(list(video_frames), self.insightface_app, tracker, first_crop=False)
            else:
                det_res = crop_and_align_face(list(video_frames), self.insightface_app)
            if det_res is None:
                print("No face detected.")
                return None

            align_faces, align_lm68s, crop_rects, Ms, face_flags = det_res

            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_face_flags = []
            loop_align_faces = []
            loop_align_lm68s = []
            loop_crop_rects = []
            loop_Ms = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_face_flags.append(face_flags)
                    loop_align_faces.append(np.stack(align_faces))
                    loop_align_lm68s.append(align_lm68s)
                    loop_crop_rects.append(np.stack(crop_rects))
                    loop_Ms.append(np.stack(Ms))
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_face_flags.append(face_flags[::-1])
                    loop_align_faces.append(np.stack(align_faces)[::-1])
                    loop_align_lm68s.append(align_lm68s[::-1])
                    loop_crop_rects.append(np.stack(crop_rects)[::-1])
                    loop_Ms.append(np.stack(Ms)[::-1])

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            face_flags = np.concatenate(loop_face_flags, axis=0)[: len(whisper_chunks)]
            align_faces = np.concatenate(loop_align_faces, axis=0)[: len(whisper_chunks)]
            align_lm68s = np.concatenate(loop_align_lm68s, axis=0)[: len(whisper_chunks)]
            crop_rects = np.concatenate(loop_crop_rects, axis=0)[: len(whisper_chunks)]
            Ms = np.concatenate(loop_Ms, axis=0)[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            if tracking:
                metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, 100)
                tracker = Tracker(metric)
                det_res = detect_and_track_face(list(video_frames), self.insightface_app, tracker, first_crop=False)
            else:
                det_res = crop_and_align_face(list(video_frames), self.insightface_app)
            if det_res is None:
                print("No face detected.")
                return None
            align_faces, align_lm68s, crop_rects, Ms, face_flags = det_res
            align_faces = np.stack(align_faces)
            face_flags = np.stack(face_flags)
            crop_rects = np.stack(crop_rects)
            Ms = np.stack(Ms)

        return video_frames, align_faces, align_lm68s, crop_rects, Ms, face_flags

    def get_inv_face(self, pred_fake, org_shape, M, need_mask=True):
        M = M.copy()[:2]
        inv_face = cv2.warpAffine(pred_fake, cv2.invertAffineTransform(M), org_shape, borderMode=cv2.INTER_NEAREST)

        if need_mask:
            up = 10
            dp = pred_fake.shape[0] - 10
            box = np.array([[up, up, 1], [up, dp, 1], [dp, dp, 1], [dp, up, 1]]).astype(np.float32).transpose(1, 0)
            trans_box = (cv2.invertAffineTransform(M) @ box).transpose(1, 0).astype(np.int32)
            trans_box = trans_box.reshape((-1, 1, 2))
            inv_mask = np.zeros_like(inv_face)
            cv2.fillPoly(inv_mask, [trans_box], (255, 255, 255), lineType=cv2.LINE_AA)
            inv_mask = inv_mask / 255.0
            return inv_face, inv_mask
        else:
            return inv_face


    @torch.no_grad()
    def stage1_forward(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(height, width, callback_steps)

        do_classifier_free_guidance = guidance_scale > 1.0

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
        total_num_frames = len(whisper_chunks)

        video_frames = read_video(video_path, change_fps=False, use_decord=False)

        det_res = self.loop_video(whisper_chunks, video_frames)
        if det_res is None:
            raise RuntimeError(f"No face detected in {video_path}")
        video_frames, faces, lms, crop_rects, Ms, face_flags = det_res

        num_channels_latents = self.vae.config.latent_channels

        all_latents = self.prepare_latents(
            total_num_frames,
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        ref_pixel_values, masked_pixel_values, masks, blur_masks = self.image_processor.prepare_masks_and_masked_images(
            faces, lms=lms, need_blur=True
        )

        mask_latents, masked_image_latents = self.prepare_mask_latents(
            masks,
            masked_pixel_values,
            height,
            width,
            weight_dtype,
            device,
            generator,
            False,
        )

        ref_latents = self.prepare_image_latents(
            ref_pixel_values,
            device,
            weight_dtype,
            generator,
            False,
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:

            # fl_frame_idxs = [i for i in range(0, total_num_frames, num_frames - 1)]
            # if fl_frame_idxs[-1] < total_num_frames - 1:
            #     fl_frame_idxs.append(total_num_frames - 1)
            # fl_frame_len = len(fl_frame_idxs)
            #
            # fl_all_latents = all_latents.clone()[:, :, fl_frame_idxs]
            #
            # for j, t in enumerate(timesteps):
            #     for index_start in range(0, fl_frame_len, num_frames):
            #         idx_list = fl_frame_idxs[index_start:index_start + num_frames]
            #         latents = []
            #         audio_embeds = []
            #         infer_mask_latents = []
            #         infer_masked_image_latents = []
            #         infer_ref_latents = []
            #         for idx in idx_list:
            #             if idx >= all_latents.shape[2]:
            #                 idx = all_latents.shape[2] - 1
            #             latents.append(fl_all_latents[:, :, fl_frame_idxs.index(idx)])
            #             audio_embeds.append(whisper_chunks[idx])
            #             infer_mask_latents.append(mask_latents[:, :, idx])
            #             infer_masked_image_latents.append(masked_image_latents[:, :, idx])
            #             infer_ref_latents.append(ref_latents[:, :, idx])
            #         latents = torch.stack(latents, 2)
            #         audio_embeds = torch.stack(audio_embeds, 0).to(device, dtype=weight_dtype)
            #         infer_mask_latents = torch.stack(infer_mask_latents, 2)
            #         infer_masked_image_latents = torch.stack(infer_masked_image_latents, 2)
            #         infer_ref_latents = torch.stack(infer_ref_latents, 2)
            #
            #         input_latents = torch.cat([latents] * 2)
            #         null_audio_embeds = torch.zeros_like(audio_embeds)
            #         audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            #         motion_switchs = torch.tensor([0, 0]).to(device, dtype=weight_dtype)
            #         infer_mask_latents = torch.cat([infer_mask_latents] * 2)
            #         infer_masked_image_latents = torch.cat([infer_masked_image_latents] * 2)
            #         infer_ref_latents = torch.cat([infer_ref_latents] * 2)
            #
            #         input_latents = torch.cat(
            #             [input_latents, infer_mask_latents, infer_masked_image_latents, infer_ref_latents],
            #             dim=1
            #         )
            #         noise_pred = self.unet(
            #             input_latents, t, encoder_hidden_states=audio_embeds, motion_switchs=motion_switchs
            #         ).sample
            #
            #         noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
            #         noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)
            #
            #         latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            #
            #         for iii in range(num_frames):
            #             p = index_start + iii
            #             if p >= fl_all_latents.shape[2]:
            #                 break
            #             fl_all_latents[:, :, p] = latents[:, :, iii]

            shift = 0
            shift_offset = 5
            overlap = 3

            for j, t in enumerate(timesteps):
                pred_latents = torch.zeros_like(all_latents)
                counter = torch.zeros((1, 1, total_num_frames, 1, 1)).to(device, weight_dtype)
                for index_start in range(0, total_num_frames, num_frames - overlap):
                    index_start -= shift
                    idx_list = list(range(index_start, index_start + num_frames))
                    latents = []
                    audio_embeds = []
                    infer_mask_latents = []
                    infer_masked_image_latents = []
                    infer_ref_latents = []
                    for idx in idx_list:
                        if idx >= all_latents.shape[2]:
                            idx = all_latents.shape[2] - 1
                        latents.append(all_latents[:, :, idx])
                        audio_embeds.append(whisper_chunks[idx])
                        infer_mask_latents.append(mask_latents[:, :, idx])
                        # if idx in fl_frame_idxs and (len(latents) == 1 or len(latents) == num_frames):
                        #     infer_masked_image_latents.append(fl_all_latents[:, :, fl_frame_idxs.index(idx)])
                        # else:
                        infer_masked_image_latents.append(masked_image_latents[:, :, idx])
                        infer_ref_latents.append(ref_latents[:, :, idx])
                    latents = torch.stack(latents, 2)
                    audio_embeds = torch.stack(audio_embeds, 0).to(device, dtype=weight_dtype)
                    infer_mask_latents = torch.stack(infer_mask_latents, 2)
                    infer_masked_image_latents = torch.stack(infer_masked_image_latents, 2)
                    infer_ref_latents = torch.stack(infer_ref_latents, 2)

                    input_latents = torch.cat([latents] * 3)
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds, null_audio_embeds])
                    motion_switchs = torch.tensor([0, 0, 1]).to(device, dtype=weight_dtype)
                    infer_mask_latents = torch.cat([infer_mask_latents] * 3)
                    infer_masked_image_latents = torch.cat([infer_masked_image_latents] * 3)
                    infer_ref_latents = torch.cat([infer_ref_latents] * 3)

                    input_latents = torch.cat(
                        [input_latents, infer_mask_latents, infer_masked_image_latents, infer_ref_latents],
                        dim=1
                    )
                    noise_pred = self.unet(
                        input_latents, t, encoder_hidden_states=audio_embeds, motion_switchs=motion_switchs
                    ).sample

                    noise_pred_uncond, noise_pred_audio, noise_pred_motion = noise_pred.chunk(3)
                    noise_pred = noise_pred_uncond + \
                                 guidance_scale * (noise_pred_audio - noise_pred_uncond) + \
                                 0.5 * (noise_pred_motion - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    for iii in range(num_frames):
                        p = index_start + iii
                        if p >= all_latents.shape[2]:
                            break
                        pred_latents[:, :, p] += latents[:, :, iii]
                        counter[:, :, p] += 1
                shift += shift_offset
                pred_latents = pred_latents / counter
                all_latents = pred_latents

                progress_bar.update()

        # all_latents[:, :, fl_frame_idxs] = fl_all_latents

        sync_outs = self.decode_latents(all_latents)

        sync_outs = self.paste_surrounding_pixels_back(
            sync_outs, ref_pixel_values, 1 - blur_masks, device, weight_dtype
        )

        org_h, org_w = video_frames[0].shape[:2]
        out_video = VideoWriter(video_out_path, org_h, org_w, fps=video_fps, audio_stream=audio_path)
        for frame, face_flag, sync_out, crop_rect, M in zip(video_frames, face_flags, sync_outs, crop_rects, Ms):
            if face_flag == 0:
                out_video.write(frame)
            else:
                sync_out = (sync_out.cpu() / 2 + 0.5).clamp(0, 1) * 255
                sync_out = np.transpose(sync_out.numpy(), (1, 2, 0)).astype(np.uint8)
                x1, y1, x2, y2 = crop_rect
                inv_face, inv_mask = self.get_inv_face(sync_out, (x2 - x1, y2 - y1), M, need_mask=True)
                frame[y1:y2, x1:x2] = inv_face * inv_mask + frame[y1:y2, x1:x2] * (1 - inv_mask)
                out_video.write(frame)
        out_video.close()

        if is_train:
            self.unet.train()

    @torch.no_grad()
    def noalign_occ_forward(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(height, width, callback_steps)

        self.scheduler.timesteps = get_scheduler_timesteps(infer_step=num_inference_steps, device=device)[:-1]
        if num_inference_steps == 1:
            self.scheduler.timesteps -= 1
        timesteps = self.scheduler.timesteps

        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
        total_num_frames = len(whisper_chunks)

        video_frames = read_video(video_path, change_fps=False, use_decord=False)

        det_res = self.loop_video(whisper_chunks, video_frames, tracking=True)
        if det_res is None:
            raise RuntimeError(f"No face detected in {video_path}")
        video_frames, faces, lms, crop_rects, Ms, face_flags = det_res

        num_channels_latents = self.vae.config.latent_channels

        all_latents = self.prepare_latents(
            total_num_frames,
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        ref_pixel_values, masked_pixel_values, masks, blur_masks = self.image_processor.prepare_masks_and_masked_images(
            faces, lms=lms, need_blur=True
        )

        mask_latents, masked_image_latents = self.prepare_mask_latents(
            masks,
            masked_pixel_values,
            height,
            width,
            weight_dtype,
            device,
            generator,
            False,
        )

        ref_latents = self.prepare_image_latents(
            ref_pixel_values,
            device,
            weight_dtype,
            generator,
            False,
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:

            # fl_frame_idxs = [i for i in range(0, total_num_frames, num_frames - 1)]
            # if fl_frame_idxs[-1] < total_num_frames - 1:
            #     fl_frame_idxs.append(total_num_frames - 1)
            # fl_frame_len = len(fl_frame_idxs)
            #
            # fl_all_latents = all_latents.clone()[:, :, fl_frame_idxs]
            #
            # for j, t in enumerate(timesteps):
            #     for index_start in range(0, fl_frame_len, num_frames):
            #         idx_list = fl_frame_idxs[index_start:index_start + num_frames]
            #         latents = []
            #         audio_embeds = []
            #         infer_mask_latents = []
            #         infer_masked_image_latents = []
            #         infer_ref_latents = []
            #         for idx in idx_list:
            #             if idx >= all_latents.shape[2]:
            #                 idx = all_latents.shape[2] - 1
            #             latents.append(fl_all_latents[:, :, fl_frame_idxs.index(idx)])
            #             audio_embeds.append(whisper_chunks[idx])
            #             infer_mask_latents.append(mask_latents[:, :, idx])
            #             infer_masked_image_latents.append(masked_image_latents[:, :, idx])
            #             infer_ref_latents.append(ref_latents[:, :, idx])
            #         latents = torch.stack(latents, 2)
            #         audio_embeds = torch.stack(audio_embeds, 0).to(device, dtype=weight_dtype)
            #         infer_mask_latents = torch.stack(infer_mask_latents, 2)
            #         infer_masked_image_latents = torch.stack(infer_masked_image_latents, 2)
            #         infer_ref_latents = torch.stack(infer_ref_latents, 2)
            #
            #         input_latents = latents.clone()
            #         motion_switchs = torch.tensor([0]).to(device, dtype=weight_dtype)
            #
            #         input_latents = torch.cat(
            #             [input_latents, infer_mask_latents, infer_masked_image_latents, infer_ref_latents],
            #             dim=1
            #         )
            #         noise_pred = self.unet(
            #             input_latents, t, encoder_hidden_states=audio_embeds, motion_switchs=motion_switchs
            #         ).sample
            #
            #         latents = one_step_sampling(self.scheduler, noise_pred, t.unsqueeze(0), latents)
            #
            #         for iii in range(num_frames):
            #             p = index_start + iii
            #             if p >= fl_all_latents.shape[2]:
            #                 break
            #             fl_all_latents[:, :, p] = latents[:, :, iii]

            for j, t in enumerate(timesteps):
                pred_latents = torch.zeros_like(all_latents)
                counter = torch.zeros((1, 1, total_num_frames, 1, 1)).to(device, weight_dtype)
                for index_start in range(0, total_num_frames, num_frames - 1):
                    idx_list = list(range(index_start, index_start + num_frames))
                    latents = []
                    audio_embeds = []
                    infer_mask_latents = []
                    infer_masked_image_latents = []
                    infer_ref_latents = []
                    for idx in idx_list:
                        if idx >= all_latents.shape[2]:
                            idx = all_latents.shape[2] - 1
                        latents.append(all_latents[:, :, idx])
                        audio_embeds.append(whisper_chunks[idx])
                        infer_mask_latents.append(mask_latents[:, :, idx])
                        # if idx in fl_frame_idxs and (len(latents) == 1 or len(latents) == num_frames):
                        #     infer_masked_image_latents.append(fl_all_latents[:, :, fl_frame_idxs.index(idx)])
                        # else:
                        infer_masked_image_latents.append(masked_image_latents[:, :, idx])
                        infer_ref_latents.append(ref_latents[:, :, idx])
                    latents = torch.stack(latents, 2)
                    audio_embeds = torch.stack(audio_embeds, 0).to(device, dtype=weight_dtype)
                    infer_mask_latents = torch.stack(infer_mask_latents, 2)
                    infer_masked_image_latents = torch.stack(infer_masked_image_latents, 2)
                    infer_ref_latents = torch.stack(infer_ref_latents, 2)

                    input_latents = latents.clone()
                    motion_switchs = torch.tensor([1]).to(device, dtype=weight_dtype)

                    input_latents = torch.cat(
                        [input_latents, infer_mask_latents, infer_masked_image_latents, infer_ref_latents],
                        dim=1
                    )
                    noise_pred = self.unet(
                        input_latents, t, encoder_hidden_states=audio_embeds, motion_switchs=motion_switchs
                    ).sample

                    latents = one_step_sampling(self.scheduler, noise_pred, t.unsqueeze(0), latents)

                    for iii in range(num_frames):
                        p = index_start + iii
                        if p >= all_latents.shape[2]:
                            break
                        pred_latents[:, :, p] += latents[:, :, iii]
                        counter[:, :, p] += 1
                pred_latents = pred_latents / counter
                all_latents = pred_latents

                progress_bar.update()

        # all_latents[:, :, fl_frame_idxs] = fl_all_latents

        sync_outs = self.decode_latents(all_latents)

        sync_outs = self.paste_surrounding_pixels_back(
            sync_outs, ref_pixel_values, 1 - blur_masks, device, weight_dtype
        )

        org_h, org_w = video_frames[0].shape[:2]
        out_video = VideoWriter(video_out_path, org_h, org_w, fps=video_fps, audio_stream=audio_path)
        for frame, face_flag, sync_out, crop_rect, M in zip(video_frames, face_flags, sync_outs, crop_rects, Ms):
            if face_flag == 0:
                out_video.write(frame)
            else:
                sync_out = (sync_out.cpu() / 2 + 0.5).clamp(0, 1) * 255
                sync_out = np.transpose(sync_out.numpy(), (1, 2, 0)).astype(np.uint8)
                inv_face, inv_mask = self.get_inv_face(sync_out, (org_w, org_h), M, need_mask=True)
                frame = inv_face * inv_mask + frame * (1 - inv_mask)
                out_video.write(frame)
        out_video.close()

        if is_train:
            self.unet.train()
