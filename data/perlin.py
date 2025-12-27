import torch
import math
import numpy as np
from typing import Union, Tuple
import cv2
from torch.nn import functional as F


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(
        np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1], axis=1)
    dot = lambda grad, shift: (
            np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                     axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
            torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                        dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])

    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def get_perlin_noise(perlin_size: Union[Tuple[int, int], list[int]],
                     sample_size=(224, 224),
                     perlin_scale=6,
                     min_perlin_scale=0,
                     threshold=0.5):
    # generate perlin noise
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_shapex = math.ceil(sample_size[0] / perlin_scalex) * perlin_scalex
    perlin_shapey = math.ceil(sample_size[1] / perlin_scaley) * perlin_scaley
    perlin_noise = rand_perlin_2d_np((perlin_shapex, perlin_shapey), (perlin_scalex, perlin_scaley))

    perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
    perlin_thr = cv2.morphologyEx(perlin_thr, cv2.MORPH_ERODE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                  iterations=1)
    perlin_thr = cv2.morphologyEx(perlin_thr, cv2.MORPH_DILATE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                  iterations=1)
    perlin_thr = cv2.resize(perlin_thr, (perlin_size[0], perlin_size[1]))
    perlin_thr[perlin_thr > 0] = 255
    perlin_thr[perlin_thr < 255] = 0
    return perlin_thr


def image_blender_with_perlin_noise(image: torch.Tensor,
                                    texture_image: torch.Tensor,
                                    perlin_noise: np.ndarray,
                                    image_weight: float,
                                    texture_weight: float, ):
    assert image_weight + texture_weight == 1
    image = image.unsqueeze(0)
    texture_image = texture_image.unsqueeze(0).contiguous()
    perlin_noise = torch.from_numpy(perlin_noise).unsqueeze(0).unsqueeze(0)
    texture_image = F.interpolate(texture_image, size=image.shape[-2:], mode='bilinear', align_corners=True)
    perlin_noise = F.interpolate(perlin_noise, size=image.shape[-2:], mode='bilinear', align_corners=True)
    assert image.shape == texture_image.shape and image.shape[-2:] == perlin_noise.shape[-2:]
    perlin_noise[perlin_noise > 0] = 1
    perlin_noise[perlin_noise < 1] = 0
    image = torch.where(perlin_noise > 0, texture_image * texture_weight + image * image_weight, image)
    image = image.squeeze(0)
    perlin_noise = perlin_noise.squeeze(0)
    return image, perlin_noise, texture_image


def convert_mask_to_coco_annotations(mask: torch.Tensor, ):
    mask = mask.squeeze().cpu().numpy()
    assert len(mask.shape) == 2
    mask = mask.astype(np.uint8)  # to CV_8UC1
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    coco_annotations = []
    for contour in contours:
        contour_points = np.squeeze(contour).tolist()
        segmentation = []
        for p in contour_points:
            segmentation.append(p[0])
            segmentation.append(p[1])
        if len(segmentation) < 6:
            continue
        annotation = {'segmentation': [segmentation],
                      'area': cv2.contourArea(contour),
                      'bbox': cv2.boundingRect(contour)}
        coco_annotations.append(annotation)

    return coco_annotations
