from __future__ import annotations

from pathlib import Path
from typing import List

import struct
import zlib


def _write_png(path: str, width: int, height: int, pixels: List[List[tuple[int, int, int]]]) -> None:
    raw = b""
    for row in pixels:
        raw += b"\x00" + b"".join(bytes([r, g, b]) for (r, g, b) in row)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def _blank_canvas(width: int, height: int, bg: tuple[int, int, int] = (255, 255, 255)):
    return [[bg for _ in range(width)] for _ in range(height)]


def _draw_line(pixels, x0, y0, x1, y1, color):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= y0 < len(pixels) and 0 <= x0 < len(pixels[0]):
            pixels[y0][x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _draw_rect(pixels, x0, y0, x1, y1, color):
    for y in range(max(0, y0), min(len(pixels), y1)):
        for x in range(max(0, x0), min(len(pixels[0]), x1)):
            pixels[y][x] = color


def save_reward_curve(rewards: List[float], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 520
    pixels = _blank_canvas(width, height)
    x0, y0, x1, y1 = 70, 70, 860, 470
    _draw_line(pixels, x0, y0, x1, y0, (0, 0, 0))
    _draw_line(pixels, x0, y1, x1, y1, (0, 0, 0))
    _draw_line(pixels, x0, y0, x0, y1, (0, 0, 0))
    _draw_line(pixels, x1, y0, x1, y1, (0, 0, 0))
    if rewards:
        pts = []
        for i, r in enumerate(rewards):
            x = x0 + int((x1 - x0) * (i / max(1, len(rewards) - 1)))
            y = y1 - int((y1 - y0) * max(0.0, min(1.0, r)))
            pts.append((x, y))
        for i in range(1, len(pts)):
            _draw_line(pixels, pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], (53, 106, 230))
    _write_png(output_path, width, height, pixels)


def save_baseline_vs_trained(baseline: float, trained: float, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    width, height = 700, 500
    pixels = _blank_canvas(width, height)
    x0, y0, x1, y1 = 90, 80, 650, 440
    _draw_line(pixels, x0, y0, x1, y0, (0, 0, 0))
    _draw_line(pixels, x0, y1, x1, y1, (0, 0, 0))
    _draw_line(pixels, x0, y0, x0, y1, (0, 0, 0))
    _draw_line(pixels, x1, y0, x1, y1, (0, 0, 0))
    bar_w = 120
    b_h = int((y1 - y0) * max(0.0, min(1.0, baseline)))
    t_h = int((y1 - y0) * max(0.0, min(1.0, trained)))
    _draw_rect(pixels, 180, y1 - b_h, 180 + bar_w, y1, (102, 102, 204))
    _draw_rect(pixels, 400, y1 - t_h, 400 + bar_w, y1, (34, 170, 102))
    _write_png(output_path, width, height, pixels)
