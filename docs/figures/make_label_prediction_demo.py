#!/usr/bin/env python3
"""Render the high-resolution label-prediction GIF embedded in the README."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1600
HEIGHT = 900
OUTPUT = Path(__file__).with_name("label-prediction-demo.gif")

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

BG = "#F8FAFC"
WHITE = "#FFFFFF"
INK = "#172033"
MUTED = "#64748B"
LINE = "#CBD5E1"
BLUE = "#2563EB"
BLUE_LIGHT = "#DBEAFE"
GREEN = "#15803D"
GREEN_LIGHT = "#DCFCE7"
RED = "#B91C1C"
RED_LIGHT = "#FEE2E2"
AMBER = "#B45309"
AMBER_LIGHT = "#FEF3C7"
SLATE_LIGHT = "#F1F5F9"


@dataclass(frozen=True)
class LabelExample:
    name: str
    answer: str
    p_yes: float
    p_no: float
    log_odds: float

    @property
    def included(self) -> bool:
        return self.answer == "yes"


EXAMPLES = [
    LabelExample("love", "yes", 0.89, 0.11, 2.1),
    LabelExample("nervousness", "yes", 0.80, 0.20, 1.4),
    LabelExample("anger", "no", 0.14, 0.86, -1.8),
]


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO if mono else FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(path, size)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    radius: int = 22,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    *,
    fill: str = INK,
) -> None:
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=text_font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    x = left + (right - left - text_width) / 2
    y = top + (bottom - top - text_height) / 2 - text_box[1]
    draw.text((x, y), text, font=text_font, fill=fill)


def chip(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    fill: str,
    outline: str,
    text_fill: str = INK,
    text_font: ImageFont.FreeTypeFont | None = None,
) -> None:
    rounded_box(draw, box, fill=fill, outline=outline, radius=20, width=3)
    centered_text(draw, box, text, text_font or font(28, bold=True), fill=text_fill)


def probability_bar(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    label: str,
    probability: float,
    progress: float,
    color: str,
) -> None:
    label_font = font(25, mono=True)
    value_font = font(23, mono=True)
    draw.text((1060, y), label, font=label_font, fill=INK)
    bar_box = (1160, y + 2, 1435, y + 34)
    rounded_box(draw, bar_box, fill=SLATE_LIGHT, outline=LINE, radius=14, width=2)
    fill_width = int((bar_box[2] - bar_box[0]) * probability * progress)
    if fill_width > 0:
        draw.rounded_rectangle(
            (bar_box[0], bar_box[1], bar_box[0] + fill_width, bar_box[3]),
            radius=12,
            fill=color,
        )
    shown_probability = probability * progress
    draw.text((1452, y + 2), f"{shown_probability:.2f}", font=value_font, fill=MUTED)


def render_frame(
    current_index: int | None,
    phase: str,
    *,
    pulse: bool = False,
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text(
        (64, 38),
        "From one masked answer to a label set",
        font=font(43, bold=True),
        fill=INK,
    )
    chip(
        draw,
        (1324, 39, 1536, 88),
        "ILLUSTRATIVE",
        fill=AMBER_LIGHT,
        outline="#F59E0B",
        text_fill=AMBER,
        text_font=font(20, bold=True),
    )

    rounded_box(
        draw,
        (64, 112, 1536, 238),
        fill=WHITE,
        outline=LINE,
        radius=24,
        width=3,
    )
    draw.text((98, 137), "DOCUMENT", font=font(21, bold=True), fill=BLUE)
    draw.text(
        (98, 177),
        '"I love the result, but I am nervous about tomorrow."',
        font=font(31),
        fill=INK,
    )

    draw.text((64, 275), "Candidate labels", font=font(25, bold=True), fill=INK)
    chip_width = 360
    chip_gap = 44
    start_x = (WIDTH - (chip_width * 3 + chip_gap * 2)) // 2
    included_before = {
        example.name
        for index, example in enumerate(EXAMPLES)
        if current_index is not None
        and (index < current_index or (index == current_index and phase == "reveal"))
        and example.included
    }
    excluded_before = {
        example.name
        for index, example in enumerate(EXAMPLES)
        if current_index is not None
        and (index < current_index or (index == current_index and phase == "reveal"))
        and not example.included
    }

    for index, example in enumerate(EXAMPLES):
        x = start_x + index * (chip_width + chip_gap)
        box = (x, 316, x + chip_width, 375)
        if index == current_index and phase != "reveal":
            chip(
                draw,
                box,
                example.name,
                fill=BLUE_LIGHT,
                outline=BLUE,
                text_fill="#1D4ED8",
            )
        elif example.name in included_before:
            chip(
                draw,
                box,
                f"{example.name}  +",
                fill=GREEN_LIGHT,
                outline=GREEN,
                text_fill=GREEN,
            )
        elif example.name in excluded_before:
            chip(
                draw,
                box,
                f"{example.name}  x",
                fill=RED_LIGHT,
                outline=RED,
                text_fill=RED,
            )
        else:
            chip(draw, box, example.name, fill=WHITE, outline=LINE, text_fill=MUTED)

    rounded_box(
        draw,
        (64, 411, 1536, 724),
        fill=WHITE,
        outline=LINE,
        radius=26,
        width=3,
    )
    draw.line((1010, 440, 1010, 692), fill=LINE, width=3)

    if current_index is None:
        draw.text((104, 452), "Same prompt shape for every label", font=font(24, bold=True), fill=BLUE)
        draw.text(
            (104, 505),
            'Question: Does this document express "<label>"?',
            font=font(29),
            fill=INK,
        )
        draw.text((104, 588), "Answer:", font=font(30, bold=True), fill=INK)
        mask_fill = BLUE_LIGHT
        mask_outline = BLUE
        chip(
            draw,
            (282, 568, 548, 631),
            "[MASK]",
            fill=mask_fill,
            outline=mask_outline,
            text_fill="#1D4ED8",
            text_font=font(29, bold=True, mono=True),
        )
        draw.text((1060, 457), "Masked-token scores", font=font(27, bold=True), fill=INK)
        draw.text(
            (1060, 525),
            "The frozen model compares",
            font=font(24),
            fill=MUTED,
        )
        draw.text((1060, 563), '"yes" with "no".', font=font(24), fill=MUTED)
        draw.text(
            (104, 661),
            "Start with an empty prediction set.",
            font=font(22),
            fill=MUTED,
        )
    else:
        example = EXAMPLES[current_index]
        draw.text(
            (104, 448),
            f"LABEL {current_index + 1} OF {len(EXAMPLES)}",
            font=font(21, bold=True),
            fill=BLUE,
        )
        draw.text(
            (104, 497),
            f'Question: Does this document express "{example.name}"?',
            font=font(28),
            fill=INK,
        )
        draw.text((104, 588), "Answer:", font=font(30, bold=True), fill=INK)

        if phase == "reveal":
            answer_fill = GREEN_LIGHT if example.included else RED_LIGHT
            answer_outline = GREEN if example.included else RED
            answer_text = GREEN if example.included else RED
            chip(
                draw,
                (282, 568, 548, 631),
                example.answer.upper(),
                fill=answer_fill,
                outline=answer_outline,
                text_fill=answer_text,
                text_font=font(31, bold=True, mono=True),
            )
            decision = "include label" if example.included else "leave label out"
            decision_color = GREEN if example.included else RED
            draw.text(
                (104, 661),
                f"Calibrated decision: {decision}",
                font=font(23, bold=True),
                fill=decision_color,
            )
            progress = 1.0
        else:
            mask_fill = "#BFDBFE" if pulse else BLUE_LIGHT
            mask_outline = "#1D4ED8" if pulse else BLUE
            chip(
                draw,
                (282, 568, 548, 631),
                "[MASK]",
                fill=mask_fill,
                outline=mask_outline,
                text_fill="#1D4ED8",
                text_font=font(29, bold=True, mono=True),
            )
            progress = 0.0 if phase == "mask" else 0.5 if phase == "score_half" else 1.0
            status = (
                "Mask the answer position."
                if phase == "mask"
                else 'Score "yes" against "no".'
            )
            draw.text((104, 661), status, font=font(23), fill=MUTED)

        draw.text((1060, 452), "Masked-token scores", font=font(27, bold=True), fill=INK)
        probability_bar(
            draw,
            y=520,
            label="yes",
            probability=example.p_yes,
            progress=progress,
            color=GREEN,
        )
        probability_bar(
            draw,
            y=584,
            label="no",
            probability=example.p_no,
            progress=progress,
            color=RED,
        )
        if phase == "reveal":
            sign = "+" if example.log_odds >= 0 else ""
            draw.text(
                (1060, 655),
                f"log-odds u = {sign}{example.log_odds:.1f}",
                font=font(23, bold=True, mono=True),
                fill=INK,
            )
        else:
            draw.text(
                (1060, 655),
                "read scores at [MASK]",
                font=font(22, mono=True),
                fill=MUTED,
            )

    rounded_box(
        draw,
        (64, 752, 1536, 837),
        fill="#EFF6FF",
        outline="#93C5FD",
        radius=22,
        width=3,
    )
    draw.text((98, 780), "Predicted set", font=font(24, bold=True), fill="#1D4ED8")
    if included_before:
        set_text = "{ " + ", ".join(
            example.name for example in EXAMPLES if example.name in included_before
        ) + " }"
        draw.text((320, 776), set_text, font=font(29, bold=True, mono=True), fill=GREEN)
    else:
        draw.text((320, 779), "{ }", font=font(29, bold=True, mono=True), fill=MUTED)

    draw.text(
        (64, 860),
        "Conceptual animation: dLLM-SetScore reads yes/no scores at [MASK]; it does not generate a free-form answer.",
        font=font(18),
        fill=MUTED,
    )
    draw.text(
        (1372, 860),
        "1600 x 900",
        font=font(18, mono=True),
        fill=MUTED,
    )
    return image


def main() -> None:
    frames: list[Image.Image] = []
    durations: list[int] = []

    frames.append(render_frame(None, "intro"))
    durations.append(1800)

    for index in range(len(EXAMPLES)):
        frames.append(render_frame(index, "mask"))
        durations.append(1100)
        frames.append(render_frame(index, "score_half", pulse=True))
        durations.append(700)
        frames.append(render_frame(index, "score_full"))
        durations.append(900)
        frames.append(render_frame(index, "reveal"))
        durations.append(1700)

    frames.append(render_frame(len(EXAMPLES) - 1, "reveal"))
    durations.append(3000)

    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Wrote {OUTPUT} ({WIDTH}x{HEIGHT}, {len(frames)} frames)")


if __name__ == "__main__":
    main()
