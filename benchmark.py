#!/usr/bin/env python3
"""
Standalone benchmark driver for KleidiCV / OpenCV perf tests.

Two subcommands:

  run      Execute every embedded benchmark case against the perf binaries
           located in a configurable OpenCV build directory and write a TSV
           result file.

  compare  Diff two TSV result files (e.g. baseline vs candidate) and render
           a bar chart PNG of the speed-up per operation.

  show     Print the perf binary command for a given test case name (handy
           for debugging or running a single case manually).

Examples
--------
  ./benchmark.py run --opencv-build build-opencv-4 --output baseline.tsv

  ./benchmark.py compare --baseline baseline.tsv \\
                         --candidate candidate.tsv \\
                         --output output.png

  ./benchmark.py show GaussianBlur5x5 --opencv-build build-opencv-4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Spec:
    name: str
    binary: str
    gtest_filter: str
    gtest_params: str


@dataclass
class Result:
    name: str
    mean: float
    gstddev: float


# ---------------------------------------------------------------------------
# Embedded benchmark cases, fully expanded at FHD (1920x1080).
# ---------------------------------------------------------------------------

CASES: list[Spec] = [
    Spec(
        "GRAY2BGR",
        "opencv_perf_imgproc",
        "*cvtColor8u/*",
        "(1920x1080, COLOR_GRAY2BGR)",
    ),
    Spec(
        "GRAY2BGRA",
        "opencv_perf_imgproc",
        "*cvtColor8u/*",
        "(1920x1080, COLOR_GRAY2BGRA)",
    ),
    Spec(
        "BGR2RGB", "opencv_perf_imgproc", "*cvtColor8u/*", "(1920x1080, COLOR_BGR2RGB)"
    ),
    Spec(
        "BGRA2RGBA",
        "opencv_perf_imgproc",
        "*cvtColor8u/*",
        "(1920x1080, COLOR_BGRA2RGBA)",
    ),
    Spec(
        "BGR2RGBA",
        "opencv_perf_imgproc",
        "*cvtColor8u/*",
        "(1920x1080, COLOR_BGR2RGBA)",
    ),
    Spec(
        "BGR2BGRA",
        "opencv_perf_imgproc",
        "*cvtColor8u/*",
        "(1920x1080, COLOR_BGR2BGRA)",
    ),
    Spec(
        "RGBA2BGR",
        "opencv_perf_imgproc",
        "*cvtColor8u/*",
        "(1920x1080, COLOR_RGBA2BGR)",
    ),
    Spec(
        "BGRA2BGR",
        "opencv_perf_imgproc",
        "*cvtColor8u/*",
        "(1920x1080, COLOR_BGRA2BGR)",
    ),
    Spec(
        "YUVSP2BGR",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2BGR_NV12)",
    ),
    Spec(
        "YUVSP2BGRA",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2BGRA_NV12)",
    ),
    Spec(
        "YUVSP2RGB",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2RGB_NV12)",
    ),
    Spec(
        "YUVSP2RGBA",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2RGBA_NV12)",
    ),
    Spec(
        "YUVP2BGR",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2BGR_YV12)",
    ),
    Spec(
        "YUVP2BGRA",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2BGRA_YV12)",
    ),
    Spec(
        "YUVP2RGB",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2RGB_YV12)",
    ),
    Spec(
        "YUVP2RGBA",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2RGBA_YV12)",
    ),
    Spec(
        "RGB2YUV_UYVY",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_RGB2YUV_UYVY)",
    ),
    Spec(
        "BGR2YUV_UYVY",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_BGR2YUV_UYVY)",
    ),
    Spec(
        "RGB2YUV_YUY2",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_RGB2YUV_YUY2)",
    ),
    Spec(
        "BGR2YUV_YUY2",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_BGR2YUV_YUY2)",
    ),
    Spec(
        "RGB2YUV_YVYU",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_RGB2YUV_YVYU)",
    ),
    Spec(
        "BGR2YUV_YVYU",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_BGR2YUV_YVYU)",
    ),
    Spec(
        "RGBA2YUV_YUY2",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_RGBA2YUV_YUY2)",
    ),
    Spec(
        "BGRA2YUV_YUY2",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_BGRA2YUV_YUY2)",
    ),
    Spec(
        "RGBA2YUV_YVYU",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_RGBA2YUV_YVYU)",
    ),
    Spec(
        "BGRA2YUV_YVYU",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_BGRA2YUV_YVYU)",
    ),
    Spec(
        "RGBA2YUV_UYVY",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_RGBA2YUV_UYVY)",
    ),
    Spec(
        "BGRA2YUV_UYVY",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_BGRA2YUV_UYVY)",
    ),
    Spec(
        "YUV2RGB_UYVY",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2RGB_UYVY)",
    ),
    Spec(
        "YUV2BGR_UYVY",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2BGR_UYVY)",
    ),
    Spec(
        "YUV2RGB_YUY2",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2RGB_YUY2)",
    ),
    Spec(
        "YUV2BGR_YUY2",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2BGR_YUY2)",
    ),
    Spec(
        "YUV2RGB_YVYU",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2RGB_YVYU)",
    ),
    Spec(
        "YUV2BGR_YVYU",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2BGR_YVYU)",
    ),
    Spec(
        "YUV2RGBA_YUY2",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2RGBA_YUY2)",
    ),
    Spec(
        "YUV2BGRA_YUY2",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2BGRA_YUY2)",
    ),
    Spec(
        "YUV2RGBA_YVYU",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2RGBA_YVYU)",
    ),
    Spec(
        "YUV2BGRA_YVYU",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2BGRA_YVYU)",
    ),
    Spec(
        "YUV2RGBA_UYVY",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2RGBA_UYVY)",
    ),
    Spec(
        "YUV2BGRA_UYVY",
        "opencv_perf_imgproc",
        "*cvtColorYUV420/*",
        "(1920x1080, COLOR_YUV2BGRA_UYVY)",
    ),
    Spec(
        "RGB2YUVP",
        "opencv_perf_imgproc",
        "*cvtColorRGB2YUV420p/*",
        "(1920x1080, COLOR_RGB2YUV_YV12)",
    ),
    Spec(
        "RGBA2YUVP",
        "opencv_perf_imgproc",
        "*cvtColorRGB2YUV420p/*",
        "(1920x1080, COLOR_RGBA2YUV_YV12)",
    ),
    Spec(
        "BGR2YUVP",
        "opencv_perf_imgproc",
        "*cvtColorRGB2YUV420p/*",
        "(1920x1080, COLOR_BGR2YUV_YV12)",
    ),
    Spec(
        "BGRA2YUVP",
        "opencv_perf_imgproc",
        "*cvtColorRGB2YUV420p/*",
        "(1920x1080, COLOR_BGRA2YUV_YV12)",
    ),
    Spec(
        "RGB2YUV", "opencv_perf_imgproc", "*cvtColor8u/*", "(1920x1080, COLOR_RGB2YUV)"
    ),
    Spec(
        "BGR2YUV", "opencv_perf_imgproc", "*cvtColor8u/*", "(1920x1080, COLOR_BGR2YUV)"
    ),
    Spec(
        "RGBA2YUV", "opencv_perf_imgproc", "*cvtColor8u/*", "(1920x1080, CX_RGBA2YUV)"
    ),
    Spec(
        "BGRA2YUV", "opencv_perf_imgproc", "*cvtColor8u/*", "(1920x1080, CX_BGRA2YUV)"
    ),
    Spec(
        "YUV2RGB", "opencv_perf_imgproc", "*cvtColor8u/*", "(1920x1080, COLOR_YUV2RGB)"
    ),
    Spec(
        "YUV2BGR", "opencv_perf_imgproc", "*cvtColor8u/*", "(1920x1080, COLOR_YUV2BGR)"
    ),
    Spec(
        "YUV2RGBA", "opencv_perf_imgproc", "*cvtColor8u/*", "(1920x1080, CX_YUV2BGRA)"
    ),
    Spec(
        "YUV2BGRA", "opencv_perf_imgproc", "*cvtColor8u/*", "(1920x1080, CX_YUV2RGBA)"
    ),
    Spec(
        "BinaryThreshold",
        "opencv_perf_imgproc",
        "*ThreshFixture_Threshold.Threshold/*",
        "(1920x1080, 8UC1, THRESH_BINARY)",
    ),
    Spec(
        "SepFilter2D_5x5_U8",
        "opencv_perf_imgproc",
        "*KleidiCV_SepFilter2D.SepFilter2D/*",
        "(1920x1080, 8UC1,  5, BORDER_REPLICATE)",
    ),
    Spec(
        "SepFilter2D_5x5_U16",
        "opencv_perf_imgproc",
        "*KleidiCV_SepFilter2D.SepFilter2D/*",
        "(1920x1080, 16UC1, 5, BORDER_REPLICATE)",
    ),
    Spec(
        "SepFilter2D_5x5_S16",
        "opencv_perf_imgproc",
        "*KleidiCV_SepFilter2D.SepFilter2D/*",
        "(1920x1080, 16SC1, 5, BORDER_REPLICATE)",
    ),
    Spec(
        "MedianBlur3x3", "opencv_perf_imgproc", "*medianBlur/*", "(1920x1080, 8UC1, 3)"
    ),
    Spec(
        "MedianBlur5x5", "opencv_perf_imgproc", "*medianBlur/*", "(1920x1080, 8UC1, 5)"
    ),
    Spec(
        "MedianBlur7x7", "opencv_perf_imgproc", "*medianBlur/*", "(1920x1080, 8UC1, 7)"
    ),
    Spec(
        "MedianBlur9x9", "opencv_perf_imgproc", "*medianBlur/*", "(1920x1080, 8UC1, 9)"
    ),
    Spec(
        "MedianBlur11x11",
        "opencv_perf_imgproc",
        "*medianBlur/*",
        "(1920x1080, 8UC1, 11)",
    ),
    Spec(
        "MedianBlur13x13",
        "opencv_perf_imgproc",
        "*medianBlur/*",
        "(1920x1080, 8UC1, 13)",
    ),
    Spec(
        "MedianBlur15x15",
        "opencv_perf_imgproc",
        "*medianBlur/*",
        "(1920x1080, 8UC1, 15)",
    ),
    Spec(
        "MedianBlur17x17",
        "opencv_perf_imgproc",
        "*medianBlur/*",
        "(1920x1080, 8UC1, 17)",
    ),
    Spec(
        "MedianBlur27x27",
        "opencv_perf_imgproc",
        "*medianBlur/*",
        "(1920x1080, 8UC1, 27)",
    ),
    Spec(
        "MedianBlur35x35",
        "opencv_perf_imgproc",
        "*medianBlur/*",
        "(1920x1080, 8UC1, 35)",
    ),
    Spec(
        "GaussianBlur3x3",
        "opencv_perf_imgproc",
        "*gaussianBlur3x3/*",
        "(1920x1080, 8UC1, BORDER_REPLICATE)",
    ),
    Spec(
        "GaussianBlur5x5",
        "opencv_perf_imgproc",
        "*gaussianBlur5x5/*",
        "(1920x1080, 8UC1, BORDER_REPLICATE)",
    ),
    Spec(
        "GaussianBlur7x7",
        "opencv_perf_imgproc",
        "*gaussianBlur7x7/*",
        "(1920x1080, 8UC1, BORDER_REPLICATE)",
    ),
    Spec(
        "GaussianBlur9x9",
        "opencv_perf_imgproc",
        "*gaussianBlur9x9/*",
        "(1920x1080, 8UC1, BORDER_REPLICATE)",
    ),
    Spec(
        "GaussianBlur3x3_CustomSigma",
        "opencv_perf_imgproc",
        "*gaussianBlur3x3_CustomSigma/*",
        "(1920x1080, 8UC1, BORDER_REPLICATE)",
    ),
    Spec(
        "GaussianBlur5x5_CustomSigma",
        "opencv_perf_imgproc",
        "*gaussianBlur5x5_CustomSigma/*",
        "(1920x1080, 8UC1, BORDER_REPLICATE)",
    ),
    Spec(
        "GaussianBlur7x7_CustomSigma",
        "opencv_perf_imgproc",
        "*gaussianBlur7x7_CustomSigma/*",
        "(1920x1080, 8UC1, BORDER_REPLICATE)",
    ),
    Spec(
        "GaussianBlur9x9_CustomSigma",
        "opencv_perf_imgproc",
        "*gaussianBlur9x9_CustomSigma/*",
        "(1920x1080, 8UC1, BORDER_REPLICATE)",
    ),
    Spec(
        "GaussianBlur15x15_CustomSigma",
        "opencv_perf_imgproc",
        "*gaussianBlur15x15_CustomSigma/*",
        "(1920x1080, 8UC1, BORDER_REPLICATE)",
    ),
    Spec(
        "GaussianBlur21x21_CustomSigma",
        "opencv_perf_imgproc",
        "*gaussianBlur21x21_CustomSigma/*",
        "(1920x1080, 8UC1, BORDER_REPLICATE)",
    ),
    Spec(
        "GaussianBlur49x49_CustomSigma",
        "opencv_perf_imgproc",
        "*gaussianBlur49x49_CustomSigma/*",
        "(1920x1080, 8UC1, BORDER_REPLICATE)",
    ),
    Spec(
        "Sobel_Gx",
        "opencv_perf_imgproc",
        "*Border3x3_sobelFilter.sobelFilter/*",
        "(1920x1080, 16SC1, (1, 0), BORDER_REPLICATE)",
    ),
    Spec(
        "Sobel_Gy",
        "opencv_perf_imgproc",
        "*Border3x3_sobelFilter.sobelFilter/*",
        "(1920x1080, 16SC1, (0, 1), BORDER_REPLICATE)",
    ),
    Spec(
        "Dilate3x3", "opencv_perf_imgproc", "*Dilate_big.big/*", "(1920x1080, 8UC1, 3)"
    ),
    Spec(
        "Dilate5x5", "opencv_perf_imgproc", "*Dilate_big.big/*", "(1920x1080, 8UC1, 5)"
    ),
    Spec(
        "Dilate17x17",
        "opencv_perf_imgproc",
        "*Dilate_big.big/*",
        "(1920x1080, 8UC1, 17)",
    ),
    Spec("Erode3x3", "opencv_perf_imgproc", "*Erode_big.big/*", "(1920x1080, 8UC1, 3)"),
    Spec("Erode5x5", "opencv_perf_imgproc", "*Erode_big.big/*", "(1920x1080, 8UC1, 5)"),
    Spec(
        "Erode17x17", "opencv_perf_imgproc", "*Erode_big.big/*", "(1920x1080, 8UC1, 17)"
    ),
    Spec(
        "Resize_0.5_8b",
        "opencv_perf_imgproc",
        "*ResizeAreaFast/*",
        "(8UC1, 1920x1080, 2)",
    ),
    Spec(
        "Resize_0.5_8b_2ch",
        "opencv_perf_imgproc",
        "*ResizeAreaFast/*",
        "(8UC2, 1920x1080, 2)",
    ),
    Spec(
        "Resize_0.5_8b_3ch",
        "opencv_perf_imgproc",
        "*ResizeAreaFast/*",
        "(8UC3, 1920x1080, 2)",
    ),
    Spec(
        "Resize_0.5_8b_4ch",
        "opencv_perf_imgproc",
        "*ResizeAreaFast/*",
        "(8UC4, 1920x1080, 2)",
    ),
    Spec(
        "Scale",
        "opencv_perf_core",
        "*convertTo/*",
        "(1920x1080, 8UC1,  8UC1,  1, 1.234, 4.567)",
    ),
    Spec(
        "Scale_float_1.0",
        "opencv_perf_core",
        "*convertTo/*",
        "(1920x1080, 32FC1, 32FC1, 1, 1,     4.567)",
    ),
    Spec(
        "Scale_float",
        "opencv_perf_core",
        "*convertTo/*",
        "(1920x1080, 32FC1, 32FC1, 1, 1.234, 4.567)",
    ),
    Spec(
        "Scale_u8_f16_1.0",
        "opencv_perf_core",
        "*convertTo/*",
        "(1920x1080, 8UC1,  16FC1, 1, 1,     4.567)",
    ),
    Spec(
        "Scale_u8_f16",
        "opencv_perf_core",
        "*convertTo/*",
        "(1920x1080, 8UC1,  16FC1, 1, 1.234, 4.567)",
    ),
    Spec("MinMax_S8", "opencv_perf_core", "*minMaxVals/*", "(1920x1080, 8SC1)"),
    Spec("MinMax_U8", "opencv_perf_core", "*minMaxVals/*", "(1920x1080, 8UC1)"),
    Spec("MinMax_S16", "opencv_perf_core", "*minMaxVals/*", "(1920x1080, 16SC1)"),
    Spec("MinMax_U16", "opencv_perf_core", "*minMaxVals/*", "(1920x1080, 16UC1)"),
    Spec("MinMax_S32", "opencv_perf_core", "*minMaxVals/*", "(1920x1080, 32SC1)"),
    Spec("MinMax_F32", "opencv_perf_core", "*minMaxVals/*", "(1920x1080, 32FC1)"),
    Spec("MinMaxLoc_U8", "opencv_perf_core", "*minMaxLoc/*", "(1920x1080, 8UC1)"),
    Spec("Sum_F32", "opencv_perf_core", "*sum/*", "(1920x1080, 32FC1)"),
    Spec(
        "FloatToInt",
        "opencv_perf_core",
        "*convertTo/*",
        "(1920x1080, 32FC1, 8SC1,  1, 1, 0)",
    ),
    Spec(
        "FloatToUint",
        "opencv_perf_core",
        "*convertTo/*",
        "(1920x1080, 32FC1, 8UC1,  1, 1, 0)",
    ),
    Spec(
        "IntToFloat",
        "opencv_perf_core",
        "*convertTo/*",
        "(1920x1080, 8SC1,  32FC1, 1, 1, 0)",
    ),
    Spec(
        "UintToFloat",
        "opencv_perf_core",
        "*convertTo/*",
        "(1920x1080, 8UC1,  32FC1, 1, 1, 0)",
    ),
    Spec("CompareGt", "opencv_perf_core", "*compare/*", "(1920x1080, 8UC1, CMP_GT)"),
    Spec(
        "InRange_U8", "opencv_perf_core", "*inRangeScalar/*", "(1920x1080,  8UC1, 1, 2)"
    ),
    Spec(
        "InRange_F32",
        "opencv_perf_core",
        "*inRangeScalar/*",
        "(1920x1080, 32FC1, 1, 2)",
    ),
    Spec(
        "Remap_S16_U8_Replicate",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080,  8UC1, 16SC2, INTER_NEAREST, BORDER_REPLICATE)",
    ),
    Spec(
        "Remap_S16_U8_Constant",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080,  8UC1, 16SC2, INTER_NEAREST, BORDER_CONSTANT)",
    ),
    Spec(
        "Remap_S16_U16_Replicate",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080, 16UC1, 16SC2, INTER_NEAREST, BORDER_REPLICATE)",
    ),
    Spec(
        "Remap_S16_U16_Constant",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080, 16UC1, 16SC2, INTER_NEAREST, BORDER_CONSTANT)",
    ),
    Spec(
        "Remap_S16Point5_U8_Replicate",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080,  8UC1, 16SC2, INTER_LINEAR,  BORDER_REPLICATE)",
    ),
    Spec(
        "Remap_S16Point5_U8_Replicate_4ch",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080,  8UC4, 16SC2, INTER_LINEAR,  BORDER_REPLICATE)",
    ),
    Spec(
        "Remap_S16Point5_U8_Constant",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080,  8UC1, 16SC2, INTER_LINEAR,  BORDER_CONSTANT)",
    ),
    Spec(
        "Remap_S16Point5_U8_Constant_4ch",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080,  8UC4, 16SC2, INTER_LINEAR,  BORDER_CONSTANT)",
    ),
    Spec(
        "Remap_S16Point5_U16_Replicate",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080, 16UC1, 16SC2, INTER_LINEAR,  BORDER_REPLICATE)",
    ),
    Spec(
        "Remap_S16Point5_U16_Replicate_4ch",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080, 16UC4, 16SC2, INTER_LINEAR,  BORDER_REPLICATE)",
    ),
    Spec(
        "Remap_S16Point5_U16_Constant",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080, 16UC1, 16SC2, INTER_LINEAR,  BORDER_CONSTANT)",
    ),
    Spec(
        "Remap_S16Point5_U16_Constant_4ch",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080, 16UC4, 16SC2, INTER_LINEAR,  BORDER_CONSTANT)",
    ),
    Spec(
        "Remap_F32_U8_Replicate_Nearest",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080,  8UC1, 32FC1, INTER_NEAREST, BORDER_REPLICATE)",
    ),
    Spec(
        "Remap_F32_U8_Constant_Nearest",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080,  8UC1, 32FC1, INTER_NEAREST, BORDER_CONSTANT)",
    ),
    Spec(
        "Remap_F32_U16_Replicate_Nearest",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080, 16UC1, 32FC1, INTER_NEAREST, BORDER_REPLICATE)",
    ),
    Spec(
        "Remap_F32_U16_Constant_Nearest",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080, 16UC1, 32FC1, INTER_NEAREST, BORDER_CONSTANT)",
    ),
    Spec(
        "Remap_F32_U8_Replicate_Nearest_2ch",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080,  8UC2, 32FC1, INTER_NEAREST, BORDER_REPLICATE)",
    ),
    Spec(
        "Remap_F32_U8_Constant_Nearest_2ch",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080,  8UC2, 32FC1, INTER_NEAREST, BORDER_CONSTANT)",
    ),
    Spec(
        "Remap_F32_U16_Replicate_Nearest_2ch",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080, 16UC2, 32FC1, INTER_NEAREST, BORDER_REPLICATE)",
    ),
    Spec(
        "Remap_F32_U16_Constant_Nearest_2ch",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080, 16UC2, 32FC1, INTER_NEAREST, BORDER_CONSTANT)",
    ),
    Spec(
        "Remap_F32_U8_Replicate_Linear",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080,  8UC1, 32FC1, INTER_LINEAR,  BORDER_REPLICATE)",
    ),
    Spec(
        "Remap_F32_U8_Constant_Linear",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080,  8UC1, 32FC1, INTER_LINEAR,  BORDER_CONSTANT)",
    ),
    Spec(
        "Remap_F32_U16_Replicate_Linear",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080, 16UC1, 32FC1, INTER_LINEAR,  BORDER_REPLICATE)",
    ),
    Spec(
        "Remap_F32_U16_Constant_Linear",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080, 16UC1, 32FC1, INTER_LINEAR,  BORDER_CONSTANT)",
    ),
    Spec(
        "Remap_F32_U8_Replicate_Linear_2ch",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080,  8UC2, 32FC1, INTER_LINEAR,  BORDER_REPLICATE)",
    ),
    Spec(
        "Remap_F32_U8_Constant_Linear_2ch",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080,  8UC2, 32FC1, INTER_LINEAR,  BORDER_CONSTANT)",
    ),
    Spec(
        "Remap_F32_U16_Replicate_Linear_2ch",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080, 16UC2, 32FC1, INTER_LINEAR,  BORDER_REPLICATE)",
    ),
    Spec(
        "Remap_F32_U16_Constant_Linear_2ch",
        "opencv_perf_imgproc",
        "*Remap/*",
        "(1920x1080, 16UC2, 32FC1, INTER_LINEAR,  BORDER_CONSTANT)",
    ),
    Spec(
        "WarpPerspective_Nearest",
        "opencv_perf_imgproc",
        "*WarpPerspective/*",
        "(1920x1080, INTER_NEAREST, BORDER_REPLICATE, 1)",
    ),
    Spec(
        "WarpPerspective_Linear",
        "opencv_perf_imgproc",
        "*WarpPerspective/*",
        "(1920x1080, INTER_LINEAR,  BORDER_REPLICATE, 1)",
    ),
    Spec(
        "WarpPerspective_Nearest_Constant",
        "opencv_perf_imgproc",
        "*WarpPerspective/*",
        "(1920x1080, INTER_NEAREST, BORDER_CONSTANT, 1)",
    ),
    Spec(
        "WarpPerspective_Linear_Constant",
        "opencv_perf_imgproc",
        "*WarpPerspective/*",
        "(1920x1080, INTER_LINEAR,  BORDER_CONSTANT, 1)",
    ),
    Spec(
        "WarpPerspectiveNear_Nearest",
        "opencv_perf_imgproc",
        "*WarpPerspectiveNear/*",
        "(1920x1080, INTER_NEAREST, BORDER_REPLICATE, 8UC1)",
    ),
    Spec(
        "WarpPerspectiveNear_Linear",
        "opencv_perf_imgproc",
        "*WarpPerspectiveNear/*",
        "(1920x1080, INTER_LINEAR,  BORDER_REPLICATE, 8UC1)",
    ),
    Spec(
        "WarpPerspectiveNear_Nearest_Constant",
        "opencv_perf_imgproc",
        "*WarpPerspectiveNear/*",
        "(1920x1080, INTER_NEAREST, BORDER_CONSTANT, 8UC1)",
    ),
    Spec(
        "WarpPerspectiveNear_Linear_Constant",
        "opencv_perf_imgproc",
        "*WarpPerspectiveNear/*",
        "(1920x1080, INTER_LINEAR,  BORDER_CONSTANT, 8UC1)",
    ),
    Spec(
        "BlurAndDownsample_8UC1",
        "opencv_perf_imgproc",
        "*pyrDown/*",
        "(1920x1080, 8UC1)",
    ),
    Spec(
        "BlurAndDownsample_8UC2",
        "opencv_perf_imgproc",
        "*pyrDown/*",
        "(1920x1080, 8UC2)",
    ),
    Spec(
        "BlurAndDownsample_8UC3",
        "opencv_perf_imgproc",
        "*pyrDown/*",
        "(1920x1080, 8UC3)",
    ),
    Spec(
        "BlurAndDownsample_8UC4",
        "opencv_perf_imgproc",
        "*pyrDown/*",
        "(1920x1080, 8UC4)",
    ),
    Spec(
        "ScharrInterleaved_8UC1",
        "opencv_perf_video",
        "*calcScharrDeriv/*",
        "(1920x1080, 8UC1)",
    ),
    Spec(
        "ScharrInterleaved_8UC2",
        "opencv_perf_video",
        "*calcScharrDeriv/*",
        "(1920x1080, 8UC2)",
    ),
    Spec(
        "ScharrInterleaved_8UC3",
        "opencv_perf_video",
        "*calcScharrDeriv/*",
        "(1920x1080, 8UC3)",
    ),
    Spec(
        "ScharrInterleaved_8UC4",
        "opencv_perf_video",
        "*calcScharrDeriv/*",
        "(1920x1080, 8UC4)",
    ),
    Spec(
        "Rotate_90_CLOCKWISE_8UC1",
        "opencv_perf_core",
        "*rotate/*",
        "(1920x1080, 0, 8UC1)",
    ),
    Spec(
        "Rotate_90_CLOCKWISE_16SC1",
        "opencv_perf_core",
        "*rotate/*",
        "(1920x1080, 0, 16SC1)",
    ),
    Spec(
        "Rotate_90_CLOCKWISE_8UC3",
        "opencv_perf_core",
        "*rotate/*",
        "(1920x1080, 0, 8UC3)",
    ),
    Spec(
        "Rotate_90_CLOCKWISE_16SC3",
        "opencv_perf_core",
        "*rotate/*",
        "(1920x1080, 0, 16SC3)",
    ),
    Spec(
        "Rotate_90_CLOCKWISE_8UC4",
        "opencv_perf_core",
        "*rotate/*",
        "(1920x1080, 0, 8UC4)",
    ),
    Spec(
        "Rotate_90_CLOCKWISE_16SC4",
        "opencv_perf_core",
        "*rotate/*",
        "(1920x1080, 0, 16SC4)",
    ),
    Spec("Rotate_90_CCW_8UC1", "opencv_perf_core", "*rotate/*", "(1920x1080, 2, 8UC1)"),
    Spec(
        "Rotate_90_CCW_16SC1", "opencv_perf_core", "*rotate/*", "(1920x1080, 2, 16SC1)"
    ),
    Spec("Rotate_90_CCW_8UC3", "opencv_perf_core", "*rotate/*", "(1920x1080, 2, 8UC3)"),
    Spec(
        "Rotate_90_CCW_16SC3", "opencv_perf_core", "*rotate/*", "(1920x1080, 2, 16SC3)"
    ),
    Spec("Rotate_90_CCW_8UC4", "opencv_perf_core", "*rotate/*", "(1920x1080, 2, 8UC4)"),
    Spec(
        "Rotate_90_CCW_16SC4", "opencv_perf_core", "*rotate/*", "(1920x1080, 2, 16SC4)"
    ),
    Spec("Transpose_8UC1", "opencv_perf_core", "*transpose2d/*", "(1920x1080, 8UC1)"),
    Spec("Transpose_16SC1", "opencv_perf_core", "*transpose2d/*", "(1920x1080, 16SC1)"),
    Spec("Transpose_8UC3", "opencv_perf_core", "*transpose2d/*", "(1920x1080, 8UC3)"),
    Spec("Transpose_16SC3", "opencv_perf_core", "*transpose2d/*", "(1920x1080, 16SC3)"),
    Spec("Transpose_8UC4", "opencv_perf_core", "*transpose2d/*", "(1920x1080, 8UC4)"),
    Spec("Transpose_16SC4", "opencv_perf_core", "*transpose2d/*", "(1920x1080, 16SC4)"),
    Spec(
        "Resize2x2_8b",
        "opencv_perf_imgproc",
        "*resizeUpLinearNonExact/*",
        "(8UC1,  (960x540, 1920x1080))",
    ),
    Spec(
        "Resize2x2_float",
        "opencv_perf_imgproc",
        "*resizeUpLinearNonExact/*",
        "(32FC1, (960x540, 1920x1080))",
    ),
    Spec(
        "Resize4x4_8b",
        "opencv_perf_imgproc",
        "*resizeUpLinearNonExact/*",
        "(8UC1,  (480x270, 1920x1080))",
    ),
    Spec(
        "Resize4x4_float",
        "opencv_perf_imgproc",
        "*resizeUpLinearNonExact/*",
        "(32FC1, (480x270, 1920x1080))",
    ),
    Spec(
        "Resize8x8_float",
        "opencv_perf_imgproc",
        "*resizeUpLinearNonExact/*",
        "(32FC1, (240x135, 1920x1080))",
    ),
    Spec(
        "ResizeDown2_8b",
        "opencv_perf_imgproc",
        "*resizeDownLinearNonExact/*",
        "(8UC1, (1920x1080, 1280x720))",
    ),
    Spec(
        "ResizeDown2_8b_2ch",
        "opencv_perf_imgproc",
        "*resizeDownLinearNonExact/*",
        "(8UC2, (1920x1080, 1280x720))",
    ),
    Spec(
        "ResizeDown2_8b_3ch",
        "opencv_perf_imgproc",
        "*resizeDownLinearNonExact/*",
        "(8UC3, (1920x1080, 1280x720))",
    ),
    Spec(
        "ResizeDown3_8b",
        "opencv_perf_imgproc",
        "*resizeDownLinearNonExact/*",
        "(8UC1, (1920x1080, 640x480))",
    ),
    Spec(
        "ResizeDown3_8b_2ch",
        "opencv_perf_imgproc",
        "*resizeDownLinearNonExact/*",
        "(8UC2, (1920x1080, 640x480))",
    ),
    Spec(
        "ResizeDown3_8b_3ch",
        "opencv_perf_imgproc",
        "*resizeDownLinearNonExact/*",
        "(8UC3, (1920x1080, 640x480))",
    ),
    # 720p, not FHD -- creating FHD optical-flow frames is non-trivial.
    # Disabled: these require opencv_extra test data (cv/optflow/frames/720p_*.png)
    # to be available via $OPENCV_TEST_DATA_PATH. Re-enable once the data is in place.
    # Spec('OpticalFlowPyrLK_inc_build_pyramid_720p', 'opencv_perf_video', '*OpticalFlowPyrLK_full/*', '("cv/optflow/frames/720p_%02d.png", 1, 1, (15, 15), 11)'),
    # Spec('OpticalFlowPyrLK_exc_build_pyramid_720p', 'opencv_perf_video', '*OpticalFlowPyrLK_self/*', '("cv/optflow/frames/720p_%02d.png", 1, 1, (15, 15), 11, true)'),
]


def find_binary(build_dir: Path, binary: str) -> Path:
    for c in (build_dir / "bin" / binary, build_dir / binary):
        if c.is_file() and os.access(c, os.X_OK):
            return c
    matches = [
        m for m in build_dir.rglob(binary) if m.is_file() and os.access(m, os.X_OK)
    ]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find perf binary {binary!r} under {build_dir}")


def parse_json_result(name: str, json_path: Path) -> Result | None:
    try:
        data = json.loads(json_path.read_text())
    except Exception as exc:
        print(f"    [FAIL] cannot parse json: {exc}", file=sys.stderr)
        return None
    if data.get("tests", 0) != 1:
        print(f"    [FAIL] expected 1 test, got {data.get('tests')}", file=sys.stderr)
        return None
    mean = gstddev = None
    for suite in data.get("testsuites", []):
        for tc in suite.get("testsuite", []):
            if "mean" in tc:
                mean = float(tc["mean"])
            if "gstddev" in tc:
                gstddev = float(tc["gstddev"])
    if mean is None:
        print("    [FAIL] mean not present in JSON output", file=sys.stderr)
        return None
    return Result(name=name, mean=mean, gstddev=gstddev or 0.0)


def run_one(
    spec: Spec,
    build_dir: Path,
    min_samples: int,
    extra: list[str],
    dry_run: bool,
    threads: int | None = None,
) -> Result | None:
    with tempfile.NamedTemporaryFile(
        prefix="bench_", suffix=".json", delete=False
    ) as tmp:
        json_path = Path(tmp.name)
    try:
        cmd = build_command(
            spec,
            build_dir,
            min_samples,
            [f"--gtest_output=json:{json_path}", *extra],
            threads=threads,
        )
        print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}", file=sys.stderr)
        if dry_run:
            return None
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(
                f"    [FAIL] exit={proc.returncode}\n{proc.stderr.strip()}",
                file=sys.stderr,
            )
            return None
        return parse_json_result(spec.name, json_path)
    finally:
        if json_path.exists():
            json_path.unlink()


def write_tsv(path: Path, results: list[Result]) -> None:
    with path.open("w") as f:
        f.write("Operation\tmean\tgstddev\n")
        for r in results:
            f.write(f"{r.name}\t{r.mean:.9g}\t{r.gstddev:.9g}\n")


def read_tsv(path: Path) -> dict[str, Result]:
    out: dict[str, Result] = {}
    with path.open() as f:
        f.readline()  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            mean = float(parts[1])
            gstddev = float(parts[2]) if len(parts) > 2 else 0.0
            out[parts[0]] = Result(name=parts[0], mean=mean, gstddev=gstddev)
    return out


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    build_dir: Path = args.opencv_build.resolve()
    if not build_dir.is_dir():
        print(f"error: --opencv-build not a directory: {build_dir}", file=sys.stderr)
        return 2

    specs = list(CASES)
    if args.filter:
        rgx = re.compile(args.filter)
        specs = [s for s in specs if rgx.search(s.name)]
    if not specs:
        print("error: no benchmark cases selected", file=sys.stderr)
        return 1

    print(
        f"Running {len(specs)} benchmark(s) at FHD (1920x1080) against {build_dir}",
        file=sys.stderr,
    )

    extra = shlex.split(args.extra_args) if args.extra_args else []
    results: list[Result] = []
    for i, spec in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] {spec.name}", file=sys.stderr)
        try:
            r = run_one(
                spec,
                build_dir,
                args.min_samples,
                extra,
                args.dry_run,
                threads=args.threads,
            )
        except FileNotFoundError as exc:
            print(f"    [SKIP] {exc}", file=sys.stderr)
            continue
        if r is not None:
            results.append(r)
            print(f"    mean={r.mean:.6g}  gstddev={r.gstddev:.4g}", file=sys.stderr)

    if args.dry_run:
        return 0

    write_tsv(args.output, results)
    print(f"\nWrote {len(results)} results to {args.output}", file=sys.stderr)
    return 0


def build_command(
    spec: Spec,
    build_dir: Path | None,
    min_samples: int,
    extra: list[str],
    threads: int | None = None,
) -> list[str]:
    if build_dir is not None:
        binary = str(find_binary(build_dir, spec.binary))
    else:
        binary = f"<opencv-build>/bin/{spec.binary}"
    # Collapse runs of whitespace: the embedded param strings sometimes use
    # extra spaces for column alignment, but gtest test names use a single
    # space, so the exact-match filter would otherwise reject them.
    params = re.sub(r"\s+", " ", spec.gtest_params).strip()
    cmd = [
        binary,
        f"--perf_min_samples={min_samples}",
        f"--gtest_filter={spec.gtest_filter}",
        f"--gtest_param_filter={params}",
    ]
    if threads is not None:
        cmd.append(f"--perf_threads={threads}")
    cmd.extend(extra)
    return cmd


def cmd_show(args: argparse.Namespace) -> int:
    matches = [s for s in CASES if s.name == args.name]
    if not matches:
        rgx = re.compile(args.name, re.IGNORECASE)
        matches = [s for s in CASES if rgx.search(s.name)]
    if not matches:
        print(f"error: no benchmark case matches {args.name!r}", file=sys.stderr)
        candidates = [s.name for s in CASES if args.name.lower() in s.name.lower()]
        if candidates:
            print("did you mean one of:", file=sys.stderr)
            for c in candidates[:10]:
                print(f"  {c}", file=sys.stderr)
        return 1

    build_dir = args.opencv_build.resolve() if args.opencv_build else None
    extra = shlex.split(args.extra_args) if args.extra_args else []
    for spec in matches:
        try:
            cmd = build_command(
                spec, build_dir, args.min_samples, extra, threads=args.threads
            )
        except FileNotFoundError as exc:
            print(f"# {spec.name}: {exc}", file=sys.stderr)
            continue
        print(f"# {spec.name}")
        print(" ".join(shlex.quote(c) for c in cmd))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "error: matplotlib is required for `compare`. Install with: pip install matplotlib",
            file=sys.stderr,
        )
        return 2

    baseline = read_tsv(args.baseline)
    candidate = read_tsv(args.candidate)

    common = sorted(set(baseline) & set(candidate))
    only_base = sorted(set(baseline) - set(candidate))
    only_cand = sorted(set(candidate) - set(baseline))
    if only_base:
        print(
            f"note: {len(only_base)} entries only in baseline (ignored)",
            file=sys.stderr,
        )
    if only_cand:
        print(
            f"note: {len(only_cand)} entries only in candidate (ignored)",
            file=sys.stderr,
        )
    if not common:
        print("error: no common operations between the two files", file=sys.stderr)
        return 1

    rows: list[tuple[str, float, float, float]] = []
    for name in common:
        b = baseline[name].mean
        c = candidate[name].mean
        if b <= 0 or c <= 0:
            continue
        rows.append((name, b, c, b / c))  # >1 means candidate is faster

    rows.sort(key=lambda r: r[3])
    if args.top:
        rows = rows[: args.top] + rows[-args.top :]

    summary_path = args.output.with_suffix(".tsv")
    with summary_path.open("w") as f:
        f.write("Operation\tbaseline_mean\tcandidate_mean\tspeedup\n")
        for name, b, c, s in rows:
            f.write(f"{name}\t{b:.9g}\t{c:.9g}\t{s:.6f}\n")
    print(f"Wrote summary {summary_path}", file=sys.stderr)

    names = [r[0] for r in rows]
    speedups = [r[3] for r in rows]
    colors = ["#2a9d8f" if s >= 1.0 else "#e76f51" for s in speedups]

    fig_h = max(2.0, 0.22 * len(rows) + 0.8)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = list(range(len(rows)))
    ax.barh(y, speedups, color=colors)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(
        f"Speed-up (baseline mean / candidate mean)\n"
        f"baseline = {args.baseline.name}    candidate = {args.candidate.name}"
    )
    ax.set_title(args.title or "Benchmark comparison")
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i, s in enumerate(speedups):
        ax.text(s, i, f" {s:.2f}x", va="center", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote chart   {args.output}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone OpenCV / KleidiCV perf benchmark driver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("run", help="Run embedded benchmarks and write a TSV.")
    pr.add_argument(
        "--opencv-build",
        type=Path,
        required=True,
        help="Path to the OpenCV build directory containing the perf binaries.",
    )
    pr.add_argument(
        "--filter", default=None, help="Regex applied to case name to select a subset."
    )
    pr.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="--perf_min_samples value. Default: 200.",
    )
    pr.add_argument(
        "--threads",
        type=int,
        default=None,
        help="If set, pass --perf_threads=N to each perf binary (0 = all cores, 1 = single-threaded).",
    )
    pr.add_argument(
        "--extra-args",
        default="",
        help="Extra args appended to every perf binary invocation (quoted shell string).",
    )
    pr.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.tsv"),
        help="Output TSV path.",
    )
    pr.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing."
    )
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser(
        "compare", help="Compare two TSV result files and render a chart PNG."
    )
    pc.add_argument("--baseline", type=Path, required=True, help="Baseline TSV.")
    pc.add_argument("--candidate", type=Path, required=True, help="Candidate TSV.")
    pc.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_compare.png"),
        help="Output chart PNG.",
    )
    pc.add_argument(
        "--top",
        type=int,
        default=0,
        help="If >0, plot only N slowest and N fastest changes.",
    )
    pc.add_argument("--title", default=None, help="Chart title.")
    pc.add_argument("--dpi", type=int, default=120, help="Chart DPI. Default: 120.")
    pc.set_defaults(func=cmd_compare)

    ps = sub.add_parser(
        "show",
        help="Print the perf binary command for a given test case name (for debugging).",
    )
    ps.add_argument(
        "name",
        help="Test case name (exact match preferred, falls back to case-insensitive regex).",
    )
    ps.add_argument(
        "--opencv-build",
        type=Path,
        default=None,
        help="Optional OpenCV build directory; if provided, the real binary path is resolved.",
    )
    ps.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="--perf_min_samples value. Default: 200.",
    )
    ps.add_argument(
        "--threads",
        type=int,
        default=None,
        help="If set, append --perf_threads=N to the command.",
    )
    ps.add_argument(
        "--extra-args",
        default="",
        help="Extra args appended to the command (quoted shell string).",
    )
    ps.set_defaults(func=cmd_show)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
