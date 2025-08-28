# src/services/viz/merge.py
from __future__ import annotations
import io, math
from typing import Iterable, Union
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes


try:
    from PIL import Image, ImageOps
except Exception as e:
    raise RuntimeError("viz.merge requires Pillow. pip install pillow") from e


# 支持的对象：matplotlib Figure、Axes、PNG 文件路径
MatObj = Union["matplotlib.figure.Figure", "matplotlib.axes.Axes", str]


def _to_image(obj: MatObj, dpi: int = 150, tight: bool = True) -> Image.Image:
    if isinstance(obj, Axes):
        fig = obj.figure
    elif isinstance(obj, Figure):
        fig = obj
    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight" if tight else None)
    buf.seek(0)
    im = Image.open(buf)
    return im.convert("RGBA") if im.mode != "RGBA" else im


def merge(objects: Iterable[MatObj], out_png: str | None = None, *, ncols: int = 2,
          pad: int = 10, bg: str = "white", dpi: int = 150, equalize: str = "height") -> Image.Image:
    """把多张图（Figure/Axes/PNG 路径）拼成一张面板图。


    - ncols: 每行列数；行数自动 = ceil(N/ncols)
    - pad: 砖块与画布的像素间距
    - bg: 背景颜色（"white" 或 "transparent"）
    - equalize: "height" | "width"，对齐各子图的高度或宽度
    - 返回 Pillow Image；如给出 out_png 则保存
    """
    imgs = [_to_image(o, dpi=dpi) for o in objects]
    if not imgs:
        raise ValueError("no images to merge")


    # 尺寸对齐
    if equalize == "height":
        h = max(im.height for im in imgs)
        imgs = [ImageOps.contain(im, (int(im.width * h / im.height), h)) if im.height != h else im for im in imgs]
    elif equalize == "width":
        w = max(im.width for im in imgs)
        imgs = [ImageOps.contain(im, (w, int(im.height * w / im.width))) if im.width != w else im for im in imgs]


    n = len(imgs)
    rows = math.ceil(n / ncols)
    colw = [0] * ncols
    rowh = [0] * rows
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, ncols)
        colw[c] = max(colw[c], im.width)
        rowh[r] = max(rowh[r], im.height)


    total_w = sum(colw) + pad * (ncols + 1)
    total_h = sum(rowh) + pad * (rows + 1)
    base = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 0) if bg == "transparent" else bg)


    y = pad
    for r in range(rows):
        x = pad
        for c in range(ncols):
            idx = r * ncols + c
            if idx >= n:
                break
            im = imgs[idx]
            dx = (colw[c] - im.width) // 2
            dy = (rowh[r] - im.height) // 2
            base.paste(im, (x + dx, y + dy), mask=im if im.mode == "RGBA" else None)
            x += colw[c] + pad
        y += rowh[r] + pad


    out = base if bg == "transparent" else base.convert("RGB")
    if out_png:
        out.save(out_png, dpi=(dpi, dpi))
    return out


# 便捷：仅合并 PNG 路径


def merge_paths(paths: Iterable[str], out_png: str, **kw) -> str:
    merge(paths, out_png=out_png, **kw)
    return out_png