# coding: utf-8

import os, sys
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np


def draw_font(bounds, commands, out_path):
    # SVG d commands:
    # MoveTo: M(x, y), m(dx, dy)
    # LineTo: L(x, y), l(dx, dy), H(x), h(dx), V(y), v(dy)
    # Quadratic Bézier Curve: Q(x1, y1, x, y), q(dx1, dy1, dx, dy), T(x, y), t(dx, dy)
    # Cubic Bézier Curve: C(x1, y1, x2, y2, x, y), c(dx1, dy1, dx2, dy2, dx, dy), S(x2, y2, x, y), s(dx2, dy2, dx, dy)
    # Elliptical Arc Curve: A(rx, ry, angle, large-arc-flag, sweep-flag, x, y), a(rx, ry, angle, large-arc-flag, sweep-flag, dx, dy)
    # ClosePath: Z, z
    codes = []
    verts = []
    start = (0, 0)
    prev = (0, 0)
    for command in commands:
        c = command['code']
        v = command['vert']
        if c == 'M':
            codes.append(Path.MOVETO)
            verts.append((v[0], v[1]))
            start = (v[0], v[1])
            prev = (v[0], v[1])
        elif c == 'm':
            codes.append(Path.MOVETO)
            verts.append((prev[0] + v[0], prev[1] + v[1]))
            start = (prev[0] + v[0], prev[1] + v[1])
            prev = (prev[0] + v[0], prev[1] + v[1])
        elif c == 'L' or c == ' ':
            codes.append(Path.LINETO)
            verts.append((v[0], v[1]))
            prev = (v[0], v[1])
        elif c == 'l':
            codes.append(Path.LINETO)
            verts.append((prev[0] + v[0], prev[1] + v[1]))
            prev = (prev[0] + v[0], prev[1] + v[1])
        elif c == 'H':
            codes.append(Path.LINETO)
            verts.append((v[0], prev[1]))
            prev = (v[0], prev[1])
        elif c == 'h':
            codes.append(Path.LINETO)
            verts.append((prev[0] + v[0], prev[1]))
            prev = (prev[0] + v[0], prev[1])
        elif c == 'V':
            codes.append(Path.LINETO)
            verts.append((prev[0], v[0]))
            prev = (prev[0], v[0])
        elif c == 'v':
            codes.append(Path.LINETO)
            verts.append((prev[0], prev[1] + v[0]))
            prev = (prev[0], prev[1] + v[0])
        elif c == 'Q':
            codes.append(Path.CURVE3)
            verts.append((v[0], v[1]))
            codes.append(Path.CURVE3)
            verts.append((v[2], v[3]))
            prev = (v[2], v[3])
        elif c == 'q':
            codes.append(Path.CURVE3)
            verts.append((prev[0] + v[0], prev[1] + v[1]))
            prev = (prev[0] + v[0], prev[1] + v[1])
            codes.append(Path.CURVE3)
            verts.append((prev[0] + v[2], prev[1] + v[3]))
            prev = (prev[0] + v[2], prev[1] + v[3])
        elif c == 'T':
            codes.append(Path.CURVE3)
            verts.append((v[0], v[1]))
            prev = (v[2], v[3])
        elif c == 't':
            codes.append(Path.CURVE3)
            verts.append((prev[0] + v[0], prev[1] + v[1]))
            prev = (prev[0] + v[0], prev[1] + v[1])
        elif c == 'C':
            codes.append(Path.CURVE4)
            verts.append((v[0], v[1]))
            codes.append(Path.CURVE4)
            verts.append((v[2], v[3]))
            codes.append(Path.CURVE4)
            verts.append((v[4], v[5]))
            prev = (v[4], v[5])
        elif c == 'c':
            codes.append(Path.CURVE4)
            verts.append((prev[0] + v[0], prev[1] + v[1]))
            prev = (prev[0] + v[0], prev[1] + v[1])
            codes.append(Path.CURVE4)
            verts.append((prev[0] + v[2], prev[1] + v[3]))
            prev = (prev[0] + v[2], prev[1] + v[3])
            codes.append(Path.CURVE4)
            verts.append((prev[0] + v[4], prev[1] + v[5]))
            prev = (prev[0] + v[4], prev[1] + v[5])
        elif c == 'S':
            codes.append(Path.CURVE4)
            verts.append((v[0], v[1]))
            codes.append(Path.CURVE4)
            verts.append((v[2], v[3]))
            prev = (v[2], v[3])
        elif c == 's':
            codes.append(Path.CURVE4)
            verts.append((prev[0] + v[0], prev[1] + v[1]))
            prev = (prev[0] + v[0], prev[1] + v[1])
            codes.append(Path.CURVE4)
            verts.append((prev[0] + v[2], prev[1] + v[3]))
            prev = (prev[0] + v[2], prev[1] + v[3])
        elif c == 'A':
            # 先用直线代替椭圆弧
            codes.append(Path.LINETO)
            verts.append((v[0], v[1]))
            prev = (v[0], v[1])
        elif c == 'a':
            # 先用直线代替椭圆弧
            codes.append(Path.LINETO)
            verts.append((prev[0] + v[0], prev[1] + v[1]))
            prev = (prev[0] + v[0], prev[1] + v[1])
        elif c == 'Z' or c == 'z':
            codes.append(Path.CLOSEPOLY)
            verts.append(start)
            prev = start
        else:
            print('Unrecognized command:', command)
    fig, ax = plt.subplots()
    fig.set_size_inches(1.0 / 1, 1.0 / 1)
    verts_np = np.array(verts)
    content_width = content_height = np.max(
        np.max(verts_np, axis=0) - np.min(verts_np, axis=0))
    bound_width = bounds[2] - bounds[0]
    bound_height = bounds[3] - bounds[1]
    delta_width = (bound_width - content_width)
    delta_height = (bound_height - content_height)
    ax.set_xlim(bounds[0] + delta_width / 3, bounds[2] - delta_width / 4)
    ax.set_ylim(bounds[1] + delta_height / 2, bounds[3] - delta_height / 6)
    ax.set_aspect(1)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='black', edgecolor='black', lw=0)
    ax.add_patch(patch)
    plt.axis('off')
    try:
        fig.savefig(out_path,
                    format='png',
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
    except:
        pass
    plt.close()


def parse_font(ttf_path):
    tt = TTFont(ttf_path)
    glyphset = tt.getGlyphSet()
    bounds = (tt['head'].xMin, tt['head'].yMin, tt['head'].xMax,
              tt['head'].yMax)
    for unicode, name in tt['cmap'].tables[0].cmap.items():
        glyph = glyphset[name]
        pen = SVGPathPen(glyphset)
        glyph.draw(pen)
        if len(pen._commands) <= 0:
            continue
        commands = []
        for command in pen._commands:
            commands.append({
                'code':
                command[0],
                'vert':
                list(
                    map(lambda n: float(n), command[1:].replace(
                        ',', '').split(' '))) if command[1:] != '' else [],
            })
        font = {
            'unicode': unicode,
            'commands': commands,
        }
        yield bounds, font


def main():
    if len(sys.argv) <= 1:
        print('Usage: ttf_extract.py <ttf_path>')

    ttf_path = sys.argv[1]
    print('Extracting %s...' % ttf_path)
    out_dir = os.path.splitext(ttf_path)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for bounds, font in parse_font(ttf_path):
        if font['unicode'] < 0x4E00 or font['unicode'] > 0x9FFF:
            continue
        print('Drawing %s...' % chr(font['unicode']))
        draw_font(bounds, font['commands'],
                  os.path.join(out_dir,
                               chr(font['unicode']) + '.png'))


if __name__ == '__main__':
    main()