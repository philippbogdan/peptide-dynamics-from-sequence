"""Draw a small molecule as shaded balls and sticks with PIL (no molecular viewer needed)."""
import numpy as np
from PIL import Image, ImageDraw

COLOUR = {"C": (88, 96, 105), "N": (48, 99, 214), "O": (220, 60, 50), "S": (226, 170, 30)}
RADIUS = {"C": 0.36, "N": 0.35, "O": 0.34, "S": 0.44}        # angstrom, drawn size of the ball
BACKGROUND = (255, 255, 255)


def superpose(X, ref, sel):
    """Rotate and translate frames X (F, n, 3) onto ref (n, 3) using the atoms sel (Kabsch)."""
    A = X[:, sel] - X[:, sel].mean(1, keepdims=True)
    B = ref[sel] - ref[sel].mean(0)
    U, _, Vt = np.linalg.svd(np.einsum("fai,aj->fij", A, B))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.broadcast_to(np.eye(3), U.shape).copy()
    D[:, 2, 2] = d
    R = U @ D @ Vt
    return (X - X[:, sel].mean(1, keepdims=True)) @ R + ref[sel].mean(0)


def view_axes(X):
    """Rows: the two widest principal axes of a set of points and the viewing direction."""
    P = X.reshape(-1, 3) - X.reshape(-1, 3).mean(0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    return Vt


def mix(c, other, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c, other))


def draw(x, elements, bonds, axes, centre, scale, size, ss=3, fog=0.45, colours=None, bond_width=0.17):
    """One frame x (n, 3) as an RGB image of `size` (w, h) pixels. scale: pixels per angstrom."""
    W, H = size[0] * ss, size[1] * ss
    img = Image.new("RGB", (W, H), BACKGROUND)
    d = ImageDraw.Draw(img)
    p = (x - centre) @ axes.T
    px = W / 2 + p[:, 0] * scale * ss
    py = H / 2 - p[:, 1] * scale * ss
    z = p[:, 2]
    zlo, zhi = z.min() - 1e-6, z.max() + 1e-6
    depth = (z - zlo) / (zhi - zlo)                     # 0 far, 1 near
    col = [(colours[i] if colours else COLOUR[e]) for i, e in enumerate(elements)]
    items = []
    for i, e in enumerate(elements):
        items.append((z[i], "atom", i, None))
    for a, b in bonds:
        mid = 0.5 * (x[a] + x[b])
        for i, j in ((a, b), (b, a)):                   # each half takes the colour of its atom
            items.append((0.5 * (z[i] + ((mid - centre) @ axes.T)[2]) - 0.01, "bond", i, j))
    outline = (30, 34, 40)
    for zz, kind, i, j in sorted(items, key=lambda t: t[0]):
        t = fog * (1 - depth[i])
        c = mix(col[i], BACKGROUND, t)
        o = mix(outline, BACKGROUND, t)
        if kind == "bond":
            mx, my = 0.5 * (px[i] + px[j]), 0.5 * (py[i] + py[j])
            w = bond_width * scale * ss
            d.line([(px[i], py[i]), (mx, my)], fill=o, width=int(round(w + 2.2 * ss)))
            d.line([(px[i], py[i]), (mx, my)], fill=c, width=int(round(w)))
        else:
            r = RADIUS[elements[i]] * scale * ss * (0.9 + 0.2 * depth[i])
            d.ellipse([px[i] - r - 1.1 * ss, py[i] - r - 1.1 * ss, px[i] + r + 1.1 * ss, py[i] + r + 1.1 * ss], fill=o)
            d.ellipse([px[i] - r, py[i] - r, px[i] + r, py[i] + r], fill=c)
            hl = mix(c, (255, 255, 255), 0.38)
            rr = 0.55 * r
            d.ellipse([px[i] - 0.25 * r - rr, py[i] - 0.3 * r - rr, px[i] - 0.25 * r + rr, py[i] - 0.3 * r + rr], fill=hl)
            hl2 = mix(c, (255, 255, 255), 0.7)
            rr = 0.2 * r
            d.ellipse([px[i] - 0.38 * r - rr, py[i] - 0.42 * r - rr, px[i] - 0.38 * r + rr, py[i] - 0.42 * r + rr], fill=hl2)
    return img.resize(size, Image.LANCZOS)
