"""Generate a synthetic street scene with German location clues, for pipeline testing."""
from PIL import Image, ImageDraw, ImageFont


def font(size, bold=True):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build(path="test_street.jpg"):
    W, H = 1280, 900
    img = Image.new("RGB", (W, H), (135, 180, 222))  # sky
    d = ImageDraw.Draw(img)

    # ground / road
    d.rectangle([0, 620, W, H], fill=(90, 92, 96))
    # sidewalk
    d.rectangle([0, 560, W, 620], fill=(170, 168, 165))
    # road centre dashes (white)
    for x in range(40, W, 160):
        d.rectangle([x, 760, x + 90, 778], fill=(240, 240, 240))

    # a row of European-style buildings
    cols = [(196, 142, 110), (210, 196, 160), (170, 120, 120), (200, 180, 150)]
    bx = 0
    i = 0
    while bx < W:
        bw = 210
        bh = 300 + (i % 3) * 40
        d.rectangle([bx, 560 - bh, bx + bw - 8, 560], fill=cols[i % len(cols)], outline=(60, 50, 45))
        # windows
        for wy in range(560 - bh + 30, 540, 70):
            for wx in range(bx + 20, bx + bw - 40, 60):
                d.rectangle([wx, wy, wx + 35, wy + 45], fill=(120, 150, 170), outline=(40, 40, 40))
        # red pitched roof
        d.polygon([(bx, 560 - bh), (bx + bw - 8, 560 - bh), (bx + bw // 2 - 4, 560 - bh - 50)],
                  fill=(150, 60, 50))
        bx += bw
        i += 1

    # shop awning + name (German)
    d.rectangle([300, 470, 700, 520], fill=(30, 70, 140))
    d.text((320, 478), "BÄCKEREI MÜLLER", font=font(34), fill="white")

    # green street sign
    d.rectangle([60, 360, 360, 415], fill=(0, 110, 60), outline="white", width=3)
    d.text((76, 370), "Hauptstraße", font=font(34), fill="white")

    # round speed-limit sign (red ring, white field) — European style
    d.ellipse([900, 360, 985, 445], fill="white", outline=(200, 30, 30), width=9)
    d.text((916, 378), "50", font=font(40), fill="black")

    # parked car on the RIGHT with a German (Munich) plate
    d.rounded_rectangle([880, 660, 1120, 770], radius=20, fill=(40, 60, 110))
    d.rectangle([905, 700, 905 + 150, 700 + 34], fill="white", outline="black", width=2)
    d.rectangle([905, 700, 925, 734], fill=(0, 51, 153))  # EU blue strip
    d.text((908, 706), "D", font=font(20), fill="white")
    d.text((932, 704), "M·AB 1234", font=font(24), fill="black")

    # German flag
    d.rectangle([770, 360, 850, 384], fill=(0, 0, 0))
    d.rectangle([770, 384, 850, 408], fill=(221, 0, 0))
    d.rectangle([770, 408, 850, 432], fill=(255, 206, 0))

    # doorway on the first building with a large enamel house-number plate "12"
    d.rectangle([40, 400, 130, 560], fill=(70, 45, 35), outline=(30, 20, 15), width=3)
    d.rounded_rectangle([150, 250, 290, 360], radius=8, fill=(0, 60, 150), outline="white", width=5)
    d.text((178, 268), "12", font=font(80), fill="white")

    img.save(path, quality=90)
    print("saved", path, img.size)
    return path


if __name__ == "__main__":
    build()
