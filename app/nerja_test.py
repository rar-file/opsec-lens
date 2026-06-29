"""Synthetic Costa del Sol roundabout scene with Nerja anchors, to test triangulation."""
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build(path="nerja_street.jpg"):
    W, H = 1280, 900
    img = Image.new("RGB", (W, H), (120, 180, 235))  # bright southern sky
    d = ImageDraw.Draw(img)

    # whitewashed Andalusian buildings with terracotta roofs
    bx = 0
    while bx < W:
        bw = 230
        bh = 230
        d.rectangle([bx, 300 - bh + 200, bx + bw - 10, 500], fill=(245, 243, 238), outline=(180, 175, 165))
        d.polygon([(bx, 300), (bx + bw - 10, 300), (bx + bw // 2, 255)], fill=(190, 95, 70))
        for wy in range(330, 470, 60):
            for wx in range(bx + 25, bx + bw - 50, 70):
                d.rectangle([wx, wy, wx + 38, wy + 40], fill=(120, 150, 175), outline=(80, 80, 80))
        bx += bw

    # ground + roundabout
    d.rectangle([0, 500, W, H], fill=(95, 97, 100))
    d.ellipse([360, 560, 920, 860], outline=(240, 240, 240), width=10)        # roundabout ring
    d.ellipse([520, 620, 760, 800], fill=(120, 160, 90))                      # central island (grass)
    # central monument (obelisk)
    d.polygon([(632, 640), (648, 640), (642, 720)], fill=(210, 205, 195), outline=(120, 120, 120))
    d.rectangle([628, 718, 652, 736], fill=(160, 155, 145))

    # palm trees (Costa del Sol vibe)
    for px in (180, 1080):
        d.rectangle([px, 430, px + 16, 540], fill=(110, 80, 50))
        for ang in (-40, -15, 15, 40):
            d.line([(px + 8, 435), (px + 8 + ang * 2, 405)], fill=(40, 130, 60), width=8)

    # blue directional road sign with place names + distance
    d.rectangle([60, 330, 360, 470], fill=(0, 70, 160), outline="white", width=5)
    d.text((80, 345), "Nerja", font=font(40), fill="white")
    d.polygon([(300, 360), (340, 378), (300, 396)], fill="white")  # arrow
    d.text((80, 405), "Málaga", font=font(34), fill="white")
    d.text((265, 408), "52", font=font(34), fill="white")

    # a real Nerja landmark business name on an awning
    d.rectangle([840, 360, 1240, 410], fill=(150, 30, 40))
    d.text((858, 368), "HOTEL BALCÓN DE EUROPA", font=font(26), fill="white")

    # Spanish license plate on a small car
    d.rounded_rectangle([980, 770, 1180, 850], radius=16, fill=(60, 60, 70))
    d.rectangle([1005, 798, 1150, 828], fill="white", outline="black", width=2)
    d.rectangle([1005, 798, 1025, 828], fill=(0, 51, 153))
    d.text((1008, 802), "E", font=font(18), fill="white")
    d.text((1032, 800), "1234 JKL", font=font(22), fill="black")

    img.save(path, quality=90)
    print("saved", path, img.size)
    return path


if __name__ == "__main__":
    build()
