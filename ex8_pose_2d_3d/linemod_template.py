import cv2
import numpy as np
import os

print(cv2.__version__)
print("LINEMOD available:", hasattr(cv2, "linemod"))


# Function to autocrop an image
def is_border(edge, color):
    im = edge.reshape(-1, edge.shape[2])
    return np.all(im == color, axis=1).all()


def autocrop(src):
    if src.shape[2] != 3:
        print("Error: src is not of type CV_8UC3!")
        return None

    win = [0, 0, src.shape[1], src.shape[0]]  # x, y, width, height
    edges = [
        [0, 0, src.shape[1], 1],
        [src.shape[1] - 2, 0, 1, src.shape[0]],
        [0, src.shape[0] - 2, src.shape[1], 1],
        [0, 0, 1, src.shape[0]],
    ]

    color = src[0, 0, :]

    nborder = sum(is_border(src[y : y + h, x : x + w], color) for (x, y, w, h) in edges)

    if nborder < 4:
        return win

    while is_border(
        src[win[1] + win[3] - 2 : win[1] + win[3], win[0] : win[0] + win[2]], color
    ):
        win[3] -= 1

    while is_border(
        src[win[1] : win[1] + win[3], win[0] + win[2] - 2 : win[0] + win[2]], color
    ):
        win[2] -= 1

    while is_border(src[win[1] : win[1] + 1, win[0] : win[0] + win[2]], color):
        win[1] += 1
        win[3] -= 1

    while is_border(src[win[1] : win[1] + win[3], win[0] : win[0] + 1], color):
        win[0] += 1
        win[2] -= 1

    return win


# Create linemod detector
def create_linemod_detector():
    color_modality = cv2.linemod.Modality().create("ColorGradient")
    detector = cv2.linemod.Detector([color_modality], [4, 2, 1])
    return detector


def main(tpath, threshold, ipath):
    templates = []
    tposes = []
    offsets = []

    cnt = 0
    detector = create_linemod_detector()

    # Load templates
    while True:
        tfile = os.path.join(tpath, f"template{cnt:04d}.png")
        t = cv2.imread(tfile, cv2.IMREAD_UNCHANGED)
        if t is None:
            break

        templates.append(t.copy())

        win = autocrop(t)
        if win:
            win[2] += 4
            win[3] += 4
            win[0] -= 2
            win[1] -= 2

            t = t[win[1] : win[1] + win[3], win[0] : win[0] + win[2]]

            offsets.append((win[2], win[3]))

            out = cv2.inRange(t, (0, 0, 244), (1, 1, 255))
            out = 255 - out

            sources = [t.copy()]
            detector.addTemplate(sources, f"{cnt:04d}", out)

        cnt += 1

    print(f"Number of templates: {len(templates)}")

    # Test on each input image
    for img_path in ipath:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Cannot read test image: {img_path}")
            continue

        sources = [img]
        matches, _ = detector.match(sources, threshold)

        if matches:
            i = 0
            cv2.imshow("template", templates[int(matches[i].class_id)])

            cv2.circle(img, (matches[i].x, matches[i].y), 8, (0, 255, 0), -1)

            pfile = os.path.join(
                tpath, f"template{int(matches[i].class_id):04d}_pose.txt"
            )
            if os.path.exists(pfile):
                with open(pfile, "r") as posefile:
                    m = np.loadtxt(posefile).reshape(4, 4)
                    print(f"Pose m: \n{m}")
            else:
                print(f"Unable to open pose file {pfile}")

            cv2.imshow("img", img)
            cv2.waitKey(0)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Invalid arguments")
        print("Example program usage: python linemod.py ./templates 55 ./files/*.png")
        print(
            "This will load templates from ../templates, set threshold to 55, and test on all .png files in ../files"
        )
    else:
        tpath = sys.argv[1]
        threshold = float(sys.argv[2])
        ipath = sys.argv[3:]
        main(tpath, threshold, ipath)
