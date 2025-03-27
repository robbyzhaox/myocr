import PIL
import PIL.Image
from PIL.Image import Image


def crop_rectangle(image: Image, box, target_height=32):
    left, top, right, bottom = map(int, (box.left, box.top, box.right, box.bottom))
    width, height = image.size

    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))

    cropped = image.crop((left, top, right, bottom))
    orig_width, orig_height = cropped.size
    aspect_ratio = orig_width / orig_height

    new_width = int(target_height * aspect_ratio)
    resized = cropped.resize((new_width, target_height), PIL.Image.Resampling.BICUBIC)
    return resized


class LabelTranslator:
    def __init__(self, alphabet):
        self.alphabet = alphabet + "ç"  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def decode(self, t, length, raw=False):
        t = t[:length]
        if raw:
            return "".join([self.alphabet[i - 1] for i in t])
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
            return "".join(char_list)
