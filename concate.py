from PIL import Image

im1 = Image.open('./presentation/1line.png')
im2 = Image.open('./presentation/2line.png')


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


get_concat_h(im1, im2).save('./presentation/concat_h.jpg')



