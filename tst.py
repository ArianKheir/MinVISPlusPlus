from PIL import Image
im = Image.open("/home/user1/Arian/MinVIS/00100.jpg")
im_lr = im.transpose(Image.FLIP_LEFT_RIGHT)   # mirror left↔right
im_lr.save("00100_r.png")