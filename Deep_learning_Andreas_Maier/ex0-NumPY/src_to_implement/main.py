import pattern
import numpy as np
import matplotlib.pyplot as plt
from generator import ImageGenerator

# c = pattern.Checker(250, 25)
# c.draw()
# c.show()


# c = pattern.Circle(1024, 200, (512, 256))
# c.draw()
# c.show()
#
# s = pattern.Spectrum(255)
# s.draw()
# s.show()

# i = ImageGenerator()

# def checkerboard(boardsize, squaresize):
#     return np.fromfunction(lambda i, j: (i//squaresize[0])%2 != (j//squaresize[1])%2, boardsize).astype(int)

# res = checkerboard((10,15), (2,3))
# print(res)
#
# plt.figure()
# # arr = np.asarray(res)
# # plt.plot(arr)
# plt.plot(res)
# plt.imshow(res, cmap='gray', vmin=0, vmax=255)


label_path = './Labels.json'
file_path = './exercise_data/'

gen = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=False, mirroring=False,
                     shuffle=False)

gen.next()
gen.next()
gen.next()
gen.next()
gen.next()

gen.next()
gen.next()
gen.next()
gen.next()
gen.next()

gen.next()
gen.next()
gen.next()
gen.next()
gen.next()

gen.next()
gen.next()
gen.next()
gen.next()
