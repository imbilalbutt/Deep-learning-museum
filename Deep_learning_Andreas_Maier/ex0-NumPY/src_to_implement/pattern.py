import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


#  Just writing the comment

class Checker:
    output: np.ndarray

    # integer resolution that defines the number of pixels in each dimension
    # integer tile_size that defines the number of pixel an individual tile has in each dimension.

    # For
    # example,
    # if resolution is 800 and tile size is 100, the resulting image will be a 8x8 grid of tiles,
    # where each tile is 100x100 pixels in size.The total size of the image will be 800x800 pixels.
    def __init__(self, resolution: int, tile_size: int):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output: ndarray = np.zeros((self.resolution, self.resolution), dtype=np.uint8)

    def draw(self):

        # reshape adds another dimension
        # result = self.output[(np.arange(self.resolution) % 2) ^ (np.arange(self.resolution) % 2 != 0)]

        ## GOOD : Interesting shape
        # self.output = (np.arange(self.resolution) % self.tile_size).reshape(-1, 1) ^ (np.arange(self.resolution) % self.tile_size == 0)

        # Increasing Contrast
        # self.output = (np.arange(self.resolution) % self.tile_size).reshape(-1, 1) ^ (np.arange(self.resolution) % self.tile_size == 0)//(2*self.tile_size)

        ## Correct: With mod 2 to : 1 by 1 0/1
        # self.output = (np.arange(self.resolution) % (2)).reshape(-1, 1) != (np.arange(self.resolution) % (2) == 0) // (2 * self.tile_size)

        # Incorrect but min max is 0 & 1
        # self.output = (np.arange(self.resolution) % (2)).reshape(-1, 1) ^ (np.arange(self.resolution) % (self.tile_size*self.tile_size) == 0)

        # self.output = (np.arange(self.resolution) % 2).reshape(-1, 1) ^ (np.arange(self.resolution) % (self.tile_size) != 0)

        # self.output = np.fromfunction(lambda i, j: (i//self.tile_size[0]) % 2 != (j//self.tile_size[1]) % 2, self.output).astype(int)

        # result = self.output[(np.arange(self.resolution) % 2).reshape(-1, self.tile_size)]


        # self.output = (np.arange(self.resolution) % 2).reshape(-1, self.tile_size)

        # Axis 0 will act on all the ROWS in each COLUMN (basically operate on x-axis)
        # Axis 1 will act on all the COLUMNS in each ROW (basically operates on y-axis)
        ## Incorrect checkerboard But Minx Max from 0 to 1
        # self.output = ((np.arange((self.resolution))) // self.tile_size).reshape(-1, 1) & ((np.arange((self.resolution))) // self.tile_size) % 2

        self.output = (np.indices((self.resolution, self.resolution)) // self.tile_size).sum(axis=0) % 2

        return self.output.copy()

    def show(self):
        plt.figure()
        plt.plot(np.asarray(self.output))
        plt.imshow(np.asarray(self.output), cmap='gray', vmin=0, vmax=1)
        plt.show()


class Circle:
    output: np.ndarray

    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.zeros((self.resolution, self.resolution), dtype=np.uint8)

    def draw(self):
        # for x: column remains the same
        # for y: row remains the same
        # both of them representing indices ( each containing a coordinate grid )
        xx, yy = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        # print("X = ", x.shape)
        # print("y = ", y.shape)
        # L2 norm distance b/w (each point of grid, circle's cntr)
        distance = np.sqrt((xx - self.position[0]) ** 2 + (yy - self.position[1]) ** 2)

        self.output[distance <= self.radius] = 1
        return self.output.copy()

    def show(self):
        plt.figure()
        plt.plot(np.asarray(self.output))
        plt.imshow(np.asarray(self.output), cmap='gray', vmin=0, vmax=1)
        plt.show()
        plt.gray()
        plt.imsave('./circle.png', self.output)

# class Spectrum:
#     output: np.ndarray
#     rgb_channel = 3
#
#     def __init__(self, resolution):
#         self.resolution = resolution
#         # self.output = np.zeros((self.resolution, self.resolution, self.rgb_channel), dtype=np.uint8)
#         self.output = np.zeros((resolution, resolution, 3), dtype=np.uint8)
#         # self.output = [np.zeros((self.resolution,self.resolution,1)), np.zeros((self.resolution,self.resolution,1)), np.zeros((self.resolution,self.resolution,1))]
#
#     def draw(self):
#         # self.output = (np.indices((self.resolution, self.resolution, 3))).sum(axis=0)
#
#         ### Method 1: prints slights incorrect but interesting result.
#         # x = (np.linspace(0, 1, self.resolution, endpoint=True))
#         # y = (np.linspace(0, 1, self.resolution, endpoint=True))
#         # xx, yy = np.meshgrid(x, y)
#         # self.output = (xx + yy) % 2
#         ###
#         # self.output[:, 0:self.resolution] = (0, np.linspace(0, 1, self.resolution), 0)
#
#         x = np.linspace(0, 1, self.resolution)
#
#         # for i in range(0, self.resolution):
#         #     self.output[1, :, :] = (x[i], 0, 0)
#         #     self.output[2, :, :] = (0, x[i], 0)
#         #     self.output[3, :, :] = (0, 0, x[i])
#
#         # blue = (255, 0, 0)
#         self.output[:, :, 1] = (x, 0, 0)
#
#         # green = (0, 255, 0)
#         self.output[:, :, 2] = (0, x, 0)
#
#         # red = (0, 0, 255)
#         self.output[:, :, 3] = (0, 0, x)
#
#         # self.output[:, :, 3] = np.linspace(0, 1, self.resolution)
#
#         # self.output = np.linspace((self.resolution, self.resolution), self.resolution)
#         # self.output = np.linspace(0, 1, self.resolution, endpoint=True)
#
#         # self.output = np.stack((result, result, result), axis=2)
#         return self.output.copy()
#
#     def show(self):
#         plt.figure()
#         plt.plot(self.output)
#         plt.imshow(self.output, cmap='rainbow')
#         plt.show()

class Spectrum:

    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        self.output = np.zeros((self.resolution, self.resolution, 3))
        red_chanel = np.linspace(0,1, self.resolution).reshape(1, self.resolution)
        red_chanel = np.tile(red_chanel, [self.resolution, 1])
        #red_chanel = np.flipud(red_chanel)
        self.output[:,:, 0] = red_chanel

        green_chanel = red_chanel.transpose(1, 0)
        self.output[:,:, 1] = green_chanel

        blue_chanel = np.fliplr(red_chanel)
        self.output[:, :, 2] = blue_chanel

        return self.output.copy()

    def show(self):
        plt.figure()
        plt.imshow(self.output)
        plt.show()