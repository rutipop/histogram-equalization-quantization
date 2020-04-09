import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from imageio import imread

valMatrix = np.array([[0.299, 0.587, 0.114],
                      [0.596, -0.275, -0.321],
                      [0.212, -0.523, 0.311]])


def read_image(filename, representation):
    im = imread(filename)
    if (im.dtype == np.uint8 or im.dtype == int or im.np.matrix.max > 1):
        im = im.astype(np.float64) / 255

    if ((representation == 1) and (len(im.shape) >= 3)):  # process image into gray scale

        im = rgb2gray(im)

    return im


def imdisplay(filename, representation):
    if (representation == 1):
        plt.imshow(read_image(filename, representation), cmap=plt.get_cmap('gray'))
    else:
        plt.imshow(read_image(filename, representation), cmap=plt.get_cmap())
    plt.show()


def rgb2yiq(imRGB):
    return np.dot(imRGB, valMatrix.T.copy())


#
#
def yiq2rgb(imYIQ):
    return np.dot(imYIQ, np.linalg.inv(valMatrix).T.copy())


#
#
#
def histogram_equalize_grayImg(im_orig):
    # 1. Compute the image histogram :
    hist_orig, bins = np.histogram(im_orig, bins=256, range=(0, 1))
    # 2. Compute the cumulative histogram :
    cumsum = np.cumsum(hist_orig)
    #  3. Normalize the cumulative histogram (divide by the total number of pixels)  and
    # create look up table:
    m = np.argmax(hist_orig > 0)

    look_up_table = ((cumsum - cumsum[m]) / (cumsum[255] - cumsum[m]) * 255).astype(int)
    # # 4. Multiply the normalized histogram by the maximal gray level value (K-1)+
    # # 5. Verify that the minimal value is 0 and that the maximal is K-1, otherwise stretch
    #  the result linearly in the range [0,K-1].+
    # # 6. Round the values to get integers
    # # 7. Map the intensity values of the image using the result of step 5.
    im_eq = np.array(look_up_table)[(im_orig * 255).astype(np.uint8)].astype(np.float64) / 255

    np.clip(im_eq, 0, 1)
    hist_eq, bins = np.histogram(im_eq, bins=256, range=(0, 1))
    return im_eq, hist_orig, hist_eq


def histogram_equalize(im_orig):
    if (len(im_orig.shape) >= 3):  # means its an rgb image so we convert to YIQ and take the Y
        im_orig = rgb2yiq(im_orig)
        y = im_orig[:, :, 0]
        y, hist_orig, hist_eq = histogram_equalize_grayImg(y)
        im_orig[:, :, 0] = y  # converting the new image
        im_eq = yiq2rgb(im_orig)  # convert back to rgb image
        return im_eq, hist_orig, hist_eq
    else:  # means its a grayscale image
        return histogram_equalize_grayImg(im_orig[:, :])


# find first z limits to start finding qi's(doing the iteration)
def findFirstZs(my_hist, n_quant):
    my_cumsum = np.cumsum(my_hist)
    numOfPixels = my_cumsum[-1]
    # how much pixels are sopoused(in best case) to be approximetly in each [zi-zi+1]:
    numOfPixelsInEachQuant = numOfPixels / n_quant
    z_array = np.zeros(n_quant + 1).astype(np.uint8)  # The first and last elements are 0 and 255 respectively.
    for i in range(1, n_quant):
        z_array[i] = (np.abs(my_cumsum - numOfPixelsInEachQuant * i)).argmin()
    z_array[-1] = 255  # last limit is the last color pixel
    return z_array


# creating new zi values according to the new qi's we just found
def newZvalues(z_array, q_array, n_quant):
    for i in range(1, n_quant):
        z_array[i] = np.ceil(((q_array[i - 1] + q_array[i]) / 2))
    return z_array


# creating new q values according to the formula from class
def newQvalues(z_array, q_array, n_quant, my_hist):
    for i in range(0, n_quant):
        q_array[i] = np.inner(np.arange(z_array[i], z_array[i + 1]), my_hist[z_array[i]: z_array[i + 1]])
        q_array[i] /= np.sum(my_hist[z_array[i]: z_array[i + 1]])
    return q_array


# calculating square error
def calculateError(z_array, q_array, n_quant, my_hist):
    ans = 0
    for i in range(0, n_quant):
        ans = ans + np.sum(np.power(q_array[i] -
                                    np.arange(z_array[i], z_array[i + 1]), 2) * my_hist[z_array[i]: z_array[i + 1]])
    return np.sqrt(ans)


# creating look up table for quantiziation purposes
def createLUT(z_array, q_array, n_quant):
    lut = np.zeros(256)
    z_array = np.rint(z_array).astype(np.uint8)
    for i in range(n_quant):
        lut[z_array[i]: z_array[i + 1]] = q_array[i]
        if (i == n_quant - 1):
            lut[255] = q_array[i]
    lut = np.rint(lut).astype(np.uint8)
    return lut


# quantizing only for gray scale images
def quantize_gray_scale(im_orig, n_quant, n_iter):
    my_hist = np.histogram(im_orig, bins=256)[0]
    z_array = findFirstZs(my_hist, n_quant)  # set first default z's positions
    q_array = np.zeros(n_quant).astype(np.float64)
    updated_z_array = np.zeros(len(z_array))  # which is n_quant+1
    error = []

    for iteration in range(n_iter):
        # Calculate our q for each section, by using the formula given
        q_array = newQvalues(z_array, q_array, n_quant, my_hist)

        # creating new z's according to new q's we got :
        z_array = newZvalues(z_array, q_array, n_quant)
        # fill in error for this iteration

        error.append(calculateError(z_array, q_array, n_quant, my_hist))

        # check if we updated our z limits, if no we finish the quantization calculation :

        if (np.array_equal(updated_z_array, z_array)):
            break

        updated_z_array = np.copy(z_array)  # save the last copy of our new z limits

    ## if we got here it means we maxed out our iteration number or we got the same z limits twice:
    lut = createLUT(z_array, q_array, n_quant)  # creating look up table according to our quantziation results
    orig = ((im_orig * 255)).astype(np.uint8)
    im_quant = lut[orig]

    im_quant = im_quant.astype(np.float64)
    im_quant = im_quant / 255

    # return im_quant, error
    return im_quant, error


def quantize(im_orig, n_quant, n_iter):
    if len(im_orig.shape) == 3:  ##means its an rgb image
        im_orig = rgb2yiq(im_orig)  # converting to grayscale
        y = im_orig[:, :, 0]

        y, error = quantize_gray_scale(y, n_quant, n_iter)

        im_orig[:, :, 0] = y

        im_quant = yiq2rgb(im_orig)  # converting back to rgb
        return im_quant, error
    else:  # means its a grayscale image
        return quantize_gray_scale(im_orig, n_quant, n_iter)

# if __name__ == "__main__":
#     im = read_image('gray.jpeg', 1)
#     # quantization:## do it for 2,3,5,50
#     a1,a2=quantize(im,2,100)
#     plt.imshow(a1, cmap=plt.cm.gray)
#     plt.show()
#     hist_orig, bins = np.histogram(a1, bins=256, range=(0, 1))
#
#     plt.plot(hist_orig)
#     plt.show()
#
#     #
#     # result = histogram_equalize(im)[0]
#     # plt.imshow(result, cmap=plt.cm.gray)
#     # plt.show()
