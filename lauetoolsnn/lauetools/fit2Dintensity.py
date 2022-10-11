# -*- coding: utf-8 -*-
"""
module to fit 2d value in array

still in development for fit of multiple peaks in ROI
"""
import pylab as p
import numpy as np

from scipy import optimize, stats

# from matplotlib.ticker import FuncFormatter

try:
    from lauetools import generaltools as GT 
except:
    import lauetoolsnn.lauetools.generaltools as GT

USE_CYTHON = False

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height * np.exp(
        -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum() * 1.0
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)] * 1.0
    width_x = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :] * 1.0
    width_y = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, _ = optimize.leastsq(errorfunction, params)
    return p


def momentsr(data, circle, rotate, vheight):
    """Returns (height, amplitude, x, y, width_x, width_y, rotation angle)
    the gaussian parameters of a 2D distribution by calculating its
    moments.  Depending on the input parameters, will only output
    a subset of the above"""
    total = data.sum() * 1.0
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)] * 1.0
    width_x = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :] * 1.0
    width_y = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    width = (width_x + width_y) / 2.0
    height = stats.mode(data.ravel())[0][0]
    amplitude = data.max() - height
    mylist = [amplitude, x, y]
    if vheight == 1:
        mylist = [height] + mylist
    if circle == 0:
        mylist = mylist + [width_x, width_y]
    else:
        mylist = mylist + [width]
    if rotate == 1:
        mylist = mylist + [0.0]  # rotation "moment" is just zero...
    return tuple(mylist)


def twodgaussian(inpars, circle, rotate, vheight):
    """Returns a 2d gaussian function of the form:
        x' = cos(rota) * x - sin(rota) * y
        y' = sin(rota) * x + cos(rota) * y
        (rota should be in degrees)
        g = b + a exp ( - ( ((x-center_x)/width_x)**2 +
        ((y-center_y)/width_y)**2 ) / 2 )

        where x and y are the input parameters of the returned function,
        and all other parameters are specified by this function

        However, the above values are passed by list.  The list should be:
        inpars = (height,amplitude,center_x,center_y,width_x,width_y,rota)

        You can choose to ignore / neglect some of the above input parameters using the following options:
            circle=0 - default is an elliptical gaussian (different x, y widths), but can reduce
                the input by one parameter if it's a circular gaussian
            rotate=1 - default allows rotation of the gaussian ellipse.  Can remove last parameter
                by setting rotate=0
            vheight=1 - default allows a variable height-above-zero, i.e. an additive constant
                for the Gaussian function.  Can remove first parameter by setting this to 0
        """
    inpars_old = inpars
    inpars = list(inpars)
    if vheight == 1:
        height = inpars.pop(0)
        height = float(height)
    else:
        height = float(0)
    amplitude, center_x, center_y = inpars.pop(0), inpars.pop(0), inpars.pop(0)
    amplitude = float(amplitude)
    center_x = float(center_x)
    center_y = float(center_y)
    if circle == 1:
        width = inpars.pop(0)
        width_x = float(width)
        width_y = float(width)
    else:
        width_x, width_y = inpars.pop(0), inpars.pop(0)
        width_x = float(width_x)
        width_y = float(width_y)
    if rotate == 1:
        rota = inpars.pop(0)
        rota = np.pi / 180.0 * float(rota)
        rcen_x = center_x * np.cos(rota) - center_y * np.sin(rota)
        rcen_y = center_x * np.sin(rota) + center_y * np.cos(rota)
    else:
        rcen_x = center_x
        rcen_y = center_y
    if len(inpars) > 0:
        raise ValueError(
            "There are still input parameters:"
            + str(inpars)
            + " and you've input: "
            + str(inpars_old)
            + " circle=%d, rotate=%d, vheight=%d" % (circle, rotate, vheight))

    def rotgauss(x, y):
        if rotate == 1:
            xp = x * np.cos(rota) - y * np.sin(rota)
            yp = x * np.sin(rota) + y * np.cos(rota)
        else:
            xp = x
            yp = y
        g = height + amplitude * np.exp(
            -(((rcen_x - xp) / width_x) ** 2 + ((rcen_y - yp) / width_y) ** 2) / 2.0)
        return g

    return rotgauss

def twodgaussian_2peaks(inpars, circle, rotate, vheight):
    """Returns a 2d gaussian function of the form:
    x' = cos(rota) * x - sin(rota) * y
    y' = sin(rota) * x + cos(rota) * y
    (rota should be in degrees)
    g = b + a exp ( - ( ((x-center_x)/width_x)**2 +
    ((y-center_y)/width_y)**2 ) / 2 )

    where x and y are the input parameters of the returned function,
    and all other parameters are specified by this function

    However, the above values are passed by list.  The list should be:
    inpars = (height,amplitude,center_x,center_y,width_x,width_y,rota, # for gaussian 1
                amplitude,center_x,center_y,width_x,width_y,rota)) for gaussian 2

    You can choose to ignore / neglect some of the above input parameters using the following options:
        circle=0 - default is an elliptical gaussian (different x, y widths), but can reduce
            the input by one parameter if it's a circular gaussian
        rotate=1 - default allows rotation of the gaussian ellipse.  Can remove last parameter
            by setting rotate=0
        vheight=1 - default allows a variable height-above-zero, i.e. an additive constant
            for the Gaussian function.  Can remove first parameter by setting this to 0
    """
    # inpars_old = inpars
    inpars = list(inpars)

    if vheight == 1:
        # height = inpars.pop(0)
        height_1 = float(inpars[0])

    else:
        print("Still Not implemented in twodgaussian_2peaks()")
        # height_1 = float(0)
        # height_2 = float(0)

    amplitude_1, center_x_1, center_y_1 = inpars[1:4]
    amplitude_2, center_x_2, center_y_2 = inpars[7:10]

    amplitude_1 = float(amplitude_1)
    amplitude_2 = float(amplitude_2)
    center_x_1 = float(center_x_1)
    center_y_1 = float(center_y_1)
    center_x_2 = float(center_x_2)
    center_y_2 = float(center_y_2)

    if circle == 1:
        print("Still Not implemented in twodgaussian_2peaks()")
        # width = inpars.pop(0)
        # width_x = float(width)
        # width_y = float(width)
    else:
        width_x_1, width_y_1 = inpars[4:6]
        width_x_1 = float(width_x_1)
        width_y_1 = float(width_y_1)
        width_x_2, width_y_2 = inpars[10:12]
        width_x_2 = float(width_x_2)
        width_y_2 = float(width_y_2)
    if rotate == 1:
        rota_1 = inpars[6]
        rota_1 = np.pi / 180.0 * float(rota_1)
        rcen_x_1 = center_x_1 * np.cos(rota_1) - center_y_1 * np.sin(rota_1)
        rcen_y_1 = center_x_1 * np.sin(rota_1) + center_y_1 * np.cos(rota_1)
        rota_2 = inpars[12]
        rota_2 = np.pi / 180.0 * float(rota_2)
        rcen_x_2 = center_x_2 * np.cos(rota_2) - center_y_2 * np.sin(rota_2)
        rcen_y_2 = center_x_2 * np.sin(rota_2) + center_y_2 * np.cos(rota_2)
    else:
        print("Still Not implemented in twodgaussian_2peaks()")
        # rcen_x = center_x
        # rcen_y = center_y
    # if len(inpars) > 0:
    # raise ValueError("There are still input parameters:" + str(inpars) + \
    # " and you've input: " + str(inpars_old) + " circle=%d, rotate=%d, vheight=%d" % (circle,rotate,vheight) )

    # def rotgauss_1(x,y):
    # if rotate==1:
    # xp = x * cos(rota_1) - y * sin(rota_1)
    # yp = x * sin(rota_1) + y * cos(rota_1)
    # else:
    # xp = x
    # yp = y
    # g1 = height_1+amplitude_1*exp(
    # -(((rcen_x_1-xp)/width_x_1)**2+
    # ((rcen_y_1-yp)/width_y_1)**2)/2.)
    # return g1

    # def rotgauss_2(x,y):
    # if rotate==1:
    # xp = x * cos(rota_2) - y * sin(rota_2)
    # yp = x * sin(rota_2) + y * cos(rota_2)
    # else:
    # xp = x
    # yp = y
    # g2 = height_2+amplitude_2*exp(
    # -(((rcen_x_2-xp)/width_x_2)**2+
    # ((rcen_y_2-yp)/width_y_2)**2)/2.)
    # return g2

    def rotgauss_2peaks(x, y):

        xp_1 = x * np.cos(rota_1) - y * np.sin(rota_1)
        yp_1 = x * np.sin(rota_1) + y * np.cos(rota_1)
        xp_2 = x * np.cos(rota_2) - y * np.sin(rota_2)
        yp_2 = x * np.sin(rota_2) + y * np.cos(rota_2)

        g = (height_1 + amplitude_1
            * np.exp(-(((rcen_x_1 - xp_1) / width_x_1) ** 2
                    + ((rcen_y_1 - yp_1) / width_y_1) ** 2)/ 2.0)
            + amplitude_2
            * np.exp(-(((rcen_x_2 - xp_2) / width_x_2) ** 2
                    + ((rcen_y_2 - yp_2) / width_y_2) ** 2)/ 2.0))

        return g

    return rotgauss_2peaks

    # if flag ==1:
    # return rotgauss_1
    # if flag ==2:
    # return rotgauss_2


def gaussfit( data, err=None, params=[], autoderiv=1, return_all=0, circle=0,
    rotate=1, vheight=1, xtol=0.0000001, Acceptable_HighestValue=False,
    Acceptable_LowestValue=False, ijindices_array=None,
    usecythonmodule=USE_CYTHON, computerrorbars=False):
    """
    Gaussian fitter with the ability to fit a variety of different forms of 2-dimensional gaussian.

    Input Parameters:
        data - 2-dimensional data array
        err=None - error array with same size as data array
        params=[] - initial input parameters for Gaussian function.
            (height, amplitude, x, y, width_x, width_y, rota)
            if not input, these will be determined from the moments of the system,
            assuming no rotation
        autoderiv=1 - use the autoderiv provided in the lmder.f function (the alternative
            is to us an analytic derivative with lmdif.f: this method is less robust)
        return_all=0 - Default is to return only the Gaussian parameters.  See below for
            detail on output
        circle=0 - default is an elliptical gaussian (different x, y widths), but can reduce
            the input by one parameter if it's a circular gaussian
        rotate=1 - default allows rotation of the gaussian ellipse.  Can remove last parameter
            by setting rotate=0
        vheight=1 - default allows a variable height-above-zero, i.e. an additive constant
            for the Gaussian function.  Can remove first parameter by setting this to 0

        Acceptable_HighestValue: if not False, pixel intensity level above which the pixel has a zero weight
        Acceptable_LowestValue: if not False, pixel intensity level below which the pixel has a zero weight

    Output:
        Default output is a set of Gaussian parameters with the same shape as the input parameters
        Can also output the covariance matrix, 'infodict' that contains a lot more detail about
            the fit (see scipy.optimize.leastsq), and a message from leastsq telling what the exit
            status of the fitting routine was

        Warning: Does NOT necessarily output a rotation angle between 0 and 360 degrees.
    """
    #     data = np.ma.array(data, mask=False)
    #
    #     data.mask[data >= 65535] = True
    # print("in gaussfit")
    if (Acceptable_HighestValue is not False) or (Acceptable_LowestValue is not False):
        err = np.ones(data.shape)
    if Acceptable_HighestValue is not False:
        err[data >= Acceptable_HighestValue] = 0
    if Acceptable_LowestValue is not False:
        err[data <= Acceptable_LowestValue] = 0

    if params == []:
        params = momentsr(data, circle, rotate, vheight)

    if ijindices_array is None:
        ijindices_array = np.indices(data.shape)

    if not usecythonmodule:
        if err is None:
            errorfunction = lambda p: np.ravel(
                (twodgaussian(p, circle, rotate, vheight)(*ijindices_array) - data))
        else:
            errorfunction = lambda p: np.ravel(
                (twodgaussian(p, circle, rotate, vheight)(*ijindices_array) - data)
                * err)
            
    if autoderiv == 0:
        # the analytic derivative, while not terribly difficult, is less efficient and useful.
        # I only bothered putting it here because I was instructed to do so for a class project.
        #  - please ask if you would like    # this feature implemented
        raise ValueError("I'm sorry, I haven't implemented this feature yet.")
    else:
        (p, cov, infodict, errmsg, success) = optimize.leastsq(
            errorfunction, params, full_output=1, xtol=xtol)

        if computerrorbars:
            # https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i/21844726
 
            n1, n2 = data.shape

            if (n1*n2 > len(p)) and cov is not None:
                s_sq = (errorfunction(p)**2).sum()/(n1*n2-len(p))
                pcov = cov * s_sq
            else:
                pcov = np.inf

            error = []
            for i in range(len(p)):
                try:
                    error.append(np.absolute(pcov[i][i])**0.5)
                except:
                    error.append(0.00)
            pfit_leastsq = p
            perr_leastsq = np.array(error)

            print("pfit_leastsq", pfit_leastsq)
            print("perr_leastsq", perr_leastsq)

    if return_all == 0:
        return p
    elif return_all == 1:
        return p, cov, infodict, errmsg


def gaussfit_2peaks( data, err=None, params=[], autoderiv=1, return_all=0,
                                                circle=0, rotate=1, vheight=1, xtol=0.0000001):
    """
    two Gaussians fitter with the ability to fit a variety of
    different forms of 2-dimensional gaussian.

    Input Parameters:
        data - 2-dimensional data array
        err=None - error array with same size as data array
        params=[] - initial input parameters for Gaussian function.
            (height, amplitude, x, y, width_x, width_y, rota)
            if not input, these will be determined from the moments of the system,
            assuming no rotation
        autoderiv=1 - use the autoderiv provided in the lmder.f function (the alternative
            is to us an analytic derivative with lmdif.f: this method is less robust)
        return_all=0 - Default is to return only the Gaussian parameters.  See below for
            detail on output
        circle=0 - default is an elliptical gaussian (different x, y widths), but can reduce
            the input by one parameter if it's a circular gaussian
        rotate=1 - default allows rotation of the gaussian ellipse.  Can remove last parameter
            by setting rotate=0
        vheight=1 - default allows a variable height-above-zero, i.e. an additive constant
            for the Gaussian function.  Can remove first parameter by setting this to 0

    Output:
        Default output is a set of Gaussian parameters with the same shape as the input parameters
        Can also output the covariance matrix, 'infodict' that contains a lot more detail about
            the fit (see scipy.optimize.leastsq), and a message from leastsq telling what the exit
            status of the fitting routine was

        Warning: Does NOT necessarily output a rotation angle between 0 and 360 degrees.
    """
    if params == []:
        print("I'm sorry, I haven't implemented this feature yet. in gaussfit_2peaks()")
        # params = (momentsr(data,circle,rotate,vheight))
    if err is None:
        errorfunction = lambda p: np.ravel(
            (twodgaussian_2peaks(p, circle, rotate, vheight)(*np.indices(data.shape))
                - data))
    else:
        errorfunction = lambda p: np.ravel(
            (twodgaussian(p, circle, rotate, vheight)(*np.indices(data.shape)) - data)
            / err)
    if autoderiv == 0:
        raise ValueError("I'm sorry, I haven't implemented this feature yet.")
    else:
        p, cov, infodict, errmsg, success = optimize.leastsq(
            errorfunction, params, full_output=1, xtol=xtol)
    if return_all == 0:
        return p
    elif return_all == 1:
        return p, cov, infodict, errmsg


def create2Dgaussiandata():
    # Create the gaussian data
    Xin, Yin = np.mgrid[0:201, 0:201]
    data = gaussian(3, 100, 100, 20, 40)(Xin, Yin) + np.random.random(Xin.shape)

    p.matshow(data, cmap=GT.GIST_EARTH_R)

    params = fitgaussian(data)
    fit = gaussian(*params)

    p.contour(fit(*np.indices(data.shape)), cmap=GT.COPPER)
    ax = p.gca()
    (height, x, y, width_x, width_y) = params

    p.text(0.95, 0.05,
        """
    x : %.1f
    y : %.1f
    width_x : %.1f
    width_y : %.1f"""
        % (x, y, width_x, width_y), fontsize=16, horizontalalignment="right",
        verticalalignment="bottom", transform=ax.transAxes)

    p.show()


# if __name__ == "__main__":

#     if 0:
#         # Create the gaussian data
#         # -----------------------------------------------------
#         Xin, Yin = np.mgrid[0:201, 0:201]
#         # data = gaussian(3, 100, 100, 20, 40)(Xin, Yin) + random.random(Xin.shape)
#         # inpars = (height,amplitude,center_x,center_y,width_x,width_y,rota)
#         inpars = (25, 100, 100, 100, 10, 20, 12)
#         inparsshort = inpars[:5]
#         circle = 0
#         rotate = 1
#         vheight = 1
#         data = twodgaussian(inpars, circle, rotate, vheight)(Xin, Yin) + 10 * np.random.random(Xin.shape)
#         # -------------------------------------

#     # fifi='Zr_A169_0220.mccd'
#     # fifi='CKRMONO.0057'
#     # fifi='BKr_0252.mccd'

#     fifi = "UO2_A163_2_0028.mccd"
#     dirname = "/home/micha/lauetools/trunk/Examples/UO2"

#     fifi = "He60keV_telque_mono_0007.mccd"
#     dirname = "/home/micha/LT4/"

#     CenCam = np.array([830, 1190])

#     if 0:
#         peak_1 = [778, 782]
#         peak_2 = [777, 778]
#         start_amplitude_1 = 22000
#         start_amplitude_2 = 1800
#     if 0:
#         peak_1 = np.array([609, 933])
#         peak_2 = peak_1 + (peak_1 - CenCam) * 0.01
#         # start_amplitude_1 = 22000
#         # start_amplitude_2 =1800

#     if 0:
#         peak_1 = np.array([1059, 937])
#         peak_2 = peak_1 + (peak_1 - CenCam) * 0.005
#         # start_amplitude_1 = 22000
#         # start_amplitude_2 =1800
#     if 1:
#         peak_1 = np.array([1392, 739])
#         peak_2 = peak_1 + (peak_1 - CenCam) * 0.005
#         peak_2 = [1398, 735]
#         # start_amplitude_1 = 22000
#         # start_amplitude_2 =1800

#     xboxsize, yboxsize = 20, 20

#     center_pixel = peak_1

#     if sys.version_info.major == 3:
#         from . import readmccd as RMCCD
#     else:
#         import readmccd as RMCCD

#     dat = IOimage.readoneimage_crop(
#         fifi, center_pixel, (xboxsize, yboxsize), dirname=dirname)

#     def fromindex_to_pixelpos_x(index, pos):
#         return center_pixel[0] - xboxsize + index

#     def fromindex_to_pixelpos_y(index, pos):
#         return center_pixel[1] - yboxsize + index

#     # ax = axes()
#     # ax.xaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_x))
#     # ax.yaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_y))
#     # imshow(dat,interpolation='nearest')#,origin='lower')
#     # show()

#     start_baseline = np.amin(dat)
#     # start_j,start_i=yboxsize,xboxsize # from input center
#     start_j, start_i = (
#         np.argmax(dat) // dat.shape[1],
#         np.argmax(dat) % dat.shape[1],
#     )  # from maximum intensity in dat
#     start_amplitude_1 = np.amax(dat) - start_baseline
#     start_amplitude_2 = start_amplitude_1 / 10.0
#     # start_amplitude=dat[start_j,start_i]-start_baseline
#     start_sigma1, start_sigma2 = 2, 2
#     start_anglerot = 0
#     # startingparams=[start_baseline,start_amplitude,start_j,start_i,start_sigma1,start_sigma2,start_anglerot,
#     # start_baseline,start_amplitude,start_j,start_i,start_sigma1,start_sigma2,start_anglerot]

#     startingparams = [start_baseline,
#         start_amplitude_1,
#         peak_1[1] - (center_pixel[1] - yboxsize),
#         peak_1[0] - (center_pixel[0] - xboxsize),
#         start_sigma1,
#         start_sigma2,
#         start_anglerot,
#         start_amplitude_2,
#         peak_2[1] - (center_pixel[1] - yboxsize),
#         peak_2[0] - (center_pixel[0] - xboxsize),
#         start_sigma1,
#         start_sigma2,
#         start_anglerot]

#     print("startingparams")
#     print(startingparams[:7])
#     print(startingparams[7:])

#     if 0:  # all(dat>0):
#         print("logscale")  # matshow seems slow !
#         p.matshow(
#             np.log(dat), cmap=GT.GIST_EARTH_R, interpolation="nearest", origin="upper")
#     else:
#         p.imshow(np.log(dat + 0.000000000000001), cmap=GT.GIST_EARTH_R,
#             interpolation="nearest",
#             origin="upper")

#     # fit one peak
#     params, cov, infodict, errmsg = gaussfit_2peaks(dat, err=None,
#         params=startingparams,
#         autoderiv=1,
#         return_all=1,
#         circle=0,
#         rotate=1,
#         vheight=1)

#     print("\n *****fitting results ************\n")
#     print("background intensity:            %.2f" % params[0])
#     print("Peak amplitude above background        %.2f" % params[1])
#     print("pixel position (X)            %.2f" % (params[3] - xboxsize + center_pixel[0]))
#     print("pixel position (Y)            %.2f" % (params[2] - yboxsize + center_pixel[1]))
#     print("std 1,std 2 (pix)            ( %.2f , %.2f )" % (params[4], params[5]))
#     print("e=min(std1,std2)/max(std1,std2)        %.3f"
#         % (min(params[4], params[5]) / max(params[4], params[5])))
#     print("Rotation angle (deg)            %.2f\n\n" % (params[6] % 360))

#     print("background intensity:            %.2f" % params[0])
#     print("Peak amplitude above background        %.2f" % params[7])
#     print("pixel position (X)            %.2f" % (params[9] - xboxsize + center_pixel[0]))
#     print("pixel position (Y)            %.2f" % (params[8] - yboxsize + center_pixel[1]))
#     print("std 1,std 2 (pix)            ( %.2f , %.2f )" % (params[10], params[11]))
#     print("e=min(std1,std2)/max(std1,std2)        %.3f"
#         % (min(params[10], params[11]) / max(params[10], params[11])))
#     print("Rotation angle (deg)            %.2f" % (params[12] % 360))
#     print("************************************\n")
#     # print params
#     inpars_res = params
#     fit_1 = twodgaussian_2peaks(inpars_res, 0, 1, 1)

#     # params = fitgaussian(data)
#     # fit = gaussian(*params)

#     # params1=[  8.37532076e+01 ,  7.46353362e+01 ,  2.30777891e+01+11 ,  2.23761344e+01+8,
#     # 6.05357888e+00  , 1.20693476e+01 , -1.16509603e+06]
#     # params2=[ 79.3635639 ,  93.12639397  , 9.3952523 +70,  29.51984472+22 ,  5.46769477
#     # ,  17.71425257 , 84.66087334]
#     # params3=[ 89.21628068  ,53.17974564,  9.98964225+18 ,  5.45738096+59 , 11.84612735,
#     # 5.08762344 , 30.90213561]
#     # fit=twodgaussian(params,0,1,1)
#     # fit1=twodgaussian(params1,0,1,1)
#     # fit2=twodgaussian(params2,0,1,1)
#     # fit3=twodgaussian(params3,0,1,1)
#     isointensity = np.logspace(np.log(params[0] * 0.5), np.log(max(params[1], params[7])), 40)
#     p.contour(fit_1(*np.indices(dat.shape)), isointensity, cmap=GT.COPPER)
#     # contour(fit(*indices(dat.shape))+fit1(*indices(dat.shape))+fit2(*indices(dat.shape))+fit3(*indices(dat.shape)), isointensity,cmap=cm.copper)
#     ax = p.gca()
#     ax.xaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_x))
#     ax.yaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_y))
#     # (height, x, y, width_x, width_y) = params

#     # text(0.95, 0.05, """
#     # x : %.1f
#     # y : %.1f
#     # width_x : %.1f
#     # width_y : %.1f""" %(x, y, width_x, width_y),
#     # fontsize=16, horizontalalignment='right',
#     # verticalalignment='bottom', transform=ax.transAxes)

#     p.show()
