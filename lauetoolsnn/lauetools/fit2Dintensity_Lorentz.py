# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as sciopt

def lorentzian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x) ** 2
    width_y = float(width_y) ** 2
    return lambda x, y: height / (
        1.0 + 4 * (x - center_x) ** 2 / width_x + 4 * (y - center_y) ** 2 / width_y)


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fitlorentzian(data):
    """Returns (height, x, y, width_x, width_y)
    the lorentzian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p123: np.ravel(lorentzian(*p123)(*np.indices(data.shape)) - data)
    p123, _ = sciopt.leastsq(errorfunction, params)
    return p123


def momentsr(data, circle, rotate, vheight):
    """Returns (height, amplitude, x, y, width_x, width_y, rotation angle)
    the gaussian parameters of a 2D distribution by calculating its
    moments.  Depending on the input parameters, will only output
    a subset of the above"""
    print("Not yet implemented for lorentzian")
    pass
    # total = data.sum()
    # X, Y = indices(data.shape)
    # x = (X*data).sum()/total
    # y = (Y*data).sum()/total
    # col = data[:, int(y)]
    # width_x = sqrt(abs((arange(col.size)-y)**2*col).sum()/col.sum())
    # row = data[int(x), :]
    # width_y = sqrt(abs((arange(row.size)-x)**2*row).sum()/row.sum())
    # width = ( width_x + width_y ) / 2.
    # height = stats.mode(data.ravel())[0][0]
    # amplitude = data.max()-height
    # mylist = [amplitude,x,y]
    # if vheight==1:
    # mylist = [height] + mylist
    # if circle==0:
    # mylist = mylist + [width_x,width_y]
    # else:
    # mylist = mylist + [width]
    # if rotate==1:
    # mylist = mylist + [0.] #rotation "moment" is just zero...
    # return tuple(mylist)


def twodlorentzian(inpars, circle, rotate, vheight):
    """Returns a 2d gaussian function of the form:
        x' = cos(rota) * x - sin(rota) * y
        y' = sin(rota) * x + cos(rota) * y
        (rota should be in degrees)
        g = b + a /( 1 + 4*(x-center_x)**2/width_x +  4*(y-center_y)**2/width_y  )

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
        raise ValueError("There are still input parameters:"
            + str(inpars) + " and you've input: "
            + str(inpars_old) + " circle=%d, rotate=%d, vheight=%d" % (circle, rotate, vheight))

    def rotlorentz(x, y):
        """
        a*w**2/((x-x0)**2+w**2)

        """
        if rotate == 1:
            xp = x * np.cos(rota) - y * np.sin(rota)
            yp = x * np.sin(rota) + y * np.cos(rota)
        else:
            xp = x
            yp = y
        # g = height+amplitude/( 1 + 4*(x-rcen_x)**2/width_x +  4*(y-rcen_y)**2/width_y  )
        g = height + amplitude / (1.0 + 4 * (xp - rcen_x) ** 2 / width_x) / (
            1 + 4 * (yp - rcen_y) ** 2 / width_y)
        return g

    return rotlorentz


def lorentzfit(data, err=None, params=[], autoderiv=1, return_all=0,
                                                        circle=0,
                                                        rotate=1,
                                                        vheight=1,
                                                        xtol=0.0000001):
    """
    Lorentzian fitter with the ability to fit a variety of different forms of 2-dimensional gaussian.

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
        params = momentsr(data, circle, rotate, vheight)
    if err is None:
        errorfunction = lambda p123: np.ravel(
            (twodlorentzian(p123, circle, rotate, vheight)(*np.indices(data.shape)) - data))
    else:
        errorfunction = lambda p123: np.ravel(
            (twodlorentzian(p123, circle, rotate, vheight)(*np.indices(data.shape)) - data) / err)
    if autoderiv == 0:
        # the analytic derivative, while not terribly difficult, is less efficient and useful.  I only bothered
        # putting it here because I was instructed to do so for a class project - please ask if you would like
        # this feature implemented
        raise ValueError("I'm sorry, I haven't implemented this feature yet.")
    else:
        p123, cov, infodict, errmsg, _ = sciopt.leastsq(
            errorfunction, params, full_output=1, xtol=xtol)
    if return_all == 0:
        return p123
    elif return_all == 1:
        return p123, cov, infodict, errmsg

# if __name__ == "__main__":

#     if 0:
#         # Create the gaussian data
#         # -----------------------------------------------------
#         Xin, Yin = np.mgrid[0:201, 0:201]
#         # data = gaussian(3, 100, 100, 20, 40)(Xin, Yin) + random.random(Xin.shape)
#         # inpars = (height,amplitude,center_x,center_y,width_x,width_y,rota)
#         inpars = (0, 200, 50, 50, 10, 40, 45)
#         inparsshort = inpars[:5]
#         circle = 0
#         rotate = 1
#         vheight = 1
#         data = twodlorentzian(inpars, circle, rotate, vheight)(Xin, Yin) + 10 * np.random.random(Xin.shape)
#         # -------------------------------------

#     fifi = "Zr_A169_0220.mccd"
#     fifi = "CKRMONO.0057"
#     fifi = "Ge_run41_1_0003.mccd"

#     center_pixel = (1296, 2048 - 1021)  # XMAS and array convention (x,2048-yfit2d)
#     center_pixel = (1079, 725)
#     center_pixel = (1588, 818)
#     xboxsize, yboxsize = 20, 20

#     dat = IOimage.readoneimage_crop(fifi, center_pixel, (xboxsize, yboxsize))

#     def fromindex_to_pixelpos_x(index, _):
#         return center_pixel[0] - xboxsize + index

#     def fromindex_to_pixelpos_y(index, _):
#         return center_pixel[1] - yboxsize + index

#     # ax = axes()
#     # ax.xaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_x))
#     # ax.yaxis.set_major_formatter(FuncFormatter(fromindex_to_pixelpos_y))
#     # imshow(dat,interpolation='nearest')#,origin='lower')
#     # show()

#     start_baseline = np.amin(dat)
#     # start_j,start_i=yboxsize,xboxsize # from input center
#     start_j, start_i = (np.argmax(dat) // dat.shape[1], np.argmax(dat) % dat.shape[1])  # from maximum intensity in dat
#     # start_amplitude=amax(dat)-start_baseline
#     start_amplitude = dat[start_j, start_i] - start_baseline
#     start_sigma1, start_sigma2 = 10, 5
#     start_anglerot = 0
#     startingparams = [start_baseline,
#                         start_amplitude,
#                         start_j,
#                         start_i,
#                         start_sigma1,
#                         start_sigma2,
#                         start_anglerot]

#     print("startingparams")
#     print(startingparams)

#     if all(dat > 0):
#         print("logscale")
#         p.matshow(np.log(dat), cmap=GT.GIST_EARTH_R, interpolation="nearest", origin="upper")
#     else:
#         p.imshow(dat, cmap=GT.GIST_EARTH_R, interpolation="nearest", origin="upper")

#     params, cov, infodict, errmsg = lorentzfit(dat,
#                                             err=None,
#                                             params=startingparams,
#                                             autoderiv=1,
#                                             return_all=1,
#                                             circle=0,
#                                             rotate=1,
#                                             vheight=1)

#     print("\n *****fitting results with Lorentzian************\n")
#     print(params)
#     print("background intensity:            %.2f" % params[0])
#     print("Peak amplitude above background        %.2f" % params[1])
#     print("pixel position (X)            %.2f" % (params[3] - xboxsize + center_pixel[0]))
#     print("pixel position (Y)            %.2f" % (params[2] - yboxsize + center_pixel[1]))
#     print("std 1,std 2 (pix)            ( %.2f , %.2f )" % (params[4], params[5]))
#     print("e=min(std1,std2)/max(std1,std2)        %.3f"
#         % (min(params[4], params[5]) / max(params[4], params[5])))
#     print("Rotation angle (deg)            %.2f" % (params[6] % 360))
#     print("************************************\n")
#     print(params)
#     inpars_res = params
#     fit = twodlorentzian(inpars_res, 0, 1, 1)

#     # params = fitgaussian(data)
#     # fit = gaussian(*params)

#     # params1=[  8.37532076e+01 ,  7.46353362e+01 ,  2.30777891e+01+11 ,  2.23761344e+01+8,
#     # 6.05357888e+00  , 1.20693476e+01 , -1.16509603e+06]
#     # params2=[ 79.3635639 ,  93.12639397  , 9.3952523 +70,  29.51984472+22 ,  5.46769477
#     # ,  17.71425257 , 84.66087334]
#     # params3=[ 89.21628068  ,53.17974564,  9.98964225+18 ,  5.45738096+59 , 11.84612735,
#     # 5.08762344 , 30.90213561]
#     # fit=twodlorentzian(params,0,1,1)
#     # fit1=twodlorentzian(params1,0,1,1)
#     # fit2=twodlorentzian(params2,0,1,1)
#     # fit3=twodlorentzian(params3,0,1,1)

#     isointensity = np.linspace(1000, np.amax(dat), 20)
#     p.contour(fit(*np.indices(dat.shape)), isointensity, cmap=GT.COPPER)
#     # contour(fit(*indices(dat.shape))+fit1(*indices(dat.shape))+fit2(*indices(dat.shape))+fit3(*indices(dat.shape)), isointensity,cmap=cm.copper)
#     ax = p.gca()
#     ax.xaxis.set_major_formatter(p.FuncFormatter(fromindex_to_pixelpos_x))
#     ax.yaxis.set_major_formatter(p.FuncFormatter(fromindex_to_pixelpos_y))
#     # (height, x, y, width_x, width_y) = params

#     # text(0.95, 0.05, """
#     # x : %.1f
#     # y : %.1f
#     # width_x : %.1f
#     # width_y : %.1f""" %(x, y, width_x, width_y),
#     # fontsize=16, horizontalalignment='right',
#     # verticalalignment='bottom', transform=ax.transAxes)

#     p.show()
