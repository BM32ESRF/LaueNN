import numpy as np
import imageio
from skimage.feature import canny
import cv2

DEG = np.pi / 180.0

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def fast_hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    """
    hough line using vectorized numpy operations,
    may take more memory, but takes much less time
    
    Input:
    img - 2D binary image with nonzeros representing edges
    angle_step - Spacing between angles to use every n-th angle
                 between -90 and 90 degrees. Default step is 1.
    lines_are_white - boolean indicating whether lines to be detected are white
    value_threshold - Pixel values above or below the value_threshold are edges

    Returns:
    accumulator - 2D array of the hough transform accumulator
    theta - array of angles used in computation, in radians.
    rhos - array of rho values. Max size is 2 times the diagonal
           distance of the input image.
    """
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step)) #can be changed
    #width, height = col.size  #if we use pillow
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
    rhos1 = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    #are_edges = cv2.Canny(img,50,150,apertureSize = 3)
    y_idxs, x_idxs = np.nonzero(are_edges)  # (row, col) indexes to edges
    # Vote in the hough accumulator
    xcosthetas = np.dot(x_idxs.reshape((-1,1)), cos_theta.reshape((1,-1)))
    ysinthetas = np.dot(y_idxs.reshape((-1,1)), sin_theta.reshape((1,-1)))
    rhosmat = np.round(xcosthetas + ysinthetas) + diag_len
    rhosmat = rhosmat.astype(np.int16)
    for i in range(num_thetas):
        rhos, counts = np.unique(rhosmat[:,i], return_counts=True)
        accumulator[rhos,i] = counts
    return accumulator, thetas, rhos1, np.arange(0,2 * diag_len)

def hough_peaks(H, num_peaks, thetas, rhos):
    H_values_sorted = np.sort(np.unique(H))[::-1]
    H_values_num_peaks = H_values_sorted[:num_peaks]
    peaks = []
    for pi in H_values_num_peaks:
        indexes = np.argwhere(H == pi)
        for find_indexes in indexes:
            rho = rhos[find_indexes[0]]
            theta = thetas[find_indexes[1]]
            peaks.append([rho, theta])
    return np.array(peaks[0:num_peaks])

def peak_votes(accumulator, thetas, rhos):
    """ Finds the max number of votes in the hough accumulator """
    idx = np.argmax(accumulator)
    rho = rhos[int(idx / accumulator.shape[1])]
    theta = thetas[idx % accumulator.shape[1]]
    return idx, theta, rho

def hough_lines_draw(image, peaks, rhos1):
    x = np.arange(0, image.shape[0]-1)
    H = np.zeros(image.shape)
    for i in range(peaks.shape[0]):
        d = rhos1[int(peaks[i, 0])]
        theta = peaks[i, 1]
        y = ((d - x*np.cos(theta))/np.sin(theta)).astype(int)
        index = np.where(y<=image.shape[1]-1)[0]
        y1 = y[index]
        x1 = x[index]
        index1 = np.where(y1>=0)[0]
        y2 = y1[index1]
        x2 = x1[index1]
        H[x2, y2] = 1
    return H

def show_hough_line(img, accumulator, lines_images, peaks, thetas, rhos, rhos1, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img+lines_images, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')
    
    ax[1].imshow(accumulator, cmap='jet',
                   extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos1[-1], rhos1[0]])
    for ii in range(len(peaks[:, 0])):
        ax[1].scatter(np.rad2deg(peaks[ii,1]), rhos1[int(peaks[ii, 0])], color='r')
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def ComputeGnomon_2(TwiceTheta_Chi, CenterProjection=(45 * DEG, 0 * DEG)):
    data_theta = TwiceTheta_Chi[0] / 2.0
    data_chi = TwiceTheta_Chi[1]
    lat = np.arcsin(np.cos(data_theta * DEG) * np.cos(data_chi * DEG))  # in rads
    longit = np.arctan(
        -np.sin(data_chi * DEG) / np.tan(data_theta * DEG))  # + ones(len(data_chi))*(np.pi)
    centerlat, centerlongit = CenterProjection
    slat0 = np.ones(len(data_chi)) * np.sin(centerlat)
    clat0 = np.ones(len(data_chi)) * np.cos(centerlat)
    longit0 = np.ones(len(data_chi)) * centerlongit
    slat = np.sin(lat)
    clat = np.cos(lat)
    cosanguldist = slat * slat0 + clat * clat0 * np.cos(longit - longit0)
    _gnomonx = clat * np.sin(longit0 - longit) / cosanguldist
    _gnomony = (slat * clat0 - clat * slat0 * np.cos(longit - longit0)) / cosanguldist
    return _gnomonx, _gnomony

def computeGnomonicImage(TwiceTheta,Chi):
    # CenterProjectionAngleTheta = 50#45
    TwiceTheta_Chi = TwiceTheta,Chi
    Xgno,Ygno = ComputeGnomon_2(TwiceTheta_Chi, CenterProjection=(45 * DEG, 0 * DEG))
    pts =(np.array([Xgno,Ygno]).T)
    nbpeaks=len(pts)
    NbptsGno = 300
    maxsize = max(Xgno.max(),Ygno.max(),-Xgno.min(),-Ygno.min())+.0
    xgnomin,xgnomax,ygnomin,ygnomax=(-0.8,0.8,-0.5,0.5)
    xgnomin,xgnomax,ygnomin,ygnomax=(-maxsize,maxsize,-maxsize,maxsize)
    
    halfdiagonal = np.sqrt(xgnomax**2+ygnomax**2)*NbptsGno
    XGNO = np.array((Xgno-xgnomin)/(xgnomax-xgnomin)*NbptsGno, dtype=int)
    YGNO = np.array((Ygno-ygnomin)/(ygnomax-ygnomin)*NbptsGno, dtype=int)
    imageGNO=np.zeros((NbptsGno+1,NbptsGno+1), dtype=int)
    imageGNO[XGNO,YGNO]=1
    return imageGNO, nbpeaks, halfdiagonal

def InverseGnomon(_gnomonX, _gnomonY):
    """ from x,y in gnomonic projection gives lat and long
    return theta and chi of Q (direction of Q)

    WARNING: assume that center of projection is centerlat, centerlongit = 45 deg, 0
    """
    lat0 = np.ones(len(_gnomonX)) * np.pi / 4
    longit0 = np.zeros(len(_gnomonX))
    Rho = np.sqrt(_gnomonX ** 2 + _gnomonY ** 2) * 1.0
    CC = np.arctan(Rho)

    # the sign should be - !!
    lalat = np.arcsin(np.cos(CC) * np.sin(lat0) + _gnomonY / Rho * np.sin(CC) * np.cos(lat0))
    lonlongit = longit0 + np.arctan2(_gnomonX * np.sin(CC),
        Rho * np.cos(lat0) * np.cos(CC) - _gnomonY * np.sin(lat0) * np.sin(CC))

    Theta = np.arcsin(np.cos(lalat) * np.cos(lonlongit))
    Chi = np.arctan(np.sin(lonlongit) / np.tan(lalat))

    return 2.*np.rad2deg(Theta), np.rad2deg(Chi)

if __name__ == '__main__':   
    # ### or load directly the peaks from cor file
    # import LaueTools.IOLaueTools as IOLT
    # filename_ = r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\misc\img\HS261120b_SH2_S5_B_0000.cor"
    # data_theta, data_chi = IOLT.readfile_cor(filename_)[1:3]
    # l_tth = data_theta * 2.
    # l_chi = data_chi
    # img, _, _ = computeGnomonicImage(l_tth, l_chi)

    import matplotlib.pyplot as plt
    from LaueTools import imageprocessing as ImProc
    import LaueTools.dict_LaueTools as dictLT
    import LaueTools.LaueGeometry as Lgeo
    
    imgpath = 'img/HS261120b_SH2_S5_B_0000.tif'
    img = imageio.imread(imgpath)
    if img.ndim == 3:
        img = rgb2gray(img)
    
    backgroundimage = ImProc.compute_autobackground_image(img, boxsizefilter=10)
    img = ImProc.computefilteredimage(img, backgroundimage, "sCMOS", usemask=False,
                                                        formulaexpression="A-1.1*B")
    img = canny(img, )
    img = img.astype(int)
    
    # img2 = np.copy(img)
    # img2[img2==0] = 1000
    # plt.imshow(img2, cmap='gray')
    
    XX,YY = np.meshgrid(np.arange(img.shape[1]),np.arange(img.shape[0]))
    table = np.vstack((img.ravel(),XX.ravel(),YY.ravel())).T
    ## remove zero values pixels from peak
    keep_ind = np.where(table[:,0]!=0)[0]
    table = np.take(table, keep_ind, axis=0)
    
    peak_XY = table[:,1:]
    twicetheta, chi = Lgeo.calc_uflab(peak_XY[:,0], peak_XY[:,1], [79.554,979.45,932.54,0.387,0.44],
                                        returnAngles=1,
                                        pixelsize=dictLT.dict_CCD["sCMOS"][1],
                                        kf_direction='Z>0')
    
    ## check Gnomonic function
    TwiceTheta_Chi = twicetheta, chi
    Xgno,Ygno = ComputeGnomon_2(TwiceTheta_Chi, CenterProjection=(45 * DEG, 0 * DEG))
    twicetheta_back, chi_back = InverseGnomon(Xgno,Ygno)
    
    assert np.all(np.round(chi_back,4)==np.round(chi,4))
    assert np.all(np.round(twicetheta_back,4)==np.round(twicetheta,4))
    
    imgGN, _, _ = computeGnomonicImage(twicetheta, chi)
    
    fig, ax = plt.subplots()
    plt.imshow(imgGN,cmap='gray')
    plt.savefig("prefix.png", bbox_inches='tight',format='png', dpi=1000)

    
    accumulator, thetas, rhos1, rhos = fast_hough_line(imgGN, angle_step=1, 
                                                       lines_are_white=True, value_threshold=0.1)
    peaks = hough_peaks(accumulator, 20, thetas, rhos)
    lines_images = hough_lines_draw(imgGN, peaks, rhos1)
    
    show_hough_line(imgGN, accumulator, lines_images, peaks, thetas, rhos, rhos1, save_path=None)
