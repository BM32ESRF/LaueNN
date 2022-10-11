# -*- coding: utf-8 -*-
"""
Created on June 18 06:54:04 2021

routine for XRF EDF to ascii conversion

@author: Ravi raj purohit PURUSHOTTAM RAJ PUROHIT (purushot@esrf.fr)
"""
from __future__ import division, print_function, absolute_import
import glob, re, os
import numpy as np
from PyMca5.PyMcaIO.EdfFile import EdfFile
import matplotlib.pyplot as plt
from scipy import optimize

# =============================================================================
# USER INPUT SPACE
# =============================================================================
plot_data = True

folder = r"E:\vijaya_lauedata\EDX"
prefix = r"HS261120b_SH2_S5_B__xia00_0001_0000_"
file_format = r"edf"

save_folder = folder + "//" + "results"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
# =============================================================================
# END OF USER INPUT SPACE
# =============================================================================

### Lets scan the folder for all the required files
list_of_files = glob.glob(folder+"//"+ prefix + '*.'+file_format)
list_of_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

all_data = []
for file in list_of_files:
    ## Read the EDF file
    edf = EdfFile(file, 'rb')
    header = edf.GetHeader(0)
    data = edf.GetData(0).T
    all_data.append(data)

all_data = np.array(all_data)

## lets reorganize the data into our desired format
new_data = np.zeros((all_data.shape[1], all_data.shape[0]*all_data.shape[2]))

for ijk in range(all_data.shape[0]):
    new_index = all_data.shape[2] * ijk
    new_index1 = all_data.shape[2] * (ijk+1)
    new_data[:,new_index:new_index1] = all_data[ijk, :, :]

image_x, image_y = all_data.shape[0], all_data.shape[2]
## write data to single dat file
file_header = "# Column Row0 Row1 ..... RowN " +"Image dimension "+str(all_data.shape[0])+" x "+str(all_data.shape[2])
output_file = save_folder + "//"+ prefix + ".dat"
with open(output_file,'w') as fout:
    fout.write(file_header + "\n")
    for i in range(new_data.shape[0]):
        strout = str(i)
        for j in range(new_data.shape[1]):
            strout = strout +'\t{0}'.format(new_data[i,j])
        strout = strout + "\n"
        fout.write(strout)
fout.close()

if plot_data:
    def _gaussian(x, amp1, cen1, sigma1):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))
    
    def _2gaussian(x, amp1, cen1, sigma1, amp2, cen2, sigma2):
         return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + \
                amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen2)/sigma2)**2)))  
    
    def _3gaussian(x, amp1, cen1, sigma1, amp2, cen2, sigma2, amp3, cen3, sigma3):
         return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + \
                amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen2)/sigma2)**2))) + \
                amp3*(1/(sigma3*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen3)/sigma3)**2)))
                
    def _4gaussian(x, amp1, cen1, sigma1, amp2, cen2, sigma2, amp3, cen3, sigma3, amp4, cen4, sigma4):
         return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + \
                amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen2)/sigma2)**2))) + \
                amp3*(1/(sigma3*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen3)/sigma3)**2))) + \
                amp4*(1/(sigma4*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen4)/sigma4)**2))) 
        
    p_guess=[20000,820-20,7, \
              10000,880-20,7, \
              9000,900-20,7, \
              900,980-20,7]      #peak4  Amp, mean, sigma
    
    # for filename in listScan: #loop on each file
    fileout=save_folder + "//"+ prefix + ".out"
    with open(fileout,'w') as fout: #write header in fileout
        strout='#Scan\tCu_Zn\tA1\tA2\tA3\tA4\tChi2\n'; fout.write(strout)  #\t : tab and \n newline
    fout.close()
    
    Data = new_data
    Cu_Zn=np.zeros(Data.shape[1])
    for i in range(Data.shape[1]):        
        N=len(Data[:,[i]])
        y=Data[:,[i]].reshape(1,Data.shape[0]).flatten()
        x=range(N)
        
        if y.max() > 200.0:

            popt, pcov = optimize.curve_fit(_4gaussian, x, y, p0=p_guess)
            perr = np.sqrt(np.diag(pcov)) #error
            chi2 = np.sum((y- _4gaussian(x,*popt)) ** 2)
            chi2dof = chi2 / N
            pars_1 = popt[0:3]
            pars_2 = popt[3:6]
            pars_3 = popt[6:9]
            pars_4 = popt[9:12]
            peak_1 = _gaussian(x, *pars_1)
            peak_2 = _gaussian(x, *pars_2)
            peak_3 = _gaussian(x, *pars_3)
            peak_4 = _gaussian(x, *pars_4)
            Cu_Zn[i-1]=(pars_1[0]+pars_3[0])/(pars_2[0]+pars_4[0])
            
            # #indiv. graph out BETTER TO NOT PLOT THESE DATA UNLESS DEBUGING
            # fig, ax = plt.subplots()
            # plt.scatter(x,y)
            # plt.plot(x,peak_1,color='red')
            # plt.plot(x,peak_2,color='blue')
            # plt.plot(x,peak_3,color='green')
            # plt.plot(x,peak_4,color='black')
            # plt.xlim(500,1200)
            # plt.show()
            
        else:
            pars_1 = [0]
            pars_2 = [0]
            pars_3 = [0]
            pars_4 = [0]
            Cu_Zn[i-1]=chi2dof=0.
            
        with open(fileout,'a+') as fout: #write results in fileout
            strout=str(i)+'\t{0:2.2f} \t{1:4.2f} \t{2:4.2f} \t{3:4.2f}  \t{4:4.2f} \t{5:4.2f}\n'.format(Cu_Zn[i-1],pars_1[0],pars_2[0],pars_3[0],pars_4[0],chi2dof)
            fout.write(strout)
        fout.close()
        print(str(i),' Cu/Zn=',Cu_Zn[i-1],' Amp_=',pars_1[0],pars_2[0],pars_3[0],pars_4[0])

    Matx=Cu_Zn.reshape(image_x,image_y) #S327 is a 21x21 dmesh cryst S2 #S339
    fig, ax = plt.subplots() #instance a new figure with axis (ax) properties
    ax.set_aspect('equal', 'box') #iso axis 
    plt.contour(Matx,15,linewidths=0.5,colors='k') #line contour (15)
    plt.contourf(Matx,30,cmap=plt.cm.jet)           #filled false color
    cbar=plt.colorbar() # draw colorbar
    cbar.ax.set_xlabel('Cu/Zn')
    plt.title(prefix)
    plt.savefig(save_folder + "//"+ prefix +'-Map'+'.png',transparent='True', bbox_inches='tight',format='png', dpi=1000)
    plt.close(fig)