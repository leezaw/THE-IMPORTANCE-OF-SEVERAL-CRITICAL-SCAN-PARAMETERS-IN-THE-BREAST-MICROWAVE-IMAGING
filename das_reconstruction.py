# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:16:01 2023

@author: leena
"""
import sys

dire0=str('C:/Users/leena/OneDrive/Desktop/OpenAndPlot')
sys.path.insert(1,dire0)

import numpy as np
import matplotlib.pyplot as plt
from IDFT import idft2
from backproj import back_proj
import umbms
from umbms.beamform.recon import das, dmas
from umbms.beamform.fdfuncs import fd_das_mp
from umbms.beamform.multscat import get_ms_pix_ts, get_ms_phase_factor
from umbms.loadsave import load_birrs_txt
from umbms.beamform.recon import das
from umbms.iqms.accuracy import get_loc_err

dire0=str('C:/Users/leena/OneDrive/Desktop/scan_2/1kHz')
# C:/Users/leena/OneDrive/Desktop/Radius Experiment/18cm

imagerecon = True
           
filesave = '/S11_sphere_x0cm_y0cm'
# filesave = '/S11_sphere_x0mm_y2.5mm'
#filesave = '/S11_sphere_x0mm_y5mm'
# filesave = '/S11_sphere_x2.5mm_y0mm'
# filesave = '/S11_sphere_x5mm_y0mm'

filesavename = str(filesave + '.txt')

openname = '/S11_sphere_open.txt'
s11 = load_birrs_txt(str(dire0+filesavename))
s11o = load_birrs_txt(str(dire0+openname))

bandwidth = False
def extract_tumor_position(filesave):
    """Extract tumor_x and tumor_y based on the filesave name."""
    # Extract x and y positions from the filename
    parts = filesave.split('_')
    x_part = parts[2].replace('x', '')
    y_part = parts[3].replace('y', '')
    
    # Convert to meters (assuming the positions are in cm)
    tumor_x = float(x_part.replace('cm', '')) / 100
    tumor_y = float(y_part.replace('cm', '')) / 100
    
    return tumor_x, tumor_y

ang_pts = 72 #angular positions
f_pts =  np.size(s11o,0) # 1001 #frequency points
maxf = 9 #GHz

t_pts =100 #time points
max_t = 3#nanoseconds

freqs = np.linspace(2e9,9e9,f_pts)

if bandwidth:
    startindex = 0
    stopindex = 200

    freqs = freqs[startindex:stopindex]
    s11 = s11[startindex:stopindex,:]


m_size = 125
phasecorrect=False
timedelay = 2*0.184e-9
#timedelay = 0
timePoints = np.linspace(0,max_t*1e-9,t_pts)

air_perm = 0 #1.0    #Reflectivity or Permativity of air
roi_rad = 0.21+0.024     #Radius of the region of interest
speed = 3e8       #Propogartion speed through air
n_ant_pos =   72  #Number of antenna positions
ini_ant_ang = -140+180

if __name__ == '__main__':
   
    p_ts= get_ms_pix_ts(roi_rad, m_size, roi_rad, speed, n_ant_pos, ini_ant_ang)
    #p_d = p_d*128/0.2 #converts to pixels instead of meters
    phaFac = get_ms_phase_factor(p_ts)
   
   
    freqs_array = np.repeat(freqs[:,None],ang_pts,1)
   
    phase_delays = 1j*timedelay*freqs_array*2*np.pi
    s11 = s11*np.exp(phase_delays)
    s11o = s11o*np.exp(phase_delays)
   
    s11_rod = s11-s11o
   
    sino = idft2(s11_rod, freqs, timePoints,0, phasecorrect)  
    sino_open = idft2(s11o, freqs, timePoints,0, phasecorrect)  
   
    x,y=extract_tumor_position(filesave)
    print(x,y)
    # plt.figure()
    # plt.imshow(np.abs(sino),cmap = 'magma',aspect=(50/t_pts))
    # plt.xticks([0,ang_pts/2,ang_pts],[0,180,360])
    # plt.yticks([0,t_pts/4,t_pts/2,3*t_pts/4,t_pts],[0,max_t/4,max_t/2,3*max_t/4,max_t])
    # plt.title('Rod Time-Domain')
    # plt.ylabel('Time (ns)')
    # plt.colorbar()
    # plt.xlabel('Angle (Degrees)')
    # # plt.savefig(str(dire0 + filesave + '.jpg'))
    # plt.show()
   
    # plt.figure()
    # plt.imshow(np.abs(sino_open),cmap = 'magma',aspect=(50/t_pts))
    # plt.xticks([0,ang_pts/2,ang_pts],[0,180,360])
    # plt.yticks([0,t_pts/2,t_pts],[0,max_t/2,max_t])
    # plt.title('Rod Time-Domain')
    # plt.ylabel('Time (ns)')
    # plt.colorbar()
    # plt.xlabel('Angle (Degrees)')
    # # plt.savefig(str(dire0 + filesave + '.jpg'))
    # plt.show()
   
   
    if imagerecon:
        I0 = fd_das_mp(s11_rod, phaFac, 1, freqs,2)
           
        # I1 = das(sino, 0, max_t*1e-9,roi_rad,speed,m_size,ini_ant_ang)    
       
        x = np.linspace(-10,10,m_size)
        x = np.repeat(x[None,:],m_size,0)
        y = np.transpose(-x)
        cond = x**2 + y**2 > 10**2
        # cond1 = (x+1)**2 + (y+5)**2 < 1.5**2    
        I0[cond] = 0
        # I1[cond] = 0
   
       
        plt.figure()
        plt.imshow(abs(I0),cmap = 'magma')
        plt.xticks([0,m_size/2,m_size],[-10,0,10])
        plt.yticks([0,m_size/2,m_size],[10,0,-10])
        plt.title('Abs')
        plt.ylabel('y (cm)')
        plt.colorbar()
        plt.xlabel('x (cm)')
        # plt.savefig(str(dire0 + filesave + '.jpg'))
        plt.show()
        
        tumor_x, tumor_y = extract_tumor_position(filesave)
        loc_err = get_loc_err(I0, roi_rad, tumor_x, tumor_y)
        print(filesave)
        print(loc_err)
   
    # plt.figure()
    # plt.imshow(abs(np.abs(I1)),cmap = 'magma')
    # plt.xticks([0,m_size/2,m_size],[-10,0,10])
    # plt.yticks([0,m_size/2,m_size],[10,0,-10])
    # plt.title('Abs')
    # plt.ylabel('y (cm)')
    # plt.colorbar()
    # plt.xlabel('x (cm)')
    # plt.show()

    # xx = abs(sino_open[:,0])
    # max_index = np.array(np.where(xx == np.max(xx) ))
    # print(timePoints[max_index])
            
    # x_pos = np.round(x[max_index[0],max_index[1]]*10,2)
    # y_pos = np.round(y[max_index[0],max_index[1]]*10,2)
    # print(str('[' + str(x_pos) + ' mm' + ', ' + str(y_pos) + ' mm'))
    # bandwidth = float(freq_dir_name.rstrip('kHz'))
    # print(bandwidth)
    
