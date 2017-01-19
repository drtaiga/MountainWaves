#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 17:55:05 2017

@author: vaw
"""

import copy as cp
import numpy as np
#import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ipywidgets as ipw
from IPython.display import display


class mountain_wave_3d:
    
    def __init__(self, input_params):
        
        self.input = cp.deepcopy(input_params) 
        
    def run(self):
        
        print("Running.")
        
        self.init()
        self.set_grid()
        self.set_topography()
        self.vertical_wavenumbers()
        
        print("Generating wave field ...")
        self.wvel()
        print("Generating precipitation distribution ...")
        self.precip()        
        print("Done.")
        print(" ")
        
    def init(self):
        
        self.rho0 = 1.29
        self.grav = 9.81
        self.scorersq_ul = (self.input.N_u/self.input.U_u)**2.0
        self.scorersq_ll = (self.input.N_l/self.input.U_l)**2.0
        
    def set_grid(self):
        
        self.x=np.linspace(-self.input.Lx/2.,self.input.Lx/2.,self.input.nx)
        self.y=np.linspace(-self.input.Ly/2.,self.input.Ly/2.,self.input.ny)
        self.z=np.linspace(0,self.input.H,self.input.nz)
        
        self.XY,self.YX=np.meshgrid(self.x,self.y)
        self.XZ,self.ZX=np.meshgrid(self.x,self.z)
        self.YZ,self.ZY=np.meshgrid(self.y,self.z)
        
        self.precipitation=np.zeros((self.input.nx,self.input.ny))
        
        
        self.h_3d = np.zeros((self.input.nz,self.input.nx,self.input.ny))
        self.p_3d = np.zeros((self.input.nz,self.input.nx,self.input.ny))
        self.u_3d = np.zeros((self.input.nz,self.input.nx,self.input.ny))
        self.v_3d = np.zeros((self.input.nz,self.input.nx,self.input.ny))
        self.w_3d = np.zeros((self.input.nz,self.input.nx,self.input.ny))
        self.d_3d = np.zeros((self.input.nz,self.input.nx,self.input.ny))

        
    def set_topography(self):
        
        self.hump_3d = self.input.h * \
            np.exp(-(self.XY**2+self.YX**2)/self.input.a**2)
        self.FFT_hump_3d= np.fft.fft2(self.hump_3d)
        self.wn_k=np.fft.fftfreq(self.XY.shape[-1],d=self.input.Lx/self.x.size)
        self.wn_l=np.fft.fftfreq(self.YX.shape[-1],d=self.input.Ly/self.y.size)
        
    def precip(self):
        
        m=self.wn_m_ul
        U=self.input.U_l
        k=self.wn_k
        
        sigma=U*k
        
        FFT_hump_3d=self.FFT_hump_3d
        
        precip_fft=self.input.Cw * 1j * sigma * FFT_hump_3d \
                    / (1.0-1j*m*self.input.Hw) \
                    / ((1.0+1j*sigma*self.input.tauc)*(1.0+1j*sigma*self.input.tauf))
                    
        self.precipitation=np.fft.ifft2(precip_fft).real
        self.precipitation[self.precipitation < 0.0] = 0.0
        
        #print(self.precipitation)
        
    def wvel(self):
    
        eps=1.0e-12     #A small parameter for preventing divide-by-zero errors
        REFL_PARAM=0.7
        m1=self.wn_m_ll
        m2=self.wn_m_ul
        k=self.wn_k
        l=self.wn_l  
        sig_sq=k**2.0/(k**2.0 + l**2.0 + eps)
        HLL=self.input.HLL
        U_l=self.input.U_l
        U_u=self.input.U_u
        N_l=self.input.N_l
        N_u=self.input.N_u
        
        rho0=self.rho0
        FFT_hump_3d=self.FFT_hump_3d
        
        amat=np.zeros((3,3))
        bvec=np.zeros((3))
        
            
        A=((U_l*m1+U_u*m2)*FFT_hump_3d*np.exp(-1j*m1*HLL)) / \
            ( U_l*m1*(np.exp(-1j*m1*HLL)+np.exp(1j*m1*HLL)) \
            - U_u*m2*(np.exp(1j*m1*HLL)-np.exp(-1j*m1*HLL))+eps)
            
        B=((U_l*m1-U_u*m2)*FFT_hump_3d*np.exp(1j*m1*HLL)) / \
            ( U_l*m1*(np.exp(-1j*m1*HLL)+np.exp(1j*m1*HLL)) \
            - U_u*m2*(np.exp(1j*m1*HLL)-np.exp(-1j*m1*HLL))+eps)
            
        C=(2.0*U_l*m1*FFT_hump_3d*np.exp(-1j*m2*HLL)) / \
            ( U_l*m1*(np.exp(-1j*m1*HLL)+np.exp(1j*m1*HLL)) \
            - U_u*m2*(np.exp(1j*m1*HLL)-np.exp(-1j*m1*HLL))+eps)
        
        
        for i in range(self.input.nz):          
            if self.z[i] <= HLL:
                hfft=A*np.exp(1j*m1*self.z[i])+REFL_PARAM*B*np.exp(-1j*m1*self.z[i])
                pfft=rho0*U_l*sig_sq*1j*m1*(A*np.exp(1j*m1*self.z[i])-
                            B*np.exp(-1j*m1*self.z[i]))
                ufft=-1.0/rho0/U_l * pfft
                
                wfft=U_l * 1j * k * hfft
                dfft=-rho0/self.grav*N_l**2.0 * hfft
                vfft=-(k/(l+eps)) * ufft - (k/(l+eps)) * U_l*1j*m1 * \
                    (A*np.exp(1j*m1*self.z[i])-B*np.exp(-1j*m1*self.z[i]))
                
                self.h_3d[i,:,:]=np.fft.ifft2(hfft).real/1000.
                self.p_3d[i,:,:]=np.fft.ifft2(pfft).real
                self.u_3d[i,:,:]=np.fft.ifft2(ufft).real
                self.v_3d[i,:,:]=np.fft.ifft2(vfft).real
                self.w_3d[i,:,:]=np.fft.ifft2(wfft).real 
                self.d_3d[i,:,:]=np.fft.ifft2(dfft).real 
                

                
        
            else:
                
                hfft=C*np.exp(1j*m2*self.z[i])
                pfft=rho0*U_u*sig_sq*1j*m2*C*np.exp(1j*m2*self.z[i])
                ufft=-1.0/rho0/U_u * pfft
                wfft=U_u * 1j * k * hfft
                dfft=-rho0/self.grav*N_u**2.0 * hfft
                vfft=-(k/(l+eps)) * ufft - (k/(l+eps)) * U_u*1j*m2 * \
                      C*np.exp(1j*m2*self.z[i])
                
                self.h_3d[i,:,:]=np.fft.ifft2(hfft).real/1000.
                self.p_3d[i,:,:]=np.fft.ifft2(pfft).real
                self.u_3d[i,:,:]=np.fft.ifft2(ufft).real
                self.v_3d[i,:,:]=np.fft.ifft2(vfft).real
                self.w_3d[i,:,:]=np.fft.ifft2(wfft).real 
                self.d_3d[i,:,:]=np.fft.ifft2(dfft).real 
                
            
        
    def vertical_wavenumbers(self):
    
        self.wn_m_ul=np.zeros((self.wn_k.size,self.wn_l.size),dtype=np.complex)
        self.wn_m_ll=np.zeros((self.wn_k.size,self.wn_l.size),dtype=np.complex)
    
        for i in range(0,self.wn_k.size):
    
            if self.wn_k[i]**2 >= self.scorersq_ul:
                self.wn_m_ll[:,i]= \
                np.lib.scimath.sqrt(((self.wn_k[i]**2+self.wn_l[:]**2)/self.wn_k[i]**2) \
                    *(self.scorersq_ll-self.wn_k[i]**2))
                self.wn_m_ul[:,i]=1j* \
                    np.lib.scimath.sqrt( ((self.wn_k[i]**2+self.wn_l[:]**2)/self.wn_k[i]**2) \
                    *(self.wn_k[i]**2-self.scorersq_ul) )
            elif self.wn_k[i] == 0:
                self.wn_m_ul[:,i]=0.
                self.wn_m_ll[:,i]=0.
            else:
                self.wn_m_ll[:,i]= \
                    np.lib.scimath.sqrt( ((self.wn_k[i]**2+self.wn_l[:]**2)/self.wn_k[i]**2) \
                    *(self.scorersq_ll-self.wn_k[i]**2))
                self.wn_m_ul[:,i]=np.sign(self.wn_k[i])* \
                    np.lib.scimath.sqrt((self.wn_k[i]**2+self.wn_l[:]**2)/(self.wn_k[i])**2  \
                    *(self.scorersq_ul-self.wn_k[i]**2))
 
    def xzplot(self,x,z,dat,yc):
        
        ycut=yc*1000.
        idx = (np.abs(self.y-ycut)).argmin()
        ykm=self.y[idx]/1000.   
        #plt.figure()
        #cs_w=plt.pcolormesh(x/1000,z/1000,dat[:,idx,:],vmin=np.max(dat),vmax=np.min(dat),cmap=cm.RdGy_r)
        cs_w=plt.pcolormesh(x/1000,z/1000,dat[:,idx,:],cmap=cm.RdGy_r)
        plt.fill_between(self.x/1000.,self.hump_3d[:,idx]/1000.,color='r')
        cbar = plt.colorbar(cs_w)
        cbar.set_label('w (m/s)', rotation=90)
        plt.xlim(-10,30)
        plt.ylim(0,50)
        plt.title("X-Z section @ Y=" + '%.1f' % ykm + " km")
        plt.xlabel("Zonal Distance (km)")
        plt.ylabel("Altitude (km)")
        #plt.show()
        
    def xyplot(self,x,y,dat,zc):
        
        zcut=zc*1000.
        idx = (np.abs(self.z-zcut)).argmin()
        zkm=self.z[idx]/1000.   
        #plt.figure()
        cs_w=plt.pcolormesh(x/1000,y/1000,dat[idx,:,:],
                vmin=np.min(dat),vmax=np.max(dat),cmap=cm.RdGy_r)
        cbar = plt.colorbar(cs_w)
        cbar.set_label('w (m/s)', rotation=90)
        plt.xlim(-10,30)
        plt.ylim(-20,20)
        plt.title("X-Y section @ Z=" + '%.2f' % zkm + " km")
        plt.xlabel("Zonal Distance (km)")
        plt.ylabel("Meridional Distance (km)")
        #plt.show()
        
    def yzplot(self,y,z,dat,xc):
        
        xcut=xc*1000.
        idx = (np.abs(self.x-xcut)).argmin()
        xkm=self.x[idx]/1000.   
        #plt.figure()
        cs_w=plt.pcolormesh(y/1000,z/1000,dat[:,:,idx],
                vmin=np.min(dat),vmax=np.max(dat),cmap=cm.RdGy)
        plt.fill_between(self.y/1000,self.hump_3d[idx,:]/1000.,color='r')
        cbar = plt.colorbar(cs_w)
        cbar.set_label('w (m/s)', rotation=90)
        plt.xlim(-20,20)
        plt.ylim(0,50)
        plt.title("Y-Z section @ X=" + '%.2f' % xkm + " km")
        plt.ylabel("Altitude (km)")
        plt.xlabel("Meridional Distance (km)")
        #plt.show()
                    
class mountain_wave_1d:
    
    def __init__(self, input_params):
        
        self.input = cp.deepcopy(input_params)
        
    def run(self):
        
        self.init()
        self.set_grid()
        self.set_topography()
        self.set_velocity_profile()
        self.set_theta_profile()
        self.calc_w()
        
    def init(self):
        
        self.rho0 = 1.29
        self.grav = 9.81
        self.scorersq_ul = (self.input.N_u/self.input.U_u)**2.0
        self.scorersq_ll = (self.input.N_l/self.input.U_l)**2.0
        
    def set_grid(self):
        
        self.x = np.linspace(-self.input.Lx/2,self.input.Lx/2,self.input.nx)
        self.z = np.linspace(0,self.input.H,self.input.nz)
        self.wn_m_ul = np.zeros(self.input.nx,dtype=np.complex)
        self.wn_m_ll = np.zeros(self.input.nx,dtype=np.complex)
        self.w = np.zeros((self.input.nz, self.input.nx))
        
    def set_topography(self):
        
        #self.hump_1d=self.input.h/(((self.x)/self.input.a)**2+1.0)**1.5
        self.hump_1d=self.input.h*np.exp(-self.x**2.0/self.input.a**2.0)
        self.F_hump_1d = np.fft.fft(self.hump_1d)
        self.wn_k=np.fft.fftfreq(self.x.shape[-1],d=self.input.Lx/self.input.nx)
         
    def calc_w(self):
        
        #self.w_lo(self.input.HLL)
        #self.w_hi(self.input.HLL)
        
        for jj in range(0,self.input.nz):
            if self.z[jj] <= self.input.HLL:
                self.w_lo(self.z[jj])
                self.w[jj,:] = self.w_ll
            else:
                self.w_hi(self.z[jj])
                self.w[jj,:] = self.w_ul
        
    def w_lo(self,z_height):
        
        eps=1.0e-12
        
        m1 = np.lib.scimath.sqrt(self.scorersq_ll - self.wn_k**2) 
        
        m2 = np.where(self.wn_k**2 >= self.scorersq_ul,
            1j * np.lib.scimath.sqrt(self.wn_k**2.0 - self.scorersq_ul),
            np.sign(self.wn_k.real) * 
                np.lib.scimath.sqrt(self.scorersq_ul-self.wn_k**2))
        
        w1 = 1j * self.wn_k * self.F_hump_1d * self.input.U_l * \
            (m1+m2) * np.exp(-1j*m1*self.input.HLL) / \
            (m1*(np.exp(1j*m1*self.input.HLL) + np.exp(-1j*m1*self.input.HLL))- \
             m2*(np.exp(1j*m1*self.input.HLL)-np.exp(-1j*m1*self.input.HLL))+eps)
        
        w2 = 1j * self.wn_k * self.F_hump_1d * self.input.U_l * \
            (m1-m2)*np.exp(1j*m1* self.input.HLL) / \
            (m1*(np.exp(1j*m1*self.input.HLL)+np.exp(-1j*m1*self.input.HLL))- \
             m2*(np.exp(1j*m1*self.input.HLL)-np.exp(-1j*m1*self.input.HLL))+eps)
            
        wll_fft=w1*np.exp(1j*m1*z_height) + w2*np.exp(-1j*m1*z_height)
    
        
        self.w_ll = np.fft.ifft(wll_fft).real
        
        #print(self.w_ll)
        
    def w_hi(self,z_height):
        
        eps = 1.0e-12
        
        m1 = np.lib.scimath.sqrt(self.scorersq_ll - self.wn_k**2)
        m2 = np.where(self.wn_k**2 >= self.scorersq_ul, 
            1j * np.lib.scimath.sqrt(self.wn_k**2.0 - self.scorersq_ul),
            np.sign(self.wn_k.real) * 
                np.lib.scimath.sqrt(self.scorersq_ul- self.wn_k**2))
        
        w3 = (2.0 * m1 * (1j * self.wn_k * self.F_hump_1d * self.input.U_l *
            np.exp(-1j*m2*self.input.HLL))) / \
            (m1 * (np.exp(1j*m1*self.input.HLL) + np.exp(-1j*m1*self.input.HLL)) - 
                 m2 * (np.exp(1j*m1*self.input.HLL) - np.exp(-1j*m1*self.input.HLL))+eps)
        
        wul_fft = w3 * np.exp(1j*m2*z_height)
        
        self.w_ul = np.fft.ifft(wul_fft).real 
        
    def set_velocity_profile(self):
        
        self.vel_prof = np.zeros(self.input.nz)
        
        self.vel_prof = np.where(self.z <= self.input.HLL,
            self.input.U_l,
            self.input.U_u )
        
    def set_theta_profile(self):
        
        self.theta_prof = np.zeros(self.input.nz)
        
        self.theta_prof = np.where(self.z <= self.input.HLL,
            self.input.theta0 * np.exp(self.input.N_l**2.0 * self.z/self.grav),
            self.input.theta0 * np.exp(self.input.N_l**2.0 * self.input.HLL/self.grav) 
                * np.exp(self.input.N_u**2.0 * (self.z - self.input.HLL)/self.grav) )
        
        
class input_params_1d:
    
    def __init__(self):
        
        #Initialize the parameters with their default values
        
        self.h      =   2.0e3   # Mountain height (m)
        self.a      =   2.0e3   # Mountain width parameter(m)
        self.U_l    =   5.0e0   # Wind velocity in lower layer (m/s)
        self.U_u    =   5.0e0   # Wind velocity in upper layer (m/s)
        self.N_l    =   0.0e-3  # Brunt-Vaisala frequency in lower layer (1/s)
        self.N_u    =   0.0e-3  # Brunt-Vaisala frequency in upper layer (1/s)
        self.theta0 =   298.15  # Potential temperature at surface (K)
        
        self.Lx     =   1000.e3     
        self.Ly     =   1000.e3
        self.H      =   50.0e3
        self.HLL    =   5.0e3
        
        self.nx     =   256*5
        self.ny     =   256*5
        self.nz     =   300+1
        
class input_params_3d:
    
    def __init__(self):
        
        #Initialize the parameters with their default values
        
        self.h      =   5.0e3   # Mountain height (m)
        self.a      =   3.0e3   # Mountain width parameter(m)
        self.U_l    =   5.0e0   # Wind velocity in lower layer (m/s)
        self.U_u    =   5.0e0   # Wind velocity in upper layer (m/s)
        self.N_l    =   0.0e-3  # Brunt-Vaisala frequency in lower layer (1/s)
        self.N_u    =   0.0e-3  # Brunt-Vaisala frequency in upper layer (1/s)
        self.theta0 =   298.15  # Potential temperature at surface (K)
        
        self.Lx     =   200.e3     
        self.Ly     =   200.e3
        self.H      =   50.0e3
        self.HLL    =   5.0e3
        
        self.nx     =   256*4
        self.ny     =   256*4
        self.nz     =   200+1
        
        self.Cw     =   0.02
        self.Hw     =   4500.
        self.tauc   =   0.
        self.tauf   =   1000.

        
        

class comparison_plot(object):
    
    def __init__(self,xy,yx,xz,zx,yz,zy,
                 x,y,z,dat1,dat2,dat3,
                 xc,yc,zc,
                 h,a):
        
        self.xy = xy
        self.yx = yx
        self.xz = xz
        self.zx = zx
        self.yz = yz
        self.zy = zy
        
        self.x = x
        self.y = y
        self.z = z
        
        self.dat1 = dat1
        self.dat2 = dat2
        self.dat3 = dat3
        
        self.xc = xc
        self.yc = yc
        self.zc = zc
        
        self.h = h
        self.a = a
           
        self.w_min = np.min([np.min(dat1),np.min(dat2),np.min(dat3)])
        self.w_max = np.max([np.max(dat1),np.max(dat2),np.max(dat3)])
        self.hump_3d = self.h * \
            np.exp(-(self.xy**2+self.yx**2)/self.a**2)
        
        
        self.xcut_slider = ipw.FloatSlider(
            min=-10.0,max=30.0,step=0.2,
            value=5.0,width=100,description='x_cut (km)',
            continuous_update=False
            )
        self.ycut_slider = ipw.FloatSlider(
            min=-20.0,max=20.0,step=0.2,
            value=0.0,width=100,description='y_cut (km)',
            continuous_update=False
            )
        self.zcut_slider = ipw.FloatSlider(
            min=0.0,max=20.0,step=0.2,
            value=5.0,width=100,description='z_cut (km)',
            continuous_update=False
            )
        
        self.widget_box = ipw.HBox([
                self.xcut_slider, self.ycut_slider, self.zcut_slider],
                background_color='#EEE')
        
  
    def do_plot(self):
        
        display(self.widget_box)
        self.make_plot()
        self.zcut_slider.observe(self.redo_zcut)
        self.ycut_slider.observe(self.redo_ycut)
        self.xcut_slider.observe(self.redo_xcut)
        
               
    def make_plot(self):
        
       
        
        #self.fig = plt.figure()        
        plt.clf()
        
        plt.subplot(331)
        self.xyplot(self.dat1)
        plt.subplot(332)
        self.xyplot(self.dat2)
        plt.subplot(333)
        self.xyplot(self.dat3)
        
        plt.subplot(334)
        self.xzplot(self.dat1)
        plt.subplot(335)
        self.xzplot(self.dat2)
        plt.subplot(336)
        self.xzplot(self.dat3)
        
        plt.subplot(337)
        self.yzplot(self.dat1)
        plt.subplot(338)
        self.yzplot(self.dat2)
        plt.subplot(339)
        self.yzplot(self.dat3)
        
        #plt.show()
        
        
    def redo_zcut(self, change):
        
        self.zc = self.zcut_slider.value
        self.make_plot()
    
    def redo_ycut(self, change):
        
        self.yc = self.ycut_slider.value
        self.make_plot()
        
    def redo_xcut(self, change):
        
        self.xc = self.xcut_slider.value
        self.make_plot()
        
        
        
#    def xyplot(x,y,zvec,dat,xc,yc,zc,w_min,w_max):
    def xyplot(self,dat):
        
        zcut=self.zc*1000.
        idx = (np.abs(self.z-zcut)).argmin()
        zkm=self.z[idx]/1000.   
        #plt.figure()
        cs_w=plt.pcolormesh(self.xy/1000,self.yx/1000,dat[idx,:,:],
                vmin=self.w_min,vmax=self.w_max,cmap=cm.RdGy_r)
        cbar = plt.colorbar(cs_w)
        #cbar.set_label('w (m/s)', rotation=90)
        plt.xlim(-10,30)
        plt.ylim(-20,20)
        plt.axvline(self.xc,linestyle=':',color='b',linewidth=1.5)
        plt.axhline(self.yc,linestyle=':',color='g',linewidth=1.5)
        plt.text(10,-17.5,'xy: z=' + '%.1f' % self.zc + " km", color='r',size="small" )
        #plt.title("X-Y section @ Z=" + '%.2f' % zkm + " km")
        #plt.xlabel("Zonal Distance (km)")
        #plt.ylabel("Meridional Distance (km)")
        
    def xzplot(self,dat):
        
        ycut=self.yc*1000.
        idx = (np.abs(self.y-ycut)).argmin()
        ykm=self.y[idx]/1000.   
        #plt.figure()
        cs_w=plt.pcolormesh(self.xz/1000,self.zx/1000,dat[:,idx,:],
                vmin=self.w_min,vmax=self.w_max,cmap=cm.RdGy_r)
        plt.fill_between(self.x/1000.,self.hump_3d[:,idx]/1000.,color='r')
        #plt.fill_between(self.x/1000.,self.hump_3d[:,idx]/1000.,color='r')
        cbar = plt.colorbar(cs_w)
        #cbar.set_label('w (m/s)', rotation=90)
        plt.xlim(-10,30)
        plt.ylim(0,20)
        plt.axhline(self.zc,linestyle=':',color='r',linewidth=1.5)
        plt.axvline(self.xc,linestyle=':',color='b',linewidth=1.5)
        plt.text(10,1.0,'xz: y=' + '%.1f' % self.yc + " km", color='g',size="small" )
        #plt.title("X-Z section @ Y=" + '%.1f' % ykm + " km")
        #plt.xlabel("Zonal Distance (km)")
        #plt.ylabel("Altitude (km)")
        #plt.show()
        
    def yzplot(self,dat):
        
        xcut=self.xc*1000.
        idx = (np.abs(self.x-xcut)).argmin()
        xkm=self.x[idx]/1000.   
        #plt.figure()
        cs_w=plt.pcolormesh(self.yz/1000,self.zy/1000,dat[:,:,idx],
                vmin=self.w_min,vmax=self.w_max,cmap=cm.RdGy_r)
        plt.fill_between(self.y/1000.,self.hump_3d[idx,:]/1000.,color='r')
        cbar = plt.colorbar(cs_w)
        #cbar.set_label('w (m/s)', rotation=90)
        plt.xlim(-20,20)
        plt.ylim(0,20)
        plt.axhline(self.zc,linestyle=':',color='r',linewidth=1.5)
        plt.axvline(self.yc,linestyle=':',color='g',linewidth=1.5)
        plt.text(1.0,1.0,'yz: x=' + '%.1f' % xkm + " km", color='b',size="small" )
        #plt.title("Y-Z section @ X=" + '%.2f' % xkm + " km")
        #plt.ylabel("Altitude (km)")
        #plt.xlabel("Meridional Distance (km)")
        

    
     