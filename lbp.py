import numpy as np
import cv2 as cv
import skimage as ski

def generate_lbp(img):
    img = np.float32(img)
    ax = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ax = ax/255
    ax = ski.util.img_as_int(ax)
    #ax = ax[:,:]
    aox = cv.resize(ax, (227, 681))
    a1 = aox
    a11 = a1*(np.exp(-(1j)*(np.pi)))
    a22 = np.zeros((700, 700))
    a22[351-340:351+341,351-113:351+114] = a11

    #Propagation by angular spectrum method:
    #lengths are in mm
    
    dx = 20*(10**(-3))
    dy = dx
    xpix = 227
    ypix = 681
    f = 250
    lmda = 0.638*(10**(-3))
    rf = 2*np.pi/(lmda*f)
    centx = 0.5*xpix
    centy = 0.5*ypix
    dz1 = 100
    
    nx = np.arange(-centx, centx, 1)
    ny = np.arange(centy, -centy, -1)
    [m, n] = np.meshgrid(nx, ny)
    kz = rf*np.sqrt(f**2-((m*dx)**2)-((n*dy)**2))
    
    pf1 = np.exp(-(1j)*kz*dz1)
    F=np.fft.fftshift(np.fft.fft2(a11))
    u1=np.fft.ifft2(np.fft.fftshift(F*pf1))

    #Ground glass to produce speckle

    dx =20e-3
    dy=dx
    lx=20
    ly=lx
    
    nx1 = np.arange(-0.5*lx, 0.5*lx, dx)
    ny1 = np.arange(0.5*ly, -0.5*ly, -dy)
    [m1, n1] = np.meshgrid(nx1, ny1)
    (wmax, smax) = (len(m1), len(m1))
    
    #Generation of scatter
    
    rho_m = 0.0020
    m2 = np.square(m1)
    n2 = np.square(n1)
    a_g2 = np.exp(-(m2 + n2)/rho_m**2)
    k_g2 = np.random.rand(wmax,smax)-0.5
    c_a2 = (np.fft.fft2(k_g2))
    h2=(np.fft.fft2(a_g2))
    c_d2=(np.fft.ifft2(c_a2*np.real(h2)))
    c2=np.exp(1j*10*np.pi*(np.real(c_d2))/np.max(np.real(np.matrix.flatten(c_d2))))
    c2=c2[0:681,0:227]
    hl= c2*u1

    #Propagation by angular spectrum method:
    #lengths are in mm
    
    dx1 = (lmda*f)/(xpix*dx)
    dy1 = dx1
    xpix = 227
    ypix = 681
    f = 250
    lmda = 0.638*(10**(-3))
    rf = 2*np.pi/(lmda*f)
    centx = 0.5*xpix
    centy = 0.5*ypix
    dz2 = 300
    
    nx = np.arange(-centx, centx, 1)
    ny = np.arange(centy, -centy, -1)
    [m, n] = np.meshgrid(nx, ny)
    kz = rf*np.sqrt(f**2-((m*dx1)**2)-((n*dy1)**2))
    
    pf1 = np.exp(-(1j)*kz*dz2)
    F=np.fft.fftshift(np.fft.fft2(hl))
    u2=np.fft.ifft2(np.fft.fftshift(F*pf1))
    u3 = np.abs(u2)
    I = np.square(u3)
    
    lbp = ski.feature.local_binary_pattern(I, 8, 8)
    lbp = lbp.reshape(227, 227, 3)

    return lbp