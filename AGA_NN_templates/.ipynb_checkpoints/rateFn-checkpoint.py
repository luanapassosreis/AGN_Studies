from numba import jit
from constants import *
from scipy.integrate import quad
from numba import jit
from scipy import special
import numpy as np
import os
import math
#from tqdm import tqdm


def ProjectCat_to_Xs(EXs, E_tab, Cat_tab):
    
    Xs_Cat = np.zeros_like(EXs) + 1.
    
    
    for i in range(len(EXs)):
        if( (EXs[i]> E_tab[0]) and (EXs[i]<E_tab[-1]) ):
            ix = int( np.log10(EXs[i]/E_tab[0]) / np.log10(E_tab[1]/E_tab[0]) )
            Xs_Cat[i] = Cat_tab[ix]

    
    return Xs_Cat




def Project_to_Xs(Ecomp, ELEcomp, EXs):
    
    Xs_ELEcomp = np.zeros_like(EXs) + small
    
    for i in range(len(EXs)):
        if( (EXs[i]>= Ecomp[0]) and (EXs[i]<=Ecomp[-1]) ):
            ix = int( np.log10(EXs[i]/Ecomp[0]) / np.log10(Ecomp[1]/Ecomp[0]) )
            Xs_ELEcomp[i] = ELEcomp[ix]
    
    return Xs_ELEcomp




def UniLogEGrid(Emin,Emax,DlE):
    '''
    This function return a vector of energies values 
    uniform spaced in log10(E/eV)
    '''
    lE = np.arange(np.log10(Emin/eV), np.log10(Emax/eV), DlE)
    E = 10**lE * eV
    
    return E



def xsecBH(eps):
    '''
    Paramerisation of Bethe-Heitler cross section
    according to Y. G. Zheng , C. Y. Yang and S. J. Kang, A&A 585, A8 (2016)
    
    Z = 1 ##(only protons)
    if( (eps>=2) and (eps<=4) ):
        f1 = ( (eps - 2)/eps )**3
        
        eeta = (eps - 2) / (eps + 2)
        f2 = 1 + 0.5*eeta + (23/40)*eeta**2 + (37/120)*eeta**3 + (61/92)*eeta**4  
        
        return 2*np.pi / 3 * alpha_fine * re**2 * Z**2 * f1 * f2  
    '''
    Z = 1
    if( eps>4 ):
        z3 = 1.20206
        g1 =  (28/9)* np.log(2*eps)
        g2 = - (218/27)
        g3 = (2/eps)**2
        g4 = 6*np.log(2*eps) - 7/2 + 2/3*( np.log(2*eps) )**3 - (np.log(2*eps) )**2 - (1/3)*np.pi**2*np.log(2*eps) + 2*z3 + np.pi**2 /6
        g5 = (2/eps)**4 * ( 3/16 * np.log(2*eps) + 1/8 )
        g6 = - (2/eps)**6 * (9/9/256 * np.log(2*eps) - 77/27/512 )
        
        return alpha_fine * re**2 * Z**2# * ( g1 + g2 + g3*g4 + g5 + g6 )
   
    else:
    
#if( (eps>=2) and (eps<=4) ):
        f1 = ( (eps - 2)/eps )**3
        
        eeta = (eps - 2) / (eps + 2)
        f2 = 1 + 0.5*eeta + (23/40)*eeta**2 + (37/120)*eeta**3 + (61/92)*eeta**4  
        
        return 2*np.pi / 3 * alpha_fine * re**2 * Z**2 #* f1 * f2      
    
    
        return 1.


#@jit(nopython=True)
def EmaxBal(E,t_acc,t_bal):
    F = t_acc - t_bal
    Emax = E[-1]
    for i in range(len(E)-1):
        if( F[i]*F[i+1]<0 ):
            Emax = 10**( 0.5* ( np.log10(E[i]/eV) + np.log10(E[i+1]/eV) )   ) * eV
            break
    
    return Emax




def Pp_pion_tot(Ep,n):
    '''
    This function returns the total energy loss rate in erg s^-1
    by one CR proton of energy Ep due to pion creation by p-p interactions
     in a environment of density n.
    The funtion is taken from Yoast-Hull et al. (2013) , which uses the 
    formula derived in Schlickeisser et al. (2002)
    '''
    gamma =  Ep / mpc2
    #beta  = ( 1 - 1/gamma**2 )**.5
    y = gamma - 1.3
   # fn = 1.82e-7 * eV * n * ( 1 + .0185 * np.log(beta) * np.heaviside(y,0.5)) 
   # fb = 2*beta**2 / ( 1e-6 + 2 * beta**3 )
    
    return 1.31e-7 * n * gamma**(1.28) *  np.heaviside(y,0.5) * eV




def EeSynch(Eg,B):
    
    return (4/3*np.pi * me*c/qe/h * (mec2)**2 * Eg / B)**0.5






def Pp_ion_tot(Ep,n):
    '''
    This function returns the total energy loss rate in erg s^-1
    by one CR proton of energy Ep due to ionisation, in a environment of density n.
    The funtion is taken from Peretti et al. 2019, which uses the 
    formula derived in Schlickeisser et al. (???)
    '''
    gam =  Ep / mpc2
    beta  = ( 1 - gam**(-2) )**(1/2)
    y = beta - 0.01
    fn = 1.82e-7*eV * n * ( 1 + .0185 * np.log(beta) * np.heaviside(y,0.5)) 
    fb = 2*beta**2 / ( 1e-6 + 2 * beta**3 )
    
    return fn*fb


def Pp_Cou_tot(Ep,ne,Te):
    '''
    This function returns the total energy loss rate in erg s^-1
    by one CR proton of energy Ep due to Coulomb interactions in a environment of density ne.
    The funtion is taken from Peretti et al. 2019, which uses the 
    formula derived in Schlickeisser et al. (???)
    '''
    gamma =  Ep / mpc2
    beta  = ( 1 - 1/gamma**2 )**.5
    y = beta - 7.4e-4 * (Te / 2e6)**(1/2)
    fr = 3.08e-7*eV * ne * beta**2 / (beta**3 + 2.34e-5 * (Te / 2e6)**(1.5) ) 
    fh = np.heaviside(y,0.5) 
    
    return fr*fh



def Ppsyn_tot(Ep,B):
    '''
    equivalent to equation (5) of Romero et al. 2010
    '''
    
    UB = B**2/(8*np.pi)
    return 4/3 * (me / mp)**2 * sigmaT * c * UB * (Ep/mpc2)**2



def Psyn_tot(Ee,B):
    '''
    equivalent to equation (5) of Romero et al. 2010
    '''
    
    UB = B**2/(8*np.pi)
    return 4/3 * sigmaT * c * UB * (Ee/mec2)**2




def PIC_tot(Ee,Uph):
    return 4/3 * sigmaT * c * Uph * (Ee/mec2)**2


def PBr_tot(Ee,n):
    gamma_e = Ee / mec2
    return 8*qe**6*n / mec2/hbar * (np.log(gamma_e)+0.36) *(gamma_e + 1.)

def Pe_Cou_tot(Ee,Te,ne):
    Te_eV = kB * Te / eV
    lamb = 24 - np.log( (ne)**.5 / Te_eV )
    gamma_e =  Ee / mec2
    beta_e  = ( 1 - 1/gamma_e**2 )**.5
    Be = lamb * ne / beta_e
    
    #x =  me * ( beta_e * c )**2 / (2 * kB * Te)
    #Ce = np.zeros_like(x)
    #for i in range(np.size(x)):
    #    Ce[i] = Psi(x[i]) - dPsi(x[i])
    #    print('    Integral of eCoulomb cooling Ce = %.8f '%(Ce[i]),end='')
    #    print('\r', end='')        
    #print('\n')
    return 4 * np.pi * qe**4 /me / c * Be #* Ce 



def dPsi(x):
    return 2/np.sqrt(np.pi) * x**.5 * np.exp(-x)
    
def Psi(xx):
    if (xx>1e4):
        xx = 1e4
    def Integrand(x):
        return x**.5 * np.exp(-x) 
    
    def Integral(x):
        return 2/np.sqrt(np.pi) * quad(Integrand,0,xx)[0]
    
    return np.vectorize(Integral)(xx)




def Uphf(eps,nph):
    deps = np.diff(eps)
    val=0.
    for i in range(len(deps)):
        val = val + nph[i] * eps[i] *deps[i]
    return val




def nph_bb(eps,T): 
    '''  erg^-1 cm^-3  '''
    return np.pi / (c*h*eps) * Bnu(eps,T)




def Bnu(nu,T):
    return 2*h*nu**3/c**2 / (np.exp(h*nu/kB/T)-1.)


def Eps_ff(n,T,eps):
    nu = eps / h
    return  6.8e-38 * n**2 * T**(-1/2)* np.exp( - h * nu / kB / T )
            

def  IntOm_Inu(phi,theta,I_nu):
    '''
    Gives the integral of the specific intensity I_nu,
    over phi and theta, asuming homogeneous I_nu
    '''
    return phi * (1. - np.cos(theta)) * I_nu






def SumCurvLog(lx,ly,lxc,lyc):
    '''
    the X domain of the base lxB must be larger that the lx domain of the component
    returns log10(S), where S is the sum of y and yc ('y base' and 'y component')
    '''
    NB = len(lx)
    Nc = len(lxc)
    S = np.zeros(NB)
    y = 10**ly
    yc_prime = np.zeros(NB)
    y = 10**ly
    Y = 0.
    M = 0.
    B = 0.
    for i in range(NB):
        for j in range((Nc-1)):
            if( ( lx[i] >= lxc[j] ) & (lx[i] <= lxc[j+1] ) ):
                M = (lyc[j+1] - lyc[j]) / (lxc[j+1] - lxc[j]) 
                B = lyc[j] - M * lxc[j]
                Y = M * lx[i] + B
                yc_prime[i] = 10**Y 
                S[i] = y[i] + yc_prime[i]
                break
            
            elif(lx[i] > lxc[Nc-1]):
                S[i] = y[i]
            elif(lx[i] < lxc[0]):
                S[i] = y[i]
    
    
    return np.log10(S)



def SumCurvLogImp(lx,ly,lxc,lyc):
    
    NB = len(lx)
    Nc = len(lxc)
    S = np.zeros(NB)+ small
    y = 10**ly
    yc_prime = np.zeros(NB)
    y = 10**ly
    Y = 0.
    M = 0.
    B = 0.
    for i in range(NB):
        for j in range((Nc-1)):
            if( ( lx[i] >= lxc[j] ) & (lx[i] < lxc[j+1] ) ):
                M = (lyc[j+1] - lyc[j]) / (lxc[j+1] - lxc[j]) 
                B = lyc[j] - M * lxc[j]
                Y = M * lx[i] + B
                yc_prime[i] = 10**Y 
                S[i] = y[i] + yc_prime[i]
                
                break
            
            if(lxc[j]>lx[NB-1]):
                break
                        
    
    return np.log10(S)






@jit(nopython=True)
def tau_gg(Eg, eps,nph, Rph):
    deps = np.diff(eps)
    Inphsigmagg = np.zeros(len(Eg))
    for j in range(len(Eg)):
        for i in range(len(deps)):
            Inphsigmagg[j] = Inphsigmagg[j] + deps[i] * nph[i] *sigma_gg(eps[i], Eg[j])
    
    return Rph * Inphsigmagg



@jit(nopython=True)    
def sigma_gg(eps, Eg):
    ''' gives the pair production cross section'''
    s = eps * Eg
    if (s>mec2**2 and (s<(1e27*eV**2) )):
        beta = (1. - mec2**2/s)**(0.5)
        A = 1.-beta**2
        B = (3 - beta**4) * np.log((1. + beta)/(1. - beta)) + 2.*beta*(beta**2-2.)
        return .5 * np.pi * re**2 * A * B
    else:
        return 0.




def EpMax(Ep,n,B,Vs,D0,Rdiff,eps,nph,Dt_out_m1):
    F = rate_accKolg(Ep,Vs,B,D0) - rate_diffKolg(Ep,B,Rdiff,D0) - rate_pp(Ep,4*n) #- rate_pg(Ep,eps,nph)# - Dt_out_m1
    count = 0
    EMAX = 1e15*eV
    for i in range( np.size(Ep)-1):
        if( (F[i]*F[i+1]) < 0. ):
            count = count + 1   
            avlE = ( np.log10(Ep[i]) + np.log10(Ep[i+1]) ) / 2.
            EMAX = 10**avlE
            print('\n  Epmax, root number %d, found'%count)
            print('\n  tEpmax = %.3f yr'%( rate_accKolg(EMAX,Vs,B,D0)**(-1) / yr   )  )
    return  EMAX




def Bmin(Epcut,n,B,Vs,D0,Rdiff,eps,nph):
    F = rate_accKolg(Epcut,Vs,B,D0) - rate_diffKolg(Epcut,B,Rdiff,D0) #- rate_pp(Epcut,4*n) #- rate_pg(Ep,eps,nph)# - Dt_out_m1
    count = 0
    BMIN = 10.
    for i in range( np.size(B)-1):
        #print('B = %1.3e, tacc = %1.3e, tdiff = %1.3e'%(B[i],rate_accKolg(Epcut,Vs,B[i],D0)**(-1) /yr, #rate_diffKolg(Epcut,B[i],Rdiff,D0)**(-1)/yr  ) )
        if( (F[i]*F[i+1]) < 0. ):
            count = count + 1   
            avlB = ( np.log10(B[i]) + np.log10(B[i+1]) ) / 2.  
            BMIN = 10**avlB
            print('\n  Bmin, root number %d, found'%count)
            print('\n  tEpmax = %.3f yr'%( rate_accKolg(Epcut,Vs,BMIN,D0)**(-1) / yr   )  )
    return  BMIN


def rate_accKolg(Ep,Vs,B,D0):
    '''
    We consider this to be and intermediate value between paaralle and oblique 
    shocks. See discussion in Protheroe R. J., 1999, in Duvernois M. A., ed., Vol. 230.
    '''    
    return Vs**2/ DKolg(D0,Ep,B)


def rate_diffKolg(Ep,B,Rdiff,D0):
    '''
    taken from Romero et al. (2010) A&A, 519
    '''
    return 2 * DKolg(D0,Ep,B)/ (Rdiff)**2


def DKolg(D0,Ep,B):
    
    B0=1
    E0=1.22*GeV
    
    return D0 * (Ep/E0)**(1/3) * (B/B0)**(-1/3)



def rL(m,E,q,B):
    '''
    gives the gyroradius as rL (larmor radius) as a function of:
    m: particle mass
    E: particle toral energy (kinetic + rest mass)
    q: particle charge
    B: background magnetic field
    '''
    g = E / (m*c**2)
    
    return ( g**2 - 1 )**(1/2) * m * c**2 / q / B



    
def rate_pp(Ep,n):
    Kpp = 0.5
    def sigmapp(Ep):
        '''
        This function gives the energy dependent cross section
        for proton-proton interactions.
        '''
        L = np.log(Ep/TeV)
        return (34.3 + 1.88*L + 0.25*L**2) * (1. - (Eth/Ep)**4)**2 * mbarn  
    
    return Kpp * sigmapp(Ep) * c * n
    
    
def rate_acc(Ep,Vs,B):
    
    return (3/20) * (Vs / c)**2 * qe * c * B / Ep






def rate_diff(Ep,B,Rdiff):
    '''
    taken from Romero et al. (2010) A&A, 519
    '''
    return 2./3. * Ep * c /qe /B / (Rdiff)**2




def rate_pg(Ep, eps, nph):
    
    Eth_pg = 145 * 1e6 * eV
    k1 = 0.2
    s1 = 340 * 1e-6 * barn
    k2 = 0.6
    s2 = 120 * 1e-6 * barn
    C1 = -.5*k1*s1*(200*MeV)**2
    C2 = -.5*k2*s2*(500*MeV)**2 + .5*k1*s1*(500*MeV)**2 - .5*k1*s1*(200*MeV)**2
    
    def II(eps,Ep):
        
        if( (2*eps*Ep /mpc2) < (200*MeV) ):
            return 0.
        
        elif( ( (200*MeV)<=(2*eps*Ep /mpc2) ) & ( (2*eps*Ep /mpc2)<(500*MeV) ) ):
            return k1*s1/2 *(2*eps*Ep/mpc2)**2 + C1
        
        elif( (2*eps*Ep/mpc2) > 500*MeV ):
            return k2*s2/2 *(2*eps*Ep/mpc2)**2 + C2
    
    SSum = np.zeros(np.size(Ep))
    deps = np.diff(eps)

    for j in range(np.size(Ep)):
        for i in range(np.size(deps)):
            if(eps[i] > (Eth_pg * mpc2 / 2 / Ep[j]) ):
                SSum[j] = SSum[j] + mp**2 * c**5 / 2 / Ep[j]**2 * deps[i] * nph[i] / eps[i]**2 * II(eps[i], Ep[j])                    
    return  SSum
    
    
    
    
@jit(nopython=True)
def rate_pg_cool(Ep, eps, nph):
    '''
    This cool function differs from the rate_pg_coll function
    in the K1, K1 factors. The rate_pg_coll function consider
    K1=K2 =1.
    '''
    
    gp = Ep / mpc2  
    deps = np.diff(eps)
    dleps = np.log10( eps[1] / eps[0] )
    Ieps = np.zeros_like(gp) + small
    epsth = 145*MeV 
    
    sigma1 = 340 * 1e-6*barn
    K1 = 0.2
    sigma2 = 120 * 1e-6*barn
    K2 = 0.6
    
    
    def fI(eps, gp):
        x = 2* eps * gp
        def I1(x):
            return sigma1*K1 * 0.5* x**2
        def I2(x):
            return sigma2*K2 * 0.5* x**2
        
        if( x<(200*MeV) ):
            return 0.
        elif( (x>= 200*MeV) and (x<500*MeV)  ):
            return I1(x) - I1(200*MeV)
        elif( x>= 500*MeV ):
            return I1(500*MeV) - I1(200*MeV) + I2(x) - I2(500*MeV)
    
    for i in range(len(gp)):
        eps_inf = epsth / (2*gp[i] )
        
        if( (eps_inf >= eps[0] ) and ( eps_inf < eps[len(deps)] ) ):
            j0 = int( math.ceil( np.log10( eps_inf / eps[0] ) / dleps ) )
            for j in range(j0,len(deps)):
                Ieps[i] = Ieps[i] + deps[j]*nph[j] / (eps[j]**2) * fI(eps[j], gp[i])
    
    return c / (2 * gp**2) * Ieps 
    
    
    
    
#@jit(nopython=True)
def rate_pg_coll(Ep, eps, nph):
    
    gp = Ep / mpc2
    deps = np.diff(eps)
    dleps = np.log10( eps[1] / eps[0] )
    Ieps = np.zeros_like(gp) + small
    epsth = 145*MeV 
    
    sigma1 = 340 * 1e-6*barn
    K1 = 1.
    sigma2 = 120 * 1e-6*barn
    K2 = 1.
    
    
    def fI(eps, gp):
        x = 2* eps * gp
        def I1(x):
            return sigma1*K1 * 0.5* x**2
        def I2(x):
            return sigma2*K2 * 0.5* x**2
        
        if( x<(200*MeV) ):
            return 0.
        elif( (x>= 200*MeV) and (x<500*MeV)  ):
            return I1(x) - I1(200*MeV)
        elif( x>= 500*MeV ):
            return I1(500*MeV) - I1(200*MeV) + I2(x) - I2(500*MeV)
    
    for i in range(len(gp)):
        eps_inf = epsth / (2*gp[i] )
        
        if( (eps_inf >= eps[0] ) and ( eps_inf < eps[len(deps)-1] ) ):
            j0 = int( math.ceil( np.log10( eps_inf / eps[0] ) / dleps ) )
            for j in range(j0,len(deps)):
                Ieps[i] = Ieps[i] + deps[j]*nph[j] / (eps[j]**2) * fI(eps[j], gp[i])
    
    return c / (2 * gp**2) * Ieps 


    





def tm1_pg(Ep, eps, nph):
    
    Eth_pg = 145 * 1e6 * eV
    k1 = 0.2
    s1 = 340 * 1e-6 * barn
    k2 = 0.6
    s2 = 120 * 1e-6 * barn
    C1 = -.5*k1*s1*(200*MeV)**2
    C2 = -.5*k2*s2*(500*MeV)**2 + .5*k1*s1*(500*MeV)**2 - .5*k1*s1*(200*MeV)**2
    
    def II(eps,Ep):
        
        if( (2*eps*Ep /mpc2) < (200*MeV) ):
            return 0.
        
        elif( ((200*MeV)<=(2*eps*Ep /mpc2)) & ((2*eps*Ep /mpc2)<(500*MeV)) ):
            return k1*s1/2 *(2*eps*Ep/mpc2)**2 + C1
        
        elif( (2*eps*Ep/mpc2) > 500*MeV):
            return k2*s2/2 *(2*eps*Ep/mpc2)**2 + C2
    
    SSum = 0.
    deps = np.diff(eps)

    for i in range(np.size(deps)):
        if(eps[i] > (Eth_pg * mpc2 / 2 / Ep ) ):
            SSum = SSum + mp**2 * c**5 / 2 / Ep**2 * deps[i] * nph[i] / eps[i]**2 * II(eps[i], Ep)                    
    
    return  SSum









    
def rate_IC(Ee,eps,nph):
    
    def F(eps,nph,Ee):
        Gamma = 4. * eps * Ee / mec2**2
        e1min = eps
        e1max = Gamma * Ee / (1. + Gamma)
        dle1 = 0.01
        le1 = np.arange(np.log10(e1min/eV), np.log10(e1max/eV), dle1) #log(E/eV)
        e1 = 10**le1 * eV
        de1 = np.diff(e1)
        
        CC =  2 * np.pi * re**2 * me**2 * c**5 
        
        Sum = 0.
        for j in range(np.size(de1)):        
            q = e1[j]/(Gamma*(Ee-e1[j]))
            Fq = 2.*q* np.log(q) + (1+2*q)*(1-q) +.5*(1-q)*(Gamma*q)**2/(1+Gamma*q)
            Sum = Sum + de1[j] * (e1[j] - eps) * CC / Ee**2 * nph / eps * Fq

        return Sum
    
    deps = np.diff(eps)
    SSum=0.
    
    tm1 =  np.zeros_like(Ee)
    SSum = np.zeros_like(Ee)
    print('\n\n Calculating IC rate (accounting for KN regime):\n')
    for j in range(np.size(tm1)):
        for i in range(np.size(deps)):
            SSum[j] = SSum[j] + F(eps[i],nph[i],Ee[j]) * deps[i] 
            #print('eps = %1.3e eV'%(eps/eV))
    
    
    tm1 = (1. / Ee) * SSum
    
    return tm1



@jit(nopython=True)
def PIC_KN_tot(Ee,eps,nph):
    
    def F(eps,nph,Ee):
        Gamma = 4. * eps * Ee / mec2**2
        e1min = eps
        e1max = Gamma * Ee / (1. + Gamma)
        dle1 = 0.05
        le1 = np.arange(np.log10(e1min/eV), np.log10(e1max/eV), dle1) #log(E/eV)
        e1 = 10**le1 * eV
        de1 = np.diff(e1)
        
        CC =  2 * np.pi * re**2 * me**2 * c**5 
        
        Sum = 0.
        for j in range(len(de1)):        
            q = e1[j]/(Gamma*(Ee-e1[j]))
            Fq = 2.*q* np.log(q) + (1+2*q)*(1-q) +.5*(1-q)*(Gamma*q)**2/(1+Gamma*q)
            Sum = Sum + de1[j] * (e1[j] - eps) * CC / Ee**2 * nph / eps * Fq

        return Sum
    
    deps = np.diff(eps)
    tm1 =  np.zeros_like(Ee)
    SSum = np.zeros_like(Ee)
    #print('\n\n Calculating IC rate (accounting for KN regime):')
    for j in range(len(tm1)):
        for i in range(len(deps)):
            SSum[j] = SSum[j] + F(eps[i],nph[i],Ee[j]) * deps[i] 
        #print('    PIC( Ee=%1.3e eV )'%(Ee[j]/eV), end='')
        #print('\r',end='')
    #print('')    
    
    
    return SSum






def sigma_pp(Ep):
    '''
    This function gives the energy dependent cross section
    for proton-proton interactions.
    '''
    L = np.log(Ep/TeV)
    return (34.3 + 1.88*L + 0.25*L**2) * (1. - (Eth/Ep)**4)**2 * mbarn   
    
    
    
    
def Poly_Interp(x,y):
    '''
    Polynomial interpolation of a data set pair x, y
    x, and y hava a size of N elements.
    This function then returns a vector A, which components are the 
    coefficients of a polynomium of degre N-1, and which interpolate 
    date x, y.
    '''    
    N = np.size(x)
    Matrix = np.empty([N,N])
    
    #print(M)    
    for i in range(N):
        for j in range(N):        
            Matrix[i][j] = x[i]**j
            
    ###vector of cieficients
    A = np.linalg.solve(Matrix,y) 
    
    return A


    
    
def Inorm_nth(q,E0,Ecut):
    '''
    This function perform the integral need to normalise the 
    P-L with exponential cutoff function:
    Q = Q0 * (E/E0)^(-q) * exp(- E / Emax) with a 
    luminosity function
    '''
    suma = 0.
    Einf = 100 * Ecut
    dlE = 0.001
    lE = np.arange( np.log10(E0/eV), np.log10(Einf/eV) , dlE )
    E = 10**lE * eV
    dE = np.diff(E)
    
    for i in range(np.size(dE)):
        
        suma = suma + dE[i] * E[i] * (E[i] / E0)**(-q) * np.exp(- E[i] / Ecut)
        
    return suma
        
    
def IL(E0,Ecut,s):
    '''
    '''
    suma = 0.
    Einf = 1000 * Ecut
    dlE = 0.001
    lE = np.arange( np.log10(E0/eV), np.log10(Einf/eV) , dlE )
    E = 10**lE * eV
    dE = np.diff(E)
    
    for i in range(np.size(dE)):
        
        suma = suma + dE[i] *  (E[i])**(s) * np.exp(- (E[i] / Ecut)**1)
        
    return suma
        
def L_Edd(M):
    
    return 1.26e38 * (M / Msun)     
    
    
def dotM_acc(M, dotm, etad):
    
    return dotm * 1.26e38 * (M / Msun) / etad / c**2
