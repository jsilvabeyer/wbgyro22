from scipy.optimize import brentq
#import sys
import numpy as np

def bv_to_bprp(bv):
    '''Inputs B-V color, returns Gaia G_BP - G_RP color
    
    Parameters from Jordi et al. (2010)'''
    return 0.0981 + bv * 1.4290 - bv**2 * 0.0269 + bv**3 * 0.0061

def bprp_to_bv(color):
    """Input Gaia G_BP - G_RP color, returns B-V color. 
    Input must be a numpy array!"""
    def function(x):
        return bv_to_bprp(x) - color[i]
    bv = []
    for i in range(color.shape[0]):
        bv_row = brentq(function, -1e5, 1e5)
        bv.append(bv_row)
        print('\r{:3.3f}%'.format(i / color.shape[0] * 100), end='')
        #sys.stdout.flush()
    print('\nGaia colors converted to B-V!')
    return np.array(bv)

def bpl_model(bprp, age, br=.43):
    '''Inputs Gaia G_BP-G_RP color and Age (in yr), returns P_rot (in days)
    Inputs can be either both arrays of same size, bprp an array and age a float, or both floats
    
    Function from Angus et al. (2019), adapted for this paper.
    '''
    if type(bprp) == float or type(bprp) == np.float64:
        bprp = np.array([bprp])
    logbprp = np.log10(bprp)
    logage = np.log10(age)
    p = [-38.96093821359645, 28.71101008315462, -4.91903824593666, 0.7160561986000329, -
         4.716546365981507, 0.6470642123725334, -13.55890448246179, 0.9361999874258322]
    logp = np.zeros(len(logbprp))
    cool = logbprp >= br
    hot = logbprp < -.25
    warm = (logbprp > -.25) & (logbprp <= br)
    if type(logage) != np.ndarray:
        logp[warm] = np.polyval(p[:5], logbprp[warm]) + p[5] * logage
        logp[cool] = np.polyval(p[6:], logbprp[cool]) + p[5] * logage
    else:
        logp[warm] = np.polyval(p[:5], logbprp[warm]) + p[5] * logage[warm]
        logp[cool] = np.polyval(p[6:], logbprp[cool]) + p[5] * logage[cool]
    return 10**logp

def isochrone2015(BV, age): 
    '''Inputs B-V color and Age (in Myr), returns P_rot (in days)
    
    Equation from Angus et al. (2015)'''
    return age**0.55 * 0.4 * (BV - 0.45)**0.31

def solve_isochrone(bprp_1, prot_1, iso='A19'):
    '''Inputs Gaia color G_BP - G_RP and Prot (in days), returns gyrochone age (in yr for A19, in Myr for A15)
        All inputs must be numpy arrays of the same size!
    iso (default \'A19\'): str, can be either \'A19\' for the Angus et al (2019) relation or  \'A15\' for the Angus et al (2015) relation.'''
    if iso == 'A19':
        def function(age):
            return bpl_model(bprp_1[i], age) - prot_1[i]
        age_1 = []
        for i in range(bprp_1.shape[0]):
            if prot_1[i] <= 0 or bprp_1[i] <= 10**-0.25:
                print('Cant find a gyrochrone for the {}-th pair! \n'.format(i))
                age_1.append(np.nan)
                print('\r{:3.3f}%'.format(i / bprp_1.shape[0] * 100), end='')
                continue
            res = brentq(function, 1e-10, 1e18)
            age_1.append(res)
            print('\r{:3.3f}%'.format(i / bprp_1.shape[0] * 100), end='')
        age_1 = np.array(age_1)
        return age_1

    elif iso=='A15':
        bv_1 = bprp_to_bv(bprp_1)
        age_1 = []
        def function(x):
            return isochrone2015(bv_1[i], x) - prot_1[i]
        for i in range(bv_1.shape[0]):
            if prot_1[i] <= 0 or bprp_1[i] <= 10**-0.25:
                print('Cant find a gyrochrone for the {}-th pair! \n'.format(i))
                age_1.append(np.nan)
                print('\r{:3.3f}%'.format(i / bv_1.shape[0] * 100), end='')
                continue
            res = brentq(function, 1e-10, 1e12)
            age_1.append(res)
            print('\r{:3.3f}%'.format(i / bv_1.shape[0] * 100), end='')
        age_1 = np.array(age_1)
        return age_1
    
def gyro_check(bprp_1, prot_1, bprp_2, prot_2, iso = 'A19'):
    '''Inputs Gaia color G_BP - G_RP and Prot (in days) from both components of a pair, returns DeltaProtGyro (in days)
    All inputs must be numpy arrays of the same size!
    iso (default \'A19\'): str, can be either \'A19\' for the Angus et al (2019) relation or  \'A15\' for the Angus et al (2015) relation.'''
    if iso == 'A19':
        age_1 = solve_isochrone(bprp_1, prot_1, iso=iso)
        delta_prot = prot_2 - bpl_model(bprp_2, age_1)
        ret = []
        for i in delta_prot:
            ret.append(i)
        return np.array(ret)
        
    elif iso=='A15':
        bv_2 = bprp_to_bv(bprp_2)
        age_1 = solve_isochrone(bprp_1, prot_1, iso=iso)
        delta_prot = prot_2 - isochrone2015(bv_2, age_1)
        ret = []
        for i in delta_prot:
            if not np.isnan(i):
                ret.append(float(i))
            else:
                ret.append(np.nan)
        return np.array([float(i) for i in ret])
    
def age_check(bprp_1, prot_1, bprp_2, prot_2, iso = 'A19'):
    ''' Inputs Gaia color G_BP - G_RP and Prot (in days) from both components of the pair, returns tuple with ages of the primary and secondary (in yr).
    All inputs must be numpy arrays of the same size!
    iso (default \'A19\'): str, can be either \'A19\' for the Angus et al (2019) relation or  \'A15\' for the Angus et al (2015) relation.'''
    if iso=='A15':
        age_1 = solve_isochrone(bprp_1, prot_1, iso=iso)
        age_2 = solve_isochrone(bprp_2, prot_2, iso=iso)
        return age_1*1e6, age_2*1e6
    elif iso=='A19':
        age_1 = solve_isochrone(bprp_1, prot_1, iso=iso)
        age_2 = solve_isochrone(bprp_2, prot_2, iso=iso)
        return age_1, age_2

def sigma_Delta_P_rot_gyro(bprp_1, prot_1, e_prot_1, bprp_2, e_prot_2, iso='A19'):
    sigma_prot_2_catalog = prop_error_iso(bprp_1, prot_1, e_prot_1, bprp_2, iso=iso)
    sigma_delta = np.sqrt(sigma_prot_2_catalog**2 + e_prot_2**2)
    return sigma_delta

def prop_error_iso(BPRP_1, prot_1, e_prot_1, BPRP_2, iso='A19'): # A15 for Angus et al. (2015) lines, A19 for Angus et al. (2019)
    '''Inputs Gaia color G_BP - G_RP, Prot (in days) and its error (in days), returns error propagated across the gyrochrone (in days)
        All inputs must be numpy arrays of the same size!
    iso (default \'A19\'): str, can be either \'A19\' for the Angus et al (2019) relation or  \'A15\' for the Angus et al (2015) relation.'''

    if iso == 'A15':
        age_plus = solve_isochrone(BPRP_1, prot_1 + e_prot_1, iso='A15')
        age_minus = solve_isochrone(BPRP_1, prot_1 - e_prot_1, iso='A15')
        BV_1 = bprp_to_bv(BPRP_1)
        BV_2 = bprp_to_bv(BPRP_2)
        prot_plus = isochrone2015(BV_2, age_plus)
        prot_minus = isochrone2015(BV_2, age_minus)
        return (prot_plus - prot_minus) / 2
    elif iso == 'A19':
        age_plus = solve_isochrone(BPRP_1, prot_1 + e_prot_1, iso='A19')
        age_minus = solve_isochrone(BPRP_1, prot_1 - e_prot_1, iso='A19')
        prot_plus = bpl_model(BPRP_2, age_plus)
        prot_minus = bpl_model(BPRP_2, age_minus)
        return (prot_plus - prot_minus) / 2
    

def solve_x(bprp_1, prot_1, e_prot_1, bprp_2, prot_2, e_prot_2, iso='A19'):
    '''Inputs Gaia color G_BP - G_RP, Prot (in days) and its error (in days), for each component of the pair, returns the pairs x-parameter
        All inputs must be numpy arrays of the same size!
    iso (default \'A19\'): str, can be either \'A19\' for the Angus et al (2019) relation or  \'A15\' for the Angus et al (2015) relation.'''

    check_catalog = gyro_check(bprp_1, prot_1, bprp_2, prot_2, iso=iso)
    sigma_delta = sigma_Delta_P_rot_gyro(bprp_1,prot_1, e_prot_1, bprp_2, e_prot_2, iso=iso)
    return abs(check_catalog / sigma_delta)