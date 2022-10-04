'''
Class and function updated to cil functionality
'''

import numpy as np
import math

from cil.framework import BlockDataContainer as  bdc
from cil.framework import BlockGeometry, ImageGeometry
from cil.optimisation.operators import GradientOperator, LinearOperator
from cil.optimisation.functions import Function

from numba import jit, prange

####### Classes #######

def SVD_decomp(mat):
    raise NotImplementedError()


### TNV functions ##
class TNV(Function):
    def __init__(self):
        super(Function, self).__init__()  
        
    def __call__(self,x):
        raise NotImplementedError()
    
    def proximal(self,x, tau):
        raise NotImplementedError()
    
    def convex_conjugate(self, x):
        raise NotImplementedError()
    
    def proximal_conjugate(self, x, tau):
        raise NotImplementedError()

### operator giveing 4x1 bdc from grad-like operation on 2x1 images ### Now uneeded after correcting use of cil BlockOperators
class MSoperator(LinearOperator):
    def __init__(self, operator=None):
        
        if operator is not None:
            self.set_up(operator)
            
        super(MSoperator, self).__init__(domain_geometry=operator.domain_geometry, 
                                       range_geometry=bdc(operator.domain_geometry(),operator.domain_geometry(),operator.domain_geometry(),operator.domain_geometry(),)) 
            
    def set_up(self, operator):
        self.operator = operator
            
    def direct(self,x, out=None):
        tmp0 = self.operator.direct(x[0])
        tmp1 = self.operator.direct(x[1])
        if out is None:
            return bdc(tmp0[0],tmp0[1],tmp1[0],tmp1[1],shape=(4,1))
        else:
            out = bdc(tmp0[0],tmp0[1],tmp1[0],tmp1[1],shape=(4,1))
        
    def adjoint(self,x, out=None):
        tmp1 = self.operator.adjoint(bdc(x[0],x[1]))
        tmp2 = self.operator.adjoint(bdc(x[2],x[3]))
        if out is None:
            return bdc(tmp1,tmp2)
        else:
            out = bdc(tmp1,tmp2)

@jit(nopython=True)
def array_dot(arr0, arr1):
    out_array = np.zeros(arr0[0].shape)
    for i in range(len(arr0)):
        out_array += arr0[i]*arr1[i]
    return out_array
    
def bdc_dot(bdc0, bdc1, image):
    arr_list0 = []
    arr_list1 = []
    for i in bdc0:
        arr_list0.append(np.squeeze(i.clone().as_array()))
    for j in bdc1:
        arr_list1.append(np.squeeze(j.clone().as_array()))
    arr = array_dot(np.array(arr_list0),np.array(arr_list1)).reshape(image.shape)
    return image.clone().fill(arr)
    
            
class DirectionalTV(LinearOperator):
    def __init__(self, anatomical_image, nu = 0.01, gamma=1, smooth = True, beta = 0.001,**kwargs):
        """Constructor method"""    
        # Consider pseudo 2D geometries with one slice, e.g., (1,voxel_num_y,voxel_num_x)
        self.is2D = False
        self.domain_shape = []
        self.ind = []
        if smooth is True:
            self.beta = beta
        self.voxel_size_order = []
        self._domain_geometry = anatomical_image
        for i, size in enumerate(list(self._domain_geometry.shape) ):
            if size!=1:
                self.domain_shape.append(size)
                self.ind.append(i)
                self.voxel_size_order.append(self._domain_geometry.spacing[i])
                self.is2D = True

        self.gradient = GradientOperator(anatomical_image, backend='numpy')
        
        self.anato = anatomical_image
    
        self.tmp_im = anatomical_image.clone()
        
       	# smoothing for xi 
        self.gamma = gamma
        
        self.anato_grad = self.gradient.direct(self.anato) # gradient of anatomical image

        self.denominator = (self.anato_grad.pnorm(2).power(2) + nu**2) #smoothed norm of anatomical image  

        self.ndim = len(self.domain_shape)

        super(DirectionalTV, self).__init__(domain_geometry=anatomical_image, 
              range_geometry = BlockGeometry(*[self._domain_geometry for _ in range(self.ndim)]))

    def direct(self, x, out=None): 
        inter_result = bdc_dot(x.clone(),self.anato_grad.clone(), self.tmp_im)

        if out is None:       
            return x.clone() - self.gamma*inter_result*(self.anato_grad/self.denominator) # (delv * delv dot del u) / (norm delv) **2
                
        else:
            out = x.clone() - self.gamma*inter_result*(self.anato_grad/self.denominator)
                
        
        
    def adjoint(self, x, out=None):  
        return self.direct(x, out=out) # self-adjoint operator
    
class igify(LinearOperator):
    def __init__(self, blockdatacontainer):
        
        self.BDC = blockdatacontainer
        self.vox_size = self.BDC[0].voxel_sizes()
        self.dims = self.BDC[0].dimensions()
        self.channels = len(self.BDC)
        self.ig = ImageGeometry(voxel_num_y=self.dims[1],
                     voxel_size_x=self.vox_size[2],
                     voxel_num_x=self.dims[2],
                     voxel_size_y=self.vox_size[1],
                     channels = self.channels)
        super(igify, self).__init__(domain_geometry=self.BDC, 
            range_geometry = self.ig)
        
    def direct(self, x, out = None):
        arrs = []
        for im in range(len(x)):
            arrs.append(np.squeeze(x[im].as_array()))
        res = self.ig.allocate()
        res.fill(np.array(arrs))
        if out is None:
            return res
        else:
            out = res
            
    def adjoint(self, x, out = None):
        arr = x.as_array()
        res = self.BDC.clone()
        for i in range(self.channels):
            res[i].fill(arr[i].reshape((1,self.dims[1],self.dims[2])))
        if out is None:
            return res
        else:
            out = res
            
class AnisoWeight(LinearOperator):
    def __init__(self, num0,num1,sens_bdc):
        self.num0 = num0
        self.num1 = num1
        self.sens_bdc = sens_bdc
        
        super(AnisoWeight, self).__init__(domain_geometry=sens_bdc, 
                range_geometry=sens_bdc) 
            
    def direct(self, x):
        return x*(((x[0]/self.num0 - x[1]/self.num1).power(2)*self.sens_bdc/(self.num0*self.num1)).sqrt()+0.001)
    
    def adjoint(self, x):
        return x/(((x[0]/self.num0 - x[1]/self.num1).power(2)*self.sens_bdc/(self.num0*self.num1)).sqrt()+0.001)
    
    
    
####### Method updates #######
    
# self.x *= ((self.num1*self.x[0] - self.num0*self.x[1]).power(2)*self.sens).sqrt()

def updateGD(self):
    '''Single iteration of Gradient Descent'''
    if self.update_step_size:
        # the next update and solution are calculated within the armijo_rule
        self.step_size = self.armijo_rule()
    else:
        '''Single iteration of Gradient Descent'''
        step = (self.x+0.001)/self.sens*self.step_size # BSREM precond
        self.objective_function.gradient(self.x, out=self.x_update)
        self.x_update.multiply(step, out = self.x_update)
        self.x.subtract(self.x_update, out = self.x)
        # remove any negative values
        self.x.add(self.x.abs(), out = self.x) ## new line
        self.x.divide(2, out = self.x)    ## new line
        
def armijo_ruleGD(self):
    f_x = self.objective_function(self.x)
    if not hasattr(self, 'x_update'):
        self.x_update = self.objective_function.gradient(self.x)
    
    while self.k < self.kmax:
        # self.x - alpha * precond * self.x_update
        try:
            step = (self.x+0.001)/self.sens*self.alpha # BSREM precond
        except:
            step = self.alpha
        self.x_update.multiply(step, out=self.x_armijo)
        self.x.subtract(self.x_armijo, out=self.x_armijo)
        
        f_x_a = self.objective_function(self.x_armijo)
        sqnorm = self.x_update.squared_norm()
        if f_x_a - f_x <= - ( self.alpha/2. ) * sqnorm:
            self.x.fill(self.x_armijo)
            break
        else:
            self.k += 1.
            # we don't want to update kmax
            self._alpha *= self.beta

    if self.k == self.kmax:
        raise ValueError('Could not find a proper step_size in {} loops. Consider increasing alpha.'.format(self.kmax))
    return self.alpha
    

def update_objectiveGD(self):
    self.loss.append(self.objective_function(self.x))
    self.x[0].write("pet_"+self.prior+"_GD_"+str(self.iteration))
    self.x[1].write("spect_"+self.prior+"_GD_"+str(self.iteration))


def updateADMM_TNV(self):
    
    self.tmp_dir += self.u
    self.tmp_dir -= self.z
    self.operator.adjoint(self.tmp_dir, out = self.tmp_adj)
    
    self.x.sapyb(1, self.tmp_adj, -(self.tau/self.sigma), out=self.x)

    # apply proximal of f
    tmp = self.f.proximal(self.x, self.tau)
    self.operator.direct(tmp, out=self.tmp_dir)
    # store the result in x
    self.x.fill(tmp)
    del tmp

    self.u += self.tmp_dir
    
    # apply proximal of g   
    self.g.proximal(self.u, self.sigma, out = self.z)

    # update 
    self.u -= self.z
    self.x.add(self.x.abs(), out = self.x) ## new line
    self.x.divide(2, out = self.x)    ## new line   
    
    self.x[2].fill(self.CT) 

def updateADMM(self):
    
    self.tmp_dir += self.u
    self.tmp_dir -= self.z
    self.operator.adjoint(self.tmp_dir, out = self.tmp_adj)
    
    self.x.sapyb(1, self.tmp_adj, -(self.tau/self.sigma), out=self.x)

    # apply proximal of f
    tmp = self.f.proximal(self.x, self.tau)
    self.operator.direct(tmp, out=self.tmp_dir)
    # store the result in x
    self.x.fill(tmp)
    del tmp

    self.u += self.tmp_dir
    
    # apply proximal of g   
    self.g.proximal(self.u, self.sigma, out = self.z)

    # update 
    self.u -= self.z
    self.x.add(self.x.abs(), out = self.x) ## new line
    self.x.divide(2, out = self.x)    ## new line
        
def update_objectiveADMM(self):
    self.loss.append(self.f(self.x) +  self.g(self.operator.direct(self.x)) ) 
    self.x[0].write("pet_"+self.prior+"_ADMM_"+str(self.iteration))
    self.x[1].write("spect_"+self.prior+"_ADMM"+str(self.iteration))

@jit (nopython=True)
def arr_sum(ref_x,ref_y, im_list0, im_list1, im_list2, lst):
    for i in prange(ref_x):
        for j in prange(ref_y):
            u,s,vh = np.linalg.svd((np.array(([im_list0[0][i][j],im_list0[1][i][j]],
                            [im_list1[0][i][j],im_list1[1][i][j]],
                            [im_list2[0][i][j],im_list2[1][i][j]]))))
            lst = np.append(lst, np.sum(s))
    return np.sum(lst) 

def TNV_val(self, bdc):
    gradientBDC0 = self.grad.direct(bdc[0])
    gradientBDC1 = self.grad.direct(bdc[1])
    gradientBDC2 = self.grad.direct(bdc[2])
    im_list0, im_list1, im_list2= [], [], []
    for im in gradientBDC0:
        im_list0.append(np.squeeze(im.as_array()))
    for im in gradientBDC1:
        im_list1.append(np.squeeze(im.as_array()))
    for im in gradientBDC2:
        im_list2.append(np.squeeze(im.as_array())) 
    ref_x = np.squeeze(bdc[2].as_array()).shape[0]
    ref_y = np.squeeze(bdc[2].as_array()).shape[1]
    lst = np.array([])
    
    return arr_sum(ref_x,ref_y, np.array(im_list0), np.array(im_list1), np.array(im_list2), lst)

def update_objectiveADMM_TNV(self):
    self.loss.append(TNV_val(self, self.x) +  self.g(self.operator.direct(self.x)))
    self.x[0].write("pet_"+self.prior+"_ADMM_"+str(self.iteration))
    self.x[1].write("spect_"+self.prior+"_ADMM"+str(self.iteration))
        
def proximalOCF(self, x, tau, out=None):      
    x += x.abs()+0.0000001
    x/=2
    op = self.operator
    tmp = self.function.proximal(op.direct(x), tau)
    tmp -= op.direct(x)
    if out is None:
        return  x + op.adjoint(tmp)
    else:
        out = x + op.adjoint(tmp)

def proximal_conjugateOCF(self, x, tau, out=None):
    """ Return the proximal of F*(Ax) = G*(x)"""
    if out is None:
        return x - self.proximal(x, tau)
    else:
        out = x - self.proximal(x, tau)

def convex_conjugateOCF(self, x, out=None):
    """ Return the proximal of F(Ax)"""   
    if out is None:
        return self.function.convex_conjugate(self.operator.direct(x))
    else:
        out = self.function.convex_conjugate(self.operator.direct(x))
        
def allocateBDC(self, value=0):
    """ Return BLockDataConntainer with constant values"""
    out = self.copy()
    for im in out:
        im = im.allocate(value)
    return out