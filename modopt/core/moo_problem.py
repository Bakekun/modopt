from array_manager.api import VectorComponentsDict, MatrixComponentsDict, Vector, Matrix, BlockMatrix
from array_manager.api import DenseMatrix, COOMatrix, CSRMatrix, CSCMatrix

from modopt.utils.options_dictionary import OptionsDictionary
from modopt.utils.general_utils import pad_name
from modopt.core.recording_and_hotstart import record, hot_start

import numpy as np
import warnings

from copy import deepcopy

from abc import ABC, abstractmethod

class MOO_Problem(ABC):
    '''
    Base class for defining optimization problems in modOpt.

    Attributes
    ----------
    problem_name : str
        Problem name assigned by the user.
    x0 : np.ndarray
        Initial guess (scaled) for design variables.
    x : array_manager.Vector
        Current iterate (unscaled) for the design variables.
    nx : int
        Number of design variables or optimization variables.
    nc : int
        Number of constraints in the optimization problem.
    options : modopt.OptionsDictionary
        Problem-specific options declared by the user in addition
        to the global problem options 'jac_format' and 'hess_format'.
    objs : dict
        Dictionary with objective names as keys and current (unscaled) objective
        function values as values.
        Note that only one objective is supported by modOpt currently.
    obj_scaler : dict
        Dictionary with objective names as keys and objective.
        scalers as values. Default value for objective scalers is 1.0.
    o_scaler: float
        Objective scaler to use for single objective optimization.
    lag : dict
        Dictionary with the objective name as key and current Lagrangian
        function value as value.
    constrained: bool
        True if the problem has constraints. False if unconstrained.
    declared_variables: list
        List of problem variables declared by the user.
        It can at most be ['dv', 'objs', 'grad', 'con', 'jac', 'jvp', 'vjp', 
        'obj_hess', 'obj_hvp', 'lag', 'lag_grad', 'lag_hess', 'lag_hvp']

    x_lower : np.ndarray or NoneType
        Vector of (scaled) lower bounds for the design variables.
        x_lower[k] = -np.inf if the variable at x[k] has no lower bound.
        x_lower = None if no variables have lower bounds.
    x_upper : np.ndarray or NoneType
        Vector of (scaled) upper bounds for the design variables.
        x_upper[k] = np.inf if the variable at x[k] has no upper bound.
        x_upper = None if no variables have upper bounds.
    c_lower : np.ndarray or NoneType
        Vector of (scaled) lower bounds for the constraints.
        c_lower[k] = -np.inf if the constraint at c[k] has no lower bound.
        c_lower = None if unconstrained or no constraints have lower bounds.
    c_upper : np.ndarray or NoneType
        Vector of (scaled) upper bounds for the constraints.
        c_upper[k] = np.inf if the constraint at c[k] has no upper bound.
        c_upper = None if unconstrained or no constraints have upper bounds.

    x_scaler : np.ndarray or NoneType
        Vector of scalers for the design variables.
        x_scaler[k] = 1.0 by default.
    c_scaler : np.ndarray or NoneType
        Vector of scalers for the constraints.
        c_scaler = None if unconstrained.
        c_scaler[k] = 1.0 by default.

    design_variables_dict : arraymanager.VectorComponentsDict
        Dictionary containing (unscaled) design variable vector metadata.
    constraints_dict : arraymanager.VectorComponentsDict
        Dictionary containing (unscaled) constraint vector metadata.

    pC_px_dict : arraymanager.MatrixComponentsDict
        Dictionary containing (unscaled) constraint Jacobian matrix metadata.
    p2F_pxx_dict : arraymanager.MatrixComponentsDict
        Dictionary containing (unscaled) objective Hessian matrix metadata.
    p2L_pxx_dict : arraymanager.MatrixComponentsDict
        Dictionary containing Lagrangian Hessian matrix metadata.

    pF_px : array_manager.Vector
        Abstract vector containing (unscaled) objective gradients.
    pL_px : array_manager.Vector
        Abstract vector containing Lagrangian gradients.
    
    con : array_manager.Vector
        Abstract vector containing (unscaled) constraints.
    jvp : array_manager.Vector
        Abstract vector containing constraint Jacobian-vector products (JVPs).
    vjp : array_manager.Vector
        Abstract vector containing constraint vector-Jacobian products (VJPs).
    obj_hvp : array_manager.Vector
        Abstract vector containing objective Hessian-vector products (HVPs).
    lag_hvp : array_manager.Vector
        Abstract vector containing Lagrangian Hessian-vector products (HVPs).

    pC_px : arraymanager.Matrix
        Abstract matrix containing (unscaled) constraint Jacobian components.
    p2F_pxx : arraymanager.Matrix
        Abstract matrix containing (unscaled) objective Hessian components.
    p2L_pxx : arraymanager.Matrix
        Abstract matrix containing Lagrangian Hessian components.

    jac : (array_manager.DenseMatrix, array_manager.COOMatrix, array_manager.CSRMatrix, array_manager.CSCMatrix)
        Standard array_manager matrix object in the format 
        self.options['jac_format'] provided by the user.
        This object provides standard numpy dense or scipy sparse matrices.
        The output format for matrices is useful to meet the requirements for the chosen optimizer.
    obj_hess : (array_manager.DenseMatrix, array_manager.COOMatrix, array_manager.CSRMatrix, array_manager.CSCMatrix)
        Standard array_manager matrix object in the format 
        self.options['hess_format'] provided by the user. 
        This object provides standard numpy dense or scipy sparse (unscaled) objective Hessian matrices.
        The output format for matrices is useful to meet the requirements for the chosen optimizer.
    lag_hess : (array_manager.DenseMatrix, array_manager.COOMatrix, array_manager.CSRMatrix, array_manager.CSCMatrix)
        Standard array_manager matrix object in the format 
        self.options['hess_format'] provided by the user. 
        This object provides standard numpy dense or scipy sparse matrices.
        The output format for matrices is useful to meet the requirements for the chosen optimizer.
    '''
    def __init__(self, **kwargs):
        '''
        Initialize the Problem() object.
        Calls user-specified initialize() method, 
        and the _setup() method.
        '''
        # if type(self) is Problem:
        #     raise TypeError("Problem cannot be instantiated directly.")

        self.options = OptionsDictionary()

        self.problem_name = 'unnamed_problem'
        self.options.declare('jac_format',
                             default='dense',
                             values=('dense', 'coo', 'csr', 'csc'))
        self.options.declare('hess_format',
                             default='dense',
                             values=('dense', 'coo', 'csr', 'csc'))

        self.x0 = None
        self.nx = 0
        self.nc = 0
        # TODO: Fix this
        self.objs = {}
        self.obj_scaler = {}
        self.lag = {}
        self.constrained = False
        self.declared_variables = []

        # private attributes for recording, hot-starting, and visualization
        self._record                = None
        self._callback_count        = 0
        self._obj_count             = 0
        self._grad_count            = 0
        self._hess_count            = 0
        self._con_count             = 0
        self._jac_count             = 0
        self._hot_start_mode        = False
        self._hot_start_record      = None
        self._num_callbacks_found   = 0
        self._hot_start_tol         = None
        self._reused_callback_count = 0
        self._visualizer            = None

        ###############################
        # Only for the SURF algorithm #
        ###############################
        self.y0 = None
        self.ny = 0
        self.nr = 0
        self.implicit_constrained = False
        ###############################

        self.x_lower = None
        self.x_upper = None
        self.x_scaler = None
        self.c_lower = None
        self.c_upper = None
        self.c_scaler = None

        self.initialize()
        self.options.update(kwargs)

        self.design_variables_dict = VectorComponentsDict()
        self.constraints_dict = VectorComponentsDict()

        ###############################
        # Only for the SURF algorithm #
        ###############################
        self.state_variables_dict = VectorComponentsDict()
        self.residuals_dict = VectorComponentsDict()
        ###############################

        self._setup()

    def _setup(self):
        '''
        Calls user-specified setup() method, then _setup_scalers(), _setup_bounds().
        Sets up vectors for pF_px, obj_hvp.
        Sets up vectors for pL_px, jvp, vjp, and lag_hvp, if there are 
        constraints declared in the setup().
        Sets up a MatrixComponentsDict() object for the objective Hessian.
        Sets up MatrixComponentsDict() objects for the constraint Jacobian 
        and Lagrangian Hessian, if there are constraints.
        
        Calls user-specified setup_derivatives().
        Sets up matrices for Jacobian (if there are constraints),
        objective or Lagrangian Hessian depending on which one is
        declared by the user.

        Finally delete any unnecessary attributes allocated but
        was not declared by the user.
        For example, vjp, jvp, pL_px, obj_hvp, lag_hvp, 
        p2F_pxx_dict, p2L_pxx_dict, etc.
        '''
        self.setup()

        self._setup_scalers()
        # CSDLProblem() overrides this method in Problem()
        self._setup_bounds()
        self._setup_vectors()

        # When problem is not defined as CSDLProblem() [Note: self.x0 is always scaled]
        if self.x0 is None:
            # Note: array_manager puts np.zeros as the initial guess if no initial guess is provided
            self.x0 = self.x.get_data() * self.x_scaler
    
        self._setup_jacobian_dict()
        self._setup_hessian_dict()

        self._setup_matrices()

        self.raise_issues_with_user_setup()
        self.user_defined_callbacks = deepcopy(self.declared_variables)
        self.user_defined_callbacks.remove('dv')

    def __str__(self):
        """
        Print the details of the optimization problem.
        """
        name = self.problem_name
        objs  = self.objs
        obj_scaler = self.obj_scaler
        dvs  = self.x
        x_s  = self.x_scaler; x_l  = self.x_lower/x_s; x_u  = self.x_upper/x_s 
        if self.constrained:
            cons = self.con
            c_s  = self.c_scaler; c_l  = self.c_lower/c_s; c_u  = self.c_upper/c_s

        # output  = '\n\t'+'-'*100
        output  = f'\n\tProblem Overview:\n\t' + '-'*100
        output += f'\n\t' + pad_name('Problem name', 25) + f': {name}'
        
        # Print objective name
        output += f'\n\t' + pad_name('Objectives', 25) + f': '+', '.join(objs.keys())
        
        # Print design variables list with their dimensions
        dv_list = ''
        for dv_name, dv in dvs.dict_.items():
            dv_list += f'{dv_name} {dv.shape}, '
        dv_list = dv_list[:-2]
        output += f'\n\t' + pad_name('Design variables', 25) + f': {dv_list}'
        
        #  Print constraints and their dimensions
        if self.constrained:
            con_list = ''
            for con_name, con in cons.dict_.items():
                con_list += f'{con_name} {con.shape}, '
            con_list = con_list[:-2]
            output += f'\n\t' + pad_name('Constraints', 25) + f': {con_list}'

        output += '\n\t' + '-'*100
        
        output += f'\n\n\tProblem Data (UNSCALED):\n\t' + '-'*100
        
        # Print objective data
        output += f'\n\tObjectives:\n'
        header = "\t%-5s | %-10s | %-13s | %-13s " % ('Index', 'Name', 'Scaler', 'Value')
        output += header
        obj_template = "\n\t{idx:>5} | {name:<10} | {scaler:<+.6e} | {value:<+.6e}"
        for i, obj_name in enumerate(objs.keys()):
            obj_value = objs[obj_name]
            obj_s     = obj_scaler[obj_name]
            output   += obj_template.format(idx=i, name=obj_name, scaler=obj_s, value=obj_value)

        
        # Print design variable data
        output += f'\n\n\tDesign Variables:\n'
        header = "\t%-5s | %-10s | %-13s | %-13s | %-13s | %-13s " % ('Index', 'Name', 'Scaler', 'Lower Limit', 'Value', 'Upper Limit')
        output += header
        dv_template = "\n\t{idx:>5} | {name:<10} | {scaler:<+.6e} | {lower:<+.6e} | {value:<+.6e} | {upper:<+.6e}"

        idx = 0
        for dv_name, dv in dvs.dict_.items():
            for i, x in enumerate(dv.flatten()):
                dv_name  = dv_name[:10] if (len(dv_name)>10) else dv_name
                l = -1.e99 if x_l[idx] == -np.inf else x_l[idx]
                u = +1.e99 if x_u[idx] == +np.inf else x_u[idx]
                output += dv_template.format(idx=idx, name=dv_name+f'[{i}]', scaler=x_s[idx], lower=l, value=x, upper=u)
                idx += 1

        # Print constraint data
        if self.constrained:
            output += f'\n\n\tConstraints:\n'
            header = "\t%-5s | %-10s | %-13s | %-13s " % ('Index', 'Objective', 'Scaler', 'Value')
            output += header

            idx = 0
            con_template = "\n\t{idx:>5} | {name:<10} | {scaler:<+.6e} | {lower:<+.6e} | {value:<+.6e} | {upper:<+.6e} "
            for con_name, con in cons.dict_.items():
                for i, c in enumerate(con.flatten()):
                    con_name  = con_name[:10] if (len(con_name) > 10) else con_name
                    l = -1.e99 if c_l[idx] == -np.inf else c_l[idx]
                    u = +1.e99 if c_u[idx] == +np.inf else c_u[idx]
                    output += con_template.format(idx=idx, name=con_name + f'[{i}]', scaler=c_s[idx], lower=l, value=c, upper=u)
                    idx += 1

        output += '\n\t' + '-'*100 + '\n'
            
        return output

    @abstractmethod
    def initialize(self):
        '''
        Set problem name and any problem-specific options.
        '''
        # raise NotImplementedError("Subclasses must implement this method.")
        pass

    def _setup_scalers(self):
        '''
        Set x_scaler and c_scaler attributes using design_variables_dict and constraints_dict.
        c_scaler = None if there are no constraints.
        '''
        self.x_scaler = self.design_variables_dict.scaler
        # When unconstrained: self.constraints_dict.scaler = []
        if self.constrained:
            self.c_scaler = self.constraints_dict.scaler

        # Use this if componentwise dv scalers or constraint scalers are needed
        # self.x_scaler_abstract = Vector(self.design_variables_dict)
        # self.x_scaler_abstract.allocate(data=self.x_scaler, setup_views=True)
        # if self.constrained:
        #     self.c_scaler_abstract = Vector(self.constraints_dict)
        #     self.c_scaler_abstract.allocate(data=self.c_scaler, setup_views=True)

    def _setup_bounds(self):
        '''
        Compute scaled bounds x_lower, x_upper, c_lower, 
        and c_upper for the optimizer.
        This method is overridden in CSDLProblem(), OpenMDAOProblem(), CUTEstProblem().
        Note that c_upper = None and c_lower = None if there are no constraints.
        '''
        self.x_lower = self.design_variables_dict.lower * self.x_scaler
        self.x_upper = self.design_variables_dict.upper * self.x_scaler
        # When unconstrained: self.constraints_dict.lower = [], self.constraints_dict.lower = []
        if self.constrained:
            self.c_lower = self.constraints_dict.lower * self.c_scaler
            self.c_upper = self.constraints_dict.upper * self.c_scaler   

    def _setup_vectors(self):
        '''
        Set up array_manager abstract vectors for design variables 'x',
        objective gradients 'pF_px', objective HVP 'obj_hvp'.
        If constrained, also set up abstract vectors for constraints 'con',
        constraint JVP 'jvp', constraint VJP 'vjp', 
        Lagrangian gradients 'pL_px', Lagrangian HVP 'lag_hvp'.
        '''
        self.x = Vector(self.design_variables_dict)
        self.x.allocate(setup_views=True)
        self.pF_px = Vector(self.design_variables_dict)
        self.pF_px.allocate(data=np.zeros((self.nx, )),
                            setup_views=True)
        self.obj_hvp = Vector(self.design_variables_dict)
        self.obj_hvp.allocate(data=np.zeros((self.nx, )),
                              setup_views=True)
        self.vec_hvp = Vector(self.design_variables_dict)
        self.vec_hvp.allocate(data=np.zeros((self.nx, )),
                              setup_views=True)
        
        if self.constrained:
            self.con = Vector(self.constraints_dict)
            self.con.allocate(setup_views=True)

            self.lag_mult = Vector(self.constraints_dict)
            self.lag_mult.allocate(data=np.zeros((self.nc, )),
                                   setup_views=True)

            self.jvp = Vector(self.constraints_dict)
            self.jvp.allocate(data=np.zeros((self.nc, )),
                              setup_views=True)
            self.vec_jvp = Vector(self.design_variables_dict)
            self.vec_jvp.allocate(data=np.zeros((self.nx, )),
                                  setup_views=True)
            
            self.vjp = Vector(self.design_variables_dict)
            self.vjp.allocate(data=np.zeros((self.nx, )),
                              setup_views=True)
            self.vec_vjp = Vector(self.constraints_dict)
            self.vec_vjp.allocate(data=np.zeros((self.nc, )),
                                  setup_views=True)
            
            self.pL_px = Vector(self.design_variables_dict)
            self.pL_px.allocate(data=np.zeros((self.nx, )),
                                setup_views=True)
            self.lag_hvp = Vector(self.design_variables_dict)
            self.lag_hvp.allocate(data=np.zeros((self.nx, )),
                                  setup_views=True)

        ###############################
        # Only for the SURF algorithm #
        ###############################
        if self.implicit_constrained:
            self.y = Vector(self.state_variables_dict)
            self.y.allocate(setup_views=True)

            self.pF_py = Vector(self.design_variables_dict)
            self.pF_py.allocate(data=np.zeros((self.ny, )),
                                setup_views=True)

            self.residuals = Vector(self.residuals_dict)
            self.residuals.allocate(setup_views=True)
        ###############################

        # self.constraint_duals = Vector(self.constraints_dict)
        # self.residual_duals = Vector(self.residuals_dict) 

    def _setup_jacobian_dict(self):
        '''
        Set up array_manager MatrixComponentDict for constraint Jacobians,
        if the problem is constrained.
        '''
        if self.constrained:
            self.pC_px_dict = MatrixComponentsDict(
                self.constraints_dict, self.design_variables_dict)
            
        ###############################
        # Only for the SURF algorithm #
        ###############################
            if self.implicit_constrained:
                self.pC_py_dict = MatrixComponentsDict(
                    self.constraints_dict, self.state_variables_dict)       
        
        if self.implicit_constrained:
            self.pR_px_dict = MatrixComponentsDict(
                self.residuals_dict, self.design_variables_dict)
            self.pR_py_dict = MatrixComponentsDict(
                self.residuals_dict, self.state_variables_dict)
        ###############################

    def _setup_hessian_dict(self):
        '''
        Set up array_manager MatrixComponentDict for objective Hessians.
        Also set up MatrixComponentDict for Lagrangian Hessians,
        if the problem is constrained.       
        '''
        self.p2F_pxx_dict = MatrixComponentsDict(
            self.design_variables_dict, self.design_variables_dict)

        if self.constrained:
            self.p2L_pxx_dict = MatrixComponentsDict(
                self.design_variables_dict, self.design_variables_dict)
        
        ###############################
        # Only for the SURF algorithm #
        ###############################
            if self.implicit_constrained:
                self.p2L_pxy_dict = MatrixComponentsDict(
                    self.design_variables_dict, self.state_variables_dict)
                self.p2L_pyy_dict = MatrixComponentsDict(
                    self.state_variables_dict, self.state_variables_dict)

        elif self.implicit_constrained:
            self.p2L_pxx_dict = MatrixComponentsDict(
                self.design_variables_dict, self.design_variables_dict)
            self.p2L_pxy_dict = MatrixComponentsDict(
                self.design_variables_dict, self.state_variables_dict)
            self.p2L_pyy_dict = MatrixComponentsDict(
                self.state_variables_dict, self.state_variables_dict)
        ###############################

    def _setup_matrices(self):
        '''
        Set up array_manager native Matrix and standard Matrix for 
        objective/Lagrangian Hessian depending on whether they are declared.
        Also set up native Matrix and standard Matrix for constraint Jacobian,
        if the problem is constrained.   
        '''
        if 'obj_hess' in self.declared_variables:
            self.p2F_pxx = Matrix(self.p2F_pxx_dict, setup_views=True)
            self.p2F_pxx.allocate()
            if self.options['hess_format'] == 'dense':
                self.obj_hess = DenseMatrix(self.p2F_pxx)
            elif self.options['hess_format'] == 'coo':
                self.obj_hess = COOMatrix(self.p2F_pxx)
            elif self.options['hess_format'] == 'csr':
                self.obj_hess = CSRMatrix(self.p2F_pxx)
            else:
                self.obj_hess = CSCMatrix(self.p2F_pxx)

        if self.constrained:
            self.pC_px = Matrix(self.pC_px_dict, setup_views=True)
            self.pC_px.allocate()
            # TODO: add standard matrices for all jac and hess
            if self.options['jac_format'] == 'dense':
                self.jac = DenseMatrix(self.pC_px)
            elif self.options['jac_format'] == 'coo':
                self.jac = COOMatrix(self.pC_px)
            elif self.options['jac_format'] == 'csr':
                self.jac = CSRMatrix(self.pC_px)
            else:
                self.jac = CSCMatrix(self.pC_px)

            if 'lag_hess' in self.declared_variables:    
                self.p2L_pxx = Matrix(self.p2L_pxx_dict, setup_views=True)
                self.p2L_pxx.allocate()
                if self.options['hess_format'] == 'dense':
                    self.lag_hess = DenseMatrix(self.p2L_pxx)
                elif self.options['hess_format'] == 'coo':
                    self.lag_hess = COOMatrix(self.p2L_pxx)
                elif self.options['hess_format'] == 'csr':
                    self.lag_hess = CSRMatrix(self.p2L_pxx)
                else:
                    self.lag_hess = CSCMatrix(self.p2L_pxx)

        ###############################
        # Only for the SURF algorithm #
        ###############################        

                if self.implicit_constrained:
                    self.p2L_pxy = Matrix(self.p2L_pxy_dict, setup_views=True)
                    self.p2L_pxy.allocate()
                    self.p2L_pyy = Matrix(self.p2L_pyy_dict, setup_views=True)
                    self.p2L_pyy.allocate()

            if self.implicit_constrained:
                self.pC_py = Matrix(self.pC_py_dict, setup_views=True)
                self.pC_py.allocate()
                self.pR_px = Matrix(self.pR_px_dict, setup_views=True)
                self.pR_px.allocate()
                self.pR_py = Matrix(self.pR_py_dict, setup_views=True)
                self.pR_py.allocate()

        elif self.implicit_constrained:
            self.pR_px = Matrix(self.pR_px_dict, setup_views=True)
            self.pR_px.allocate()
            self.pR_py = Matrix(self.pR_py_dict, setup_views=True)
            self.pR_py.allocate()
            if 'lag_hess' in self.declared_variables:
                self.p2L_pxx = Matrix(self.p2L_pxx_dict, setup_views=True)
                self.p2L_pxx.allocate()    
                self.p2L_pxy = Matrix(self.p2L_pxy_dict, setup_views=True)
                self.p2L_pxy.allocate()
                self.p2L_pyy = Matrix(self.p2L_pyy_dict, setup_views=True)
                self.p2L_pyy.allocate()
        ###############################
            
    def raise_issues_with_user_setup(self):
        '''
        Raise errors or warnings associated with declarations made by the user in setup().
        Overridden when using interfaced modeling frameworks like CSDL or OpenMDAO and CUTEst.
        '''
        if 'dv' not in self.declared_variables:
            raise Exception("No design variables are declared.")

        if 'objs' in self.declared_variables:
            if not hasattr(self, 'compute_objectives'):
                raise Exception("Objective is declared but compute_objectives() method is not implemented.")
        else:
            if 'con' not in self.declared_variables:
                raise Exception("No objective or constraints are declared.")
            warnings.warn("No objective is declared. Running a feasibility problem.")
            self.add_objectives('dummy_obj')
            self.objs['dummy_obj'] = 0.  # Default value 1. is replaced with 0. for feasibility problems

            # Set a dummy function for compute_objectives to avoid NotImplementedError
            self.compute_objectives = lambda dvs, objs: None

        if 'con' in self.declared_variables:
            if not hasattr(self, 'compute_constraints'):
                raise Exception("Constraints are declared but compute_constraints() method is not implemented.")

        if self.constrained:
            if 'jac' in self.declared_variables and not hasattr(self, 'compute_constraint_jacobian'):
                raise Exception("Constraint Jacobian is declared but compute_constraint_jacobian() method is not implemented.")

            if 'jvp' in self.declared_variables and not hasattr(self, 'compute_constraint_jvp'):
                raise Exception("Constraint JVP is declared but compute_constraint_jvp() method is not implemented.")

            if 'vjp' in self.declared_variables and not hasattr(self, 'compute_constraint_vjp'):
                raise Exception("Constraint VJP is declared but compute_constraint_vjp() method is not implemented.")

        if self.constrained and all(x not in self.declared_variables for x in ['jac', 'jvp', 'vjp']):
            warnings.warn("No constraint-related derivatives (jacobian, jvp, vjp) are declared.")

    def add_design_variables(self,
                             name=None,
                             shape=(1, ),
                             scaler=None,
                             lower=None,
                             upper=None,
                             equals=None,
                             vals=None):
        '''
        User calls this method within Problem.setup() method
        to add design variable vectors for the problem.

        Parameters
        ----------
        name : str
            Design variable name assigned by the user.
        shape : tuple, default=(1,)
            Design variable shape. (1,) by default.
        scaler : float or np.ndarray, optional
            Design variable scaling factor.
            It can be a single scaler for all variables in the vector,
            or an array of scalers with the same shape as the design variable.
        lower : float or np.ndarray, optional
            Design variable lower bound.
            It can be a float in which case the given lower bound applies to all variables
            in the design variable vector.
            An array of lower bounds with the same shape as the design variable is also
            acceptable.
        upper: float or np.ndarray, optional
            Design variable upper bound.
            It can be a float in which case the given upper bound applies to all variables
            in the design variable vector.
            An array of upper bounds with the same shape as the design variable is also
            acceptable.
        equals: float or np.ndarray, optional
            Employing this makes the design variable a fixed constant.
            This must be used only for debugging purposes.
        vals: float or np.ndarray, optional
            Initial values for the design variables.
            It can be a single value for all variables in the vector,
            or an array of initial values with the same shape as the design variable.
            If nothing is provided, 0. will be taken as the initial guess.
        '''
        # array_manager automatically sets vals = np.zeros(size) if vals is None
        # Autonaming index starts from x0; for entries inside: xi_j (i dv_index, j dv_sub_index)
        if name is None:
            raise ValueError('A name must be provided for adding design variables.')
            # name = 'x' + str(len(self.design_variables_dict))

        self.design_variables_dict[name] = dict(
            shape=shape,
            scaler=scaler,
            lower=lower,
            upper=upper,
            equals=equals,
            vals=vals,
        )

        if 'dv' not in self.declared_variables:
            self.declared_variables.append('dv')

        # Update the number of design variables
        self.nx += np.prod(shape)

    def add_objectives(self, obj_names, scalers=None):
        """
        User calls this method within Problem.setup() to add multiple objectives.

        Parameters
        ----------
        obj_names : list of str
            Names of the objectives.
        scalers : list of float, optional
            Scaling factors for each objective (default is 1.0 for each).
        """
        if not isinstance(obj_names, list) or len(obj_names) < 2:
            raise ValueError("Multi-objective optimization requires at least two objectives.")

        self.objs = {}  # Reset objectives dictionary
        self.obj_scaler = {}

        for i, name in enumerate(obj_names):
            self.objs[name] = 0.0  # Default value for each objective
            self.obj_scaler[name] = scalers[i] if scalers else 1.0  # Default scaler to 1.0 if not provided

        print(f"Setting objective names as {', '.join(self.objs.keys())}.")

        if 'objs' not in self.declared_variables:
            self.declared_variables.append('objs')

    def add_constraints(self,
                        name=None,
                        shape=(1, ),
                        scaler=None,
                        lower=None,
                        upper=None,
                        equals=None):
        '''
        User calls this method within Problem.setup() method
        to add constraints for the problem.

        Parameters
        ----------
        name : str
            Constraint name assigned by the user.
        shape : tuple, default=(1,)
            Constraint shape. (1,) by default.
        scaler : float or np.ndarray, optional
            Constraint scaling factor.
            It can be a single scaler for all constraints in the vector,
            or an array of scalers with the same shape as the constraint.
        lower : float or np.ndarray, optional
            Constraint lower bound.
            It can be a float in which case the given lower bound applies to all constraints
            in the constraint vector.
            An array of lower bounds with the same shape as the constraint is also
            acceptable.
        upper: float or np.ndarray, optional
            Constraint upper bound.
            It can be a float in which case the given upper bound applies to all constraints
            in the constraint vector.
            An array of upper bounds with the same shape as the constraint is also
            acceptable.
        equals: float or np.ndarray, optional
            Used for defining an equality constraint.
            It can be a float in which case the given constant applies to all constraints
            in the constraint vector.
            An array of floats with the same shape as the constraint is also acceptable.
            It is used when the right-hand side constants for the equality constraints
            are different.
        '''
        # Autonaming index starts from 0; for entries inside: ci_j (i con_index, j con_sub_index)
        if name is None:
            raise ValueError('A name must be provided for adding constraints.')
            # name = 'c' + str(len(self.constraints_dict))

        self.constraints_dict[name] = dict(
            shape=shape,
            scaler=scaler,
            lower=lower,
            upper=upper,
            equals=equals,
        )

        if not self.constrained:
            self.constrained = True
        
        if 'con' not in self.declared_variables:
            self.declared_variables.append('con')
        
        # Update the number of constraints
        self.nc += np.prod(shape)

    def declare_constraint_jacobian(self,
                                    of,
                                    wrt,
                                    shape=None,
                                    vals=None,
                                    rows=None,
                                    cols=None,
                                    ind_ptr=None):
        '''
        User calls this method within Problem.setup_derivatives() method
        to declare nonzero constraint Jacobians.
        Jacobian components that are undeclared are assumed to be zeros.
        If the Jacobian is provided later (in compute_constraint_jacobian() method) 
        in one of the sparse formats coo, csr, or csc, declare the sparsity 
        by calling this method with kwargs (rows, cols), (rows, ind_ptr), 
        or (cols, ind_ptr), respectively.

        Parameters
        ----------
        of : str
            Name of the constraint for which the Jacobian needs to be declared.
        wrt : str
            Name of the variable w.r.t. which the Jacobian needs to be declared.
        rows : np.ndarray, optional
            Row indices corresponding to vals. 
            Needs to be declared if the format for the declared Jacobian is coo or csr.
        cols : np.ndarray, optional
            Column indices corresponding to vals. 
            Needs to be declared if the format for the declared Jacobian is coo or csc.
        ind_ptr : np.ndarray, optional
            Index pointer array for compressed index formats.
            Needs to be declared if the format for the declared Jacobian is csr or csc.
        shape : tuple, optional
            Shape in which 'vals' are going to be provided later by the user 
            (for dense or sparse formats).
            Note that this is not the shape of the declared Jacobian.
        vals : float or np.ndarray, optional
            Values for the constraint Jacobian. Useful if the constraint is 
            independent of or linearly-dependent on the declared "wrt" design variables.
            'vals' are nonzero entries corresponding to 'rows' or 'cols' if the 
            declared Jacobian is in sparse format.
        '''
        if of not in self.constraints_dict:
            raise KeyError(f'Jacobian is declared for undeclared constraint {of}.')
        if wrt not in self.design_variables_dict:
            raise KeyError(f'Jacobian is declared with respect to undeclared design variable {wrt}')

        if 'jac' not in self.declared_variables:
            self.declared_variables.append('jac')
        
        self.pC_px_dict[of, wrt] = dict(
            vals=vals,
            rows=rows,
            cols=cols,
            ind_ptr=ind_ptr,
            vals_shape=shape,
        )

    def declare_constraint_jvp(self, of, vals=None):
        '''
        User calls this method within Problem.setup_derivatives() method
        to declare constraint Jacobian-vector product (JVP).

        Parameters
        ----------
        of : str
            Name of the constraint for which the JVP needs to be declared.
        vals : float or np.ndarray, optional
            Values for the constraint JVP. Useful if the "of" constraint is 
            only linearly-dependent on all of the design variables.
        '''
        if of not in self.constraints_dict:
            raise KeyError(f'JVP is declared for undeclared constraint {of}.')

        if 'jvp' not in self.declared_variables:
            self.declared_variables.append('jvp')
        if vals is not None:
            self.jvp[of] = vals

    def declare_constraint_vjp(self, wrt, vals=None):
        '''
        User calls this method within Problem.setup_derivatives() method
        to declare constraint vector-Jacobian product (VJP).

        Parameters
        ----------
        wrt : str
            Name of the variable w.r.t. which the VJP needs to be declared.
        vals : float or np.ndarray, optional
            Values for the constraint VJP. Useful if all the constraints are 
            independent of or linearly-dependent on the "wrt" design variables.
        '''
        if wrt not in self.design_variables_dict:
            raise KeyError(f'VJP is declared with respect to undeclared design variable {wrt}.')

        if 'vjp' not in self.declared_variables:
            self.declared_variables.append('vjp')
        if vals is not None:
            self.vjp[wrt] = vals

    def declare_objectives_hessian(self, obj_names, wrt, shape=None, vals=None, rows=None, cols=None, ind_ptr=None):
        """
        Declare nonzero Hessian components for multiple objectives.

        Parameters
        ----------
        obj_names : list of str
            Names of the objectives.
        wrt : str
            Name of the variable w.r.t. which the Hessian needs to be declared.
        """
        for obj_name in obj_names:
            if (wrt not in self.design_variables_dict) or (obj_name not in self.objs):
                raise KeyError(f'Hessian declared for undeclared variable or objective ({obj_name}, {wrt}).')

            if 'obj_hess' not in self.declared_variables:
                self.declared_variables.append('obj_hess')

            self.p2F_pxx_dict[obj_name, wrt] = dict(
                vals=vals, rows=rows, cols=cols, ind_ptr=ind_ptr, vals_shape=shape
            )

    def declare_objectives_hvp(self, obj_names, wrt, vals=None):
        """
        Declare Hessian-Vector Products (HVPs) for multiple objectives.

        Parameters
        ----------
        obj_names : list of str
            Names of the objectives.
        wrt : str
            Name of the variables w.r.t. which the HVP needs to be declared.
        """
        for obj_name in obj_names:
            if wrt not in self.design_variables_dict:
                raise KeyError(f'HVP declared with respect to undeclared design variable {wrt}')

            if 'obj_hvp' not in self.declared_variables:
                self.declared_variables.append('obj_hvp')

            self.obj_hvp[obj_name, wrt] = vals

    def raise_not_implemented_error(self, method_name):
        '''
        Raise NotImplementedError when an optional abstract method is called 
        but not implemented by the user in the derived class.

        Parameters
        ----------
        method_name : str
            Name of the method that is not implemented.
        '''
        raise NotImplementedError(f"{method_name}() method is not implemented by the user in the derived class {self.__class__.__name__}.")

    def compute_objectives(self, dvs, objs):
        """
        Compute multiple objectives given the design variable vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        objs : dict
            Dictionary of objective function names and values.
        """
        self.raise_not_implemented_error('compute_objectives')

    def compute_constraints(self, dvs, con):
        """
        Compute the constraint vector given the design variable vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        con : array_manager.Vector
            Vector of constraints.
            This abstract vector has dictionary-type views for 
            component constraint vectors.
        """
        self.raise_not_implemented_error('compute_constraints')

    def compute_objectives_gradient(self, dvs, grads):
        """
        Compute the gradients of multiple objective functions.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
        grads : dict
            Dictionary of gradients, one for each objective.
        """
        self.raise_not_implemented_error('compute_objectives_gradient')

    def compute_constraint_jacobian(self, dvs, jac):
        """
        Compute the constraint Jacobian with respect to the design variable vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        jac : array_manager.Matrix
            Jacobian matrix of the constraints with respect to the design variable vector.
            This abstract matrix has dictionary-type views for 
            component sub-Jacobians with keys (of,wrt) where 
            'of' is the constraint name and 'wrt' the design variable name.
        """
        self.raise_not_implemented_error('compute_constraint_jacobian')

    def compute_constraint_jvp(self, dvs, vec, jvp):
        """
        Compute the constraint Jacobian-vector product (JVP) for the given design variable vector and
        multiplying vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        vec : array_manager.Vector
            Vector to multiply with the Jacobian.
            This abstract vector has dictionary-type views corresponding to 
            component design variable vectors.
        jvp : array_manager.Vector
            Constraint Jacobian-vector product.
            This abstract vector has dictionary-type views corresponding to 
            component constraint vectors.
        """
        self.raise_not_implemented_error('compute_constraint_jvp')

    def compute_constraint_vjp(self, dvs, vec, vjp):
        """
        Compute the constraint vector-Jacobian product (VJP) for the given design variable vector and
        multiplying vector.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
            This abstract vector has dictionary-type views for 
            component design variable vectors.
        vec : array_manager.Vector
            Vector to multiply with the Jacobian.
            This abstract vector has dictionary-type views corresponding to 
            component constraint vectors.
        vjp : array_manager.Vector
            Constraint vector-Jacobian product.
            This abstract vector has dictionary-type views corresponding to 
            component design variable vectors.
        """
        self.raise_not_implemented_error('compute_constraint_vjp')

    def compute_objectives_hessian(self, dvs, obj_hess_dict):
        """
        Compute Hessians for multiple objective functions.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
        obj_hess_dict : dict
            Dictionary containing Hessians for multiple objectives.
        """
        self.raise_not_implemented_error('compute_objectives_hessian')

    def compute_objectives_hvp(self, dvs, vec, obj_hvp_dict):
        """
        Compute Hessian-vector products for multiple objective functions.

        Parameters
        ----------
        dvs : array_manager.Vector
            Design variable vector.
        vec : array_manager.Vector
            Vector to multiply with the Hessian.
        obj_hvp_dict : dict
            Dictionary containing HVP results for each objective.
        """
        self.raise_not_implemented_error('compute_objectives_hvp')

    def use_finite_differencing(self, derivative, step=1e-6):
        '''
        User calls this method within compute methods to approximate derivatives.

        Parameters
        ----------
        derivative : str
            Derivative to approximate.
            Available derivatives are `'objective_gradient'`, `'objective_hessian'`, `'constraint_jacobian'`,
            `'objective_hvp'`, `'constraint_jvp'`.
        step : float or np.ndarray, default=1e-6
            Finite difference step size.
        '''
        if np.isscalar(step):
            if not np.isreal(step):
                raise ValueError('Step size "step" must be a real number.')
        elif step.shape != (self.nx,):
            raise ValueError('Step size "step" must be a scalar or an array of size (nx,) where nx is the number of design variables.')
        if derivative == 'objective_gradient':
            x = self.x.get_data()
            self.compute_objective(self.x, self.objs)
            f0 = list(self.objs.values())[0]
            g_fd = np.zeros((self.nx,))
            for i in range(self.nx):
                e = np.zeros((self.nx,))
                e[i] = 1.
                self.x.set_data(x + step*e)
                self.compute_objective(self.x, self.objs)
                f1 = list(self.objs.values())[0]
                g_fd[i] = (f1 - f0)
            g_fd /= step
            self.pF_px.set_data(g_fd)
            
        elif derivative == 'objective_hessian':
            x = self.x.get_data()
            self.compute_objective_gradient(self.x, self.pF_px)
            g0 = self.pF_px.get_data()
            H_fd = np.zeros((self.nx, self.nx))
            for i in range(self.nx):
                e = np.zeros((self.nx,))
                e[i] = 1.
                self.x.set_data(x + step*e)
                self.compute_objective_gradient(self.x, self.pF_px)
                g1 = self.pF_px.get_data()
                H_fd[:, i] = (g1 - g0)
            H_fd /= step # rowwise division if step is a 1d array
            self.p2F_pxx.vals.set_data(H_fd.flatten())

        elif derivative == 'constraint_jacobian':
            x = self.x.get_data()
            self.compute_constraints(self.x, self.con)
            c0 = self.con.get_data()
            J_fd = np.zeros((self.nc, self.nx))
            for i in range(self.nx):
                e = np.zeros((self.nx,))
                e[i] = 1.
                self.x.set_data(x + step*e)
                self.compute_constraints(self.x, self.con)
                c1 = self.con.get_data()
                J_fd[:, i] = (c1 - c0)
            J_fd /= step # rowwise division if step is a 1d array
            self.pC_px.vals.set_data(J_fd.flatten())
        
        elif not np.isscalar(step):
            raise ValueError('Step size "step" must be a scalar for JVP or HVP derivative.')

        elif derivative == 'objective_hvp':
            x = self.x.get_data()
            self.compute_objective_gradient(self.x, self.pF_px)
            g0 = self.pF_px.get_data()
            v = self.vec_hvp.get_data()

            self.x.set_data(x + step*v)
            self.compute_objective_gradient(self.x, self.pF_px)
            g1 = self.pF_px.get_data()
            hvp_fd = (g1 - g0) / step
            self.obj_hvp.set_data(hvp_fd)

        elif derivative == 'constraint_jvp':
            x = self.x.get_data()
            self.compute_constraints(self.x, self.con)
            c0 = self.con.get_data()
            v = self.vec_jvp.get_data()

            self.x.set_data(x + step*v)
            self.compute_constraints(self.x, self.con)
            c1 = self.con.get_data()
            jvp_fd = (c1 - c0) / step
            self.jvp.set_data(jvp_fd)

        else:
            raise NotImplementedError('Finite differencing is not implemented for the requested derivative.')

    # WRAPPER FOR USER-DEFINED COMPUTE METHODS BELOW (USED BY Optimizer() OBJECTS):
    # =============================================================================
    @record(['x'], ['objs'])
    @hot_start(['x'], ['objs'])
    def _compute_objectives(self, x):
        '''
        Wrapper for user-defined compute_objectives().
        Scales multiple objectives before passing them to the optimizer.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.

        Returns
        -------
        np.ndarray
            Array of objective function values.
        '''
        self.x.set_data(x / self.x_scaler)
        self.compute_objectives(self.x, self.objs)  # Modify to handle multiple objectives
        objectives = np.array(list(self.objs.values()))
        objective_scalers = np.array(list(self.obj_scaler.values()))
        return objectives * objective_scalers  # Return array instead of single value

    @record(['x'], ['grads'])
    @hot_start(['x'], ['grads'])
    def _compute_objectives_gradient(self, x):
        '''
        Wrapper for user-defined compute_objectives_gradient().
        Scales multiple objective gradients before passing them to the optimizer.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.

        Returns
        -------
        np.ndarray
            2D array where each row corresponds to the gradient of an objective.
        '''
        self.x.set_data(x / self.x_scaler)
        self.compute_objectives_gradient(self.x, self.grads)
        objective_scalers = np.array(list(self.obj_scaler.values()))
        return np.vstack([self.grads[objs] * objective_scalers[i] / self.x_scaler
                          for i, objs in enumerate(self.grads.keys())])

    @record(['x'], ['obj_hess_dict'])
    @hot_start(['x'], ['obj_hess_dict'])
    def _compute_objectives_hessian(self, x):
        '''
        Wrapper for user-defined compute_objectives_hessian().
        Scales multiple objective Hessians before passing them to the optimizer.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.

        Returns
        -------
        dict
            Dictionary of Hessians for multiple objectives.
        '''
        self.x.set_data(x / self.x_scaler)
        self.compute_objectives_hessian(self.x, self.obj_hess_dict)
        objective_scalers = np.array(list(self.obj_scaler.values()))
        hessians = {key: self.obj_hess_dict[key].get_std_array() *
                         (objective_scalers[i] / np.outer(self.x_scaler, self.x_scaler))
                    for i, key in enumerate(self.obj_hess_dict.keys())}
        return hessians

    @record(['x', 'v'], ['obj_hvp_dict'])
    @hot_start(['x', 'v'], ['obj_hvp_dict'])
    def _compute_objectives_hvp(self, x, v):
        '''
        Wrapper for user-defined compute_objectives_hvp().
        Scales multiple Hessian-vector products before passing them to the optimizer.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        v : np.ndarray
            Vector to right-multiply Hessian with.

        Returns
        -------
        dict
            Dictionary of Hessian-vector products for multiple objectives.
        '''
        self.x.set_data(x / self.x_scaler)
        self.vec_hvp.set_data(v / self.x_scaler)
        self.compute_objectives_hvp(self.x, self.vec_hvp, self.obj_hvp_dict)
        objective_scalers = np.array(list(self.obj_scaler.values()))
        hvps = {key: self.obj_hvp_dict[key].get_data() * objective_scalers[i] / self.x_scaler
                for i, key in enumerate(self.obj_hvp_dict.keys())}
        return hvps

    @record(['x'],['con'])
    @hot_start(['x'],['con'])
    def _compute_constraints(self, x):
        '''
        Wrapper for user-defined compute_constraints(). 
        Arguments are numpy arrays, performs problem- and optimizer-independent scaling.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
                        
        Returns
        -------
        np.ndarray
            1-dimensional constraint vector.
        '''
        self.x.set_data(x / self.x_scaler)
        self.compute_constraints(self.x, self.con)
        # print('con', self.con.get_data())
        return self.con.get_data() * self.c_scaler

    @record(['x'],['jac'])
    @hot_start(['x'],['jac'])
    def _compute_constraint_jacobian(self, x):
        '''
        Wrapper for user-defined compute_constraint_jacobian(). 
        Arguments are numpy arrays, performs problem- and optimizer-independent scaling.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
                        
        Returns
        -------
        np.ndarray
            2-dimensional constraint Jacobian matrix.
        '''
        self.x.set_data(x / self.x_scaler)
        self.compute_constraint_jacobian(self.x, self.pC_px)
        self.jac.update_bottom_up()
        # print('jac', self.jac.get_std_array())
        return self.jac.get_std_array() * np.outer(self.c_scaler, 1./self.x_scaler)
    
    @record(['x', 'v'],['jvp'])
    @hot_start(['x', 'v'],['jvp'])
    def _compute_constraint_jvp(self, x, v):
        '''
        Wrapper for user-defined compute_constraint_jvp(). 
        Arguments are numpy arrays, performs problem- and optimizer-independent scaling.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        v : np.ndarray
            Vector to right-multiply Jacobian with.
                        
        Returns
        -------
        np.ndarray
            1-dimensional constraint JVP vector.
        '''
        self.x.set_data(x / self.x_scaler)
        self.vec_jvp.set_data(v/self.x_scaler)
        self.compute_constraint_jvp(self.x, self.vec_jvp, self.jvp)
        # print('jvp', self.jvp.get_data())
        return self.jvp.get_data() * self.c_scaler
    
    @record(['x', 'v'],['vjp'])
    @hot_start(['x', 'v'],['vjp'])
    def _compute_constraint_vjp(self, x, v):
        '''
        Wrapper for user-defined compute_constraint_vjp(). 
        Arguments are numpy arrays, performs problem- and optimizer-independent scaling.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        v : np.ndarray
            Vector to left-multiply Jacobian with.
                        
        Returns
        -------
        np.ndarray
            1-dimensional constraint VJP vector.
        '''
        self.x.set_data(x / self.x_scaler)
        self.vec_vjp.set_data(v*self.c_scaler)
        self.compute_constraint_vjp(self.x, self.vec_vjp, self.vjp)
        # print('vjp', self.vjp.get_data())
        return self.vjp.get_data() / self.x_scaler

    def add_state_variables(self,
                            name,
                            shape=(1, ),
                            lower=None,
                            upper=None,
                            equals=None,
                            vals=None):
        if vals is None:
            vals == np.zeros(shape)
        self.state_variables_dict[name] = dict(shape=shape,
                                               lower=lower,
                                               upper=upper,
                                               equals=equals,
                                               vals=vals)

        self.ny += np.prod(shape)

    def add_residuals(self, name, shape=(1, )):
        self.residuals_dict[name] = dict(shape=shape,
                                         lower=None,
                                         upper=None,
                                         equals=np.zeros(shape))

        self.nr += np.prod(shape)

    def declare_pF_px_gradient(self, wrt, shape=(1,), vals=None):

        """
        Declare gradients of multiple objective functions w.r.t. design variables.
        """
        if wrt not in self.design_variables_dict:
            raise Exception(
                f'Undeclared design variable {wrt} for objective gradients.'
            )

        if 'grad' not in self.declared_variables:
            self.declared_variables.append('grad')

        # Allow for multiple objectives
        if vals is not None:
            for objs in self.objs.keys():
                self.pF_px[objs, wrt] = vals

# # Override for specific problems if bounds are not available in this format, eg. csdl
    def declare_variable_bounds(self, x_lower, x_upper):
        self.x_lower = x_lower
        self.x_upper = x_upper

    # # Override for specific problems if bounds are not available in this format, eg. csdl
    def declare_constraint_bounds(self, c_lower, c_upper):
        self.c_lower = c_lower
        self.c_upper = c_upper

    def evaluate_lagrangian_hessian(self, x, y, lag_mult):
        hessian = np.zeros((len(x), len(x)))  # Initialize Hessian manually
        for i, obj_name in enumerate(self.obj.keys()):
            if (obj_name, obj_name) in self.p2F_pxx_dict:
                hessian += self.p2F_pxx_dict[obj_name, obj_name]['vals'] * lag_mult[i]
        return hessian

    def evaluate_penalty_hessian(self, x, y, rho):
        hessian = np.zeros((len(x), len(x)))
        for i, obj_name in enumerate(self.obj.keys()):
            if (obj_name, obj_name) in self.p2F_pxx_dict:
                hessian += self.p2F_pxx_dict[obj_name, obj_name]['vals'] * rho[i]
        return hessian

    # With Hessian-vector products, we can also compute products with the augmented Lagrangian Hessian
    def evaluate_hvp(self, x, y, lag_mult, rho, vx, vy):
        """
        Evaluate the Hessian-vector product along the direction vector specified, for given design and state vectors, for a given 
        Hessian (objective, penalty, Lagrangian, or Augmneted Lagrangian) specified by the Lagrange multipliers and/or penalty parameters.

        Parameters
        ----------
        x : np.ndarray
            Design variable vector.
        y : np.ndarray
            State variable vector.
        vx : np.ndarray
            x block of vector to be right-multiplied.
        vy : np.ndarray
            y block of vector to be right-multiplied.
        rho : np.ndarray
            Vector of penalty parameters.
        lag_mult : np.ndarray
            Lagrange multiplier vector.

        Returns
        -------
        wx : np.ndarray
            x block of the Hessian-vector product.
        wy : np.ndarray
            y block of the Hessian-vector product.
        """
        pass