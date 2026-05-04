import ctypes
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# set up logging
logger = logging.getLogger(__name__)

class MTDecomposer:
    """
    A robust Python wrapper for the compiled Fortran 'mtdcmp' library.
    Handles the conversion between Moment Tensor (MT) and Strike/Dip/Rake (SDR).
    """

    def __init__(self, lib_path: Optional[Union[str, Path]] = None):
        """
        Initialize the wrapper. Tries to locate the shared object file automatically.

        Args:
            lib_path: Explicit path to the .so file. If None, looks in the current directory.
        """
        if lib_path is None:
            # auto-detect: assume .so file is in the same directory as this python file
            current_dir = Path(__file__).parent.absolute()
            lib_path = current_dir / "mtdcmp.so"
        
        self.lib_path = Path(lib_path)
        self.lib: Optional[ctypes.CDLL] = None
        
        self._load_library()

    def _load_library(self):
        """Attempts to load the shared library and define argument types."""
        if not self.lib_path.exists():
            logger.warning(
                f"Fortran library not found at {self.lib_path}. "
                "SDR conversion functions (mt_to_sdr) will not work. "
                "Did you run 'make build'?"
            )
            return

        try:
            self.lib = ctypes.CDLL(str(self.lib_path))
            # define Fortran function signatures
            double_ptr = ctypes.POINTER(ctypes.c_double)
            
            if hasattr(self.lib, 'mt_to_sdr_'):
                self.lib.mt_to_sdr_.argtypes = [
                    double_ptr, double_ptr, double_ptr, # mxx, myy, mzz
                    double_ptr, double_ptr, double_ptr, # mxy, mxz, myz
                    double_ptr, double_ptr              # sdr1 output, sdr2 output
                ]
            else:
                logger.error("Function 'mt_to_sdr_' not found in the shared library.")
                self.lib = None

        except OSError as e:
            logger.error(f"Failed to load shared library: {e}")
            self.lib = None

    def mt_to_sdr(self, mt: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Convert Moment Tensor to Strike, Dip, Rake.
        
        Args:
            mt: A numpy array containing either:
                - 6 components: [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
                - 5 components: [Mxx, Myy, Mxy, Mxz, Myz] (Assumes Mzz = -(Mxx+Myy))
        
        Returns:
            Tuple of (sdr1, sdr2), where each is a numpy array [strike, dip, rake].
            Returns (zeros, zeros) if library is not loaded.
        """
        if self.lib is None:
            raise RuntimeError("MTDecomposer library is not loaded. Cannot perform conversion.")

        # process input
        if len(mt) == 6:
            mxx, myy, mzz, mxy, mxz, myz = mt
        elif len(mt) == 5:
            mxx, myy, mxy, mxz, myz = mt
            mzz = -(mxx + myy) # Trace=0 condition for deviatoric MT
        else:
            raise ValueError(f"Expected input mt to have 5 or 6 elements, got {len(mt)}")

        # Prepare ctypes
        c_mxx = ctypes.c_double(mxx)
        c_myy = ctypes.c_double(myy)
        c_mzz = ctypes.c_double(mzz)
        c_mxy = ctypes.c_double(mxy)
        c_mxz = ctypes.c_double(mxz)
        c_myz = ctypes.c_double(myz)

        # Prepare output containers
        sdr1 = np.zeros(3, dtype=np.float64)
        sdr2 = np.zeros(3, dtype=np.float64)

        # Call Fortran function
        self.lib.mt_to_sdr_(
            ctypes.byref(c_mxx), ctypes.byref(c_myy), ctypes.byref(c_mzz),
            ctypes.byref(c_mxy), ctypes.byref(c_mxz), ctypes.byref(c_myz),
            sdr1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            sdr2.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )

        return sdr1, sdr2