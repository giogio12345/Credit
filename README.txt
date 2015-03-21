This library is a simple tool that can be used to price financial derivatives on credit and interest rate underlying with Monte Carlo and Quasi Monte Carlo methods, on CPU and multiple GPUs if available and with different models.
In particular, it can price N^th-to-Default Swap, Synthetic CDO and it can evaluate the expected exposure and the credit valuation adjustment of an Plain Interest Rate Swap.

*** INSTALLATION
There is not a real installation, just modify the Makefile in order to set your CUDA and CXX compiler and then type:
	make
and it will create static library lib/libcredit.a. Then, if you want to try some tests you can type:
	make test
	
*** DOCUMENTATION
The whole library is documented via Doxygen. So, if you type:
	doxygen doxy-config
you will find in the folder doc every information about the use of this library.

*** USE
To create a new program, just include "Credit.hpp" and use the library as in test sources. Then, compile with:
	$(NVCC) $(NVCC_FLAGS) -Llib -lcredit myfile.cpp -o myprogram
