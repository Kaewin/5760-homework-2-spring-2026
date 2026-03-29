/*
Link and run program with Kokkos where you initialize a View and print out its name with the .label()
method.
*/
#include <iostream>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
    	Kokkos::initialize(argc, argv);

    	{

		int n = 8;

		// Creates a view:
		// Kokkos::View<int*> - 1D array of integers
		// "check" is the label (name) of the view
		// n - Gives the array a size of 8 elements
    	Kokkos::View<int*> check("check", n);

		// This function returns the label of the view
		std::cout << check.label() << std::endl;

    	}

    	Kokkos::finalize();
    	return 0;
}