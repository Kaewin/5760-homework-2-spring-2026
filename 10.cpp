/*
Do a matrix multiply between a 2D View and a 1D View with at least one loop of parallelism. For full credit,
make a function and check for correct shape/dimensions. You can use test Views A and B from Problem 6.
*/
#include <iostream>
#include <Kokkos_Core.hpp>

void matrix_multiply(Kokkos::View<int*> vector, 
                     Kokkos::View<int**> matrix,
                     Kokkos::View<int*> result) {

	// Check for proper dimensions
	if (vector.extent(0) != 3) {
		std::cout << "Wrong dimensions. Only takes 1x3 vector." << std::endl;
		return;
	} if (matrix.extent(0) != 3 || matrix.extent(1) != 3) {
		std::cout << "Matrix has wrong dimensions. Only takes 3x3 matrix." << std::endl;
		return;
	}
	// For each row of matrix
	Kokkos::parallel_for(3, KOKKOS_LAMBDA(int i) {
		int sum = 0;                        // Initialize sum for this row
		for (int j = 0; j < 3; j++) {       // For each column of Matrix
			sum += matrix(i, j) * vector(j);// Multiply and accumulate
		}
		result(i) = sum; 
	});
}

int main(int argc, char** argv) {
	Kokkos::initialize(argc, argv);
	{
		Kokkos::View<int*>  vector("Vector", 3);
		Kokkos::View<int**>  matrix("Matrix", 3, 3);
		Kokkos::View<int*>  result("result", 3);

		// Fill with some values
		vector(0) = 221;
		vector(1) = 12;
		vector(2) = 157;

		matrix(0,0) = 130;
		matrix(0,1) = 147;
		matrix(0,2) = 115;
	
		matrix(1,0) = 224;
		matrix(1,1) = 158;
		matrix(1,2) = 187;

		matrix(2,0) = 54;
		matrix(2,1) = 158;
		matrix(2,2) = 120;

		matrix_multiply(vector, matrix, result);

		// Copy to host so we can print
		auto view_host = Kokkos::create_mirror_view_and_copy(
			Kokkos::HostSpace(), result);

		// NOW print (serially, on CPU)
		for (int i = 0; i < 3; i++) {
			std::cout << view_host(i) << "\t";
			std::cout << std::endl;
		}	
	}
	Kokkos::finalize();
	return 0;
}

