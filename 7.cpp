// Create a program that does adds the elements of a 1D View (a vector) to each row of a 2D View (a matrix)
// with at least one loop of parallelism.
#include <iostream>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
	Kokkos::initialize(argc, argv);

	{
		Kokkos::View<int*> vector("vector", 3);
		Kokkos::View<int**> matrix("matrix", 3, 3);

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

		// Copy to host
		auto view_host = Kokkos::create_mirror_view_and_copy(
		Kokkos::HostSpace(), matrix);
	
		// Print it out (4 nested loops)
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				std::cout << view_host(i, j) << " ";
			}
			std::cout << std::endl;
		}

		Kokkos::parallel_for(3, KOKKOS_LAMBDA(int i) {
			for (int j = 0; j < 3; j++){
				matrix(i, j) += vector(j);
			}
			});

		// Copy to host
		auto view_host1 = Kokkos::create_mirror_view_and_copy(
		Kokkos::HostSpace(), matrix);
	
		// Print it out (4 nested loops)
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				std::cout << view_host1(i, j) << " ";
			}
			std::cout << std::endl;
		}
	}
	Kokkos::finalize();
	return 0;
}