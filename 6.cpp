// Create a program that compares a parallel for loop and a standard for loop for summing rows of a View
// with Kokkos Timer
#include <iostream>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
	Kokkos::initialize(argc, argv);
	{

		int n_rows = 100;
		int n_cols = 100;

		Kokkos::View<int**> matrix("matrix", n_rows, n_cols);

		// Fill with some values
		Kokkos::parallel_for(n_rows, KOKKOS_LAMBDA(int i) {
			for (int j = 0; j < n_cols; j++) {
				matrix(i, j) = i + j;
			}
		});

		// Result views to store row sums
		Kokkos::View<int*> row_sums_parallel("sums_parallel", n_rows);
		Kokkos::View<int*> row_sums_serial("sums_serial", n_rows);

		// Parallel Version:
		Kokkos::Timer timer_parallel;

		Kokkos::parallel_for(n_rows, KOKKOS_LAMBDA(int i) {
			int sum = 0;
			for (int j = 0; j < n_cols; j++){
				sum += matrix(i, j);
			}
			row_sums_parallel(i) = sum;
		});

		// Wait for parallel work to finish
		Kokkos::fence(); 

		double time_parallel = timer_parallel.seconds();

		std::cout << time_parallel << std::endl;

		// Serial Version
		// Copy matrix to host for serial processing
		auto matrix_host = Kokkos::create_mirror_view_and_copy(
			Kokkos::HostSpace(), matrix);
		auto row_sums_serial_host = Kokkos::create_mirror_view(row_sums_serial);

		Kokkos::Timer timer_serial;

		// Standard serial for loop
		for (int i = 0; i < n_rows; i++) {
			int sum = 0;
			for (int j = 0; j < n_cols; j++) {
				sum += matrix_host(i, j);
			}
			row_sums_serial_host(i) = sum;
		}

		double time_serial = timer_serial.seconds();

		std::cout << time_serial << std::endl;
	}
	Kokkos::finalize();
	return 0;
}