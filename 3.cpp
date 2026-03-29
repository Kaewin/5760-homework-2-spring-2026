// Make an n × m View where each index equals 1000 × i × j
#include <iostream>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
	Kokkos::initialize(argc, argv);

	// Include these or else you get an error:
	// Kokkos allocation "View" is being deallocated after Kokkos::finalize was called
	{
	int n = 5;
	int m = 5;

	Kokkos::View<int**> view("View", n, m);

	std::cout << view.label() << std::endl;

	// Access using parentheses (like Fortran)
	Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) {
		for (int j = 0; j < m; j++) {
			view(i, j) = 1000 * i * j;  // Set values
			// Don't do this:
			// std::cout is a CPU operation
			// Kokkos might run on GPU
			// std::cout << view(i, j) << std::endl;
		}
	});

	// Copy to host so we can print
	auto view_host = Kokkos::create_mirror_view_and_copy(
		Kokkos::HostSpace(), view);
	
	// NOW print (serially, on CPU)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			std::cout << view_host(i, j) << "\t";
		}
		std::cout << std::endl;

	}	

	}

	Kokkos::finalize();
	return 0;
}