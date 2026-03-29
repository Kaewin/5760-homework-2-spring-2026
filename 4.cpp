// Declare a 5 × 7 × 12 × n View
#include <iostream>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
	Kokkos::initialize(argc, argv);

	{

	int n = 10;

	Kokkos::View<int****> view("view", 5, 7, 12, n);
	std::cout << view.label() << std::endl;

	// Fill the 4D view in parallel
	Kokkos::parallel_for(5, KOKKOS_LAMBDA(int i) {
		for (int j = 0; j < 7; j++) {
			for (int k = 0; k < 12; k++) {
				for (int l = 0; l < n; l++) {
					view(i, j, k, l) = i + j + k + l;
				}
			}
		}
	});

	// Copy to host
	auto view_host = Kokkos::create_mirror_view_and_copy(
		Kokkos::HostSpace(), view);
	
	// Print it out (4 nested loops)
	for (int i = 0; i < 5; i++) {
		std::cout << "i = " << i << std::endl;
		for (int j = 0; j < 7; j++) {
			std::cout << "  j = " << j << std::endl;
			for (int k = 0; k < 12; k++) {
				std::cout << "    k = " << k << ": ";
				for (int l = 0; l < n; l++) {
					std::cout << view_host(i, j, k, l) << " ";
				}
				std::cout << std::endl;
			}
		}
	}

	}

	Kokkos::finalize();
	return 0;
}