// Do simple parallel reduce to output the maximum element in a View
#include <iostream>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
	Kokkos::initialize(argc, argv);
	{
		int n = 100;
		Kokkos::View<double*> view("view", n);

		// Fill with some values
		Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) {
			view(i) = i * i; // 0, 1, 4, 9, 16, ... , 9801
		});

		// Print out values
		Kokkos::fence();

		// Copy to host so we can print
		auto view_host = Kokkos::create_mirror_view_and_copy(
			Kokkos::HostSpace(), view);

		// NOW print (serially, on CPU)
		for (int i = 0; i < n; i++) {
			std::cout << view_host(i) << "\t";
			std::cout << std::endl;
		}	

		// Find the maximum using parallel reduce
		double max_value = 0.0;
		Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(int i, double& lmax) {
			if(view(i) > lmax) {
				lmax = view(i);
			}
		}, Kokkos::Max<double>(max_value));

		std::cout << "Maximum value: " << max_value << std::endl;
	}
	Kokkos::finalize();
	return 0;
}