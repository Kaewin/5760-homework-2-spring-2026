// Conduct a prefix sum using the Kokkos parallel scan function. Use Kokkos timer. Print out resulting partial
// sum Views along with timer results. Run several times to compare results
#include <iostream>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
	Kokkos::initialize(argc, argv);
	{
		const int n = 20;
		const int num_trials = 5;

		// Input view
		Kokkos::View<int*> input("input", n);
		Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) {
			input(i) = i + 1;
		});

		// Output view
		Kokkos::View<int*> output("output", n);

		std::cout << "Running prefix sum " << num_trials << " times on " << n << " elements\n" << std::endl;

		// Run multiple trials
		for (int trial = 0; trial < num_trials; trial++) {
			// Reset output
			Kokkos::deep_copy(output, 0);

			// Timer
			Kokkos::Timer timer;

			Kokkos::parallel_scan(n, KOKKOS_LAMBDA(const int i, int& update, const bool final) {
				// Update holds the running sum
				const int val_i = input(i);

				// Add current value to running sum
				update += val_i;

				if (final) {
					// Final pass - write result
					output(i) = update;
				}
			});

			Kokkos::fence(); // Wait for completion
			double time = timer.seconds();

			std::cout << "Trial " << trial + 1 << ": " << time << " seconds" << std::endl;
		}

		// Copy results to host for printing
		auto input_host = Kokkos::create_mirror_view_and_copy(
			Kokkos::HostSpace(), input);
		auto output_host = Kokkos::create_mirror_view_and_copy(
			Kokkos::HostSpace(), output);

		// Print the results
		std::cout << "\nInput array:" << std::endl;
		for (int i = 0; i < n; i++) {
			std::cout << input_host(i) << " ";
		}
		std::cout << "\nOutput array:" << std::endl;
		for (int i = 0; i < n; i++) {
			std::cout << output_host(i) << " ";
		}
		std::cout << std::endl;
	}
	Kokkos::finalize();
	return 0;
}