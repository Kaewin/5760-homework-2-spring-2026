#include <iostream>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cstdlib>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            std::cout << "Usage: " << argv[0] << " <array_size>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int N = std::stoi(argv[1]);

    if (size < 2) {
        if (rank == 0)
            std::cout << "Need at least two processes to run this program." << std::endl;
        MPI_Finalize();
        return 1;
    }

    if (N % size != 0) {
        if (rank == 0)
            std::cout << "N must be divisible by the number of processes." << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::vector<int> array;

    if (rank == 0) {
        array.resize(N);
        for (int i = 0; i < N; i++)
            array[i] = rand() % 1000;
    }

    int chunk_size = N / size;
    std::vector<int> local(chunk_size);

    MPI_Scatter(
        array.data(),    // send buffer (only matters on rank 0)
        chunk_size,      // how many elements to send to each process
        MPI_INT,
        local.data(),    // receive buffer
        chunk_size,      // how many elements to receive
        MPI_INT,
        0,               // root
        MPI_COMM_WORLD
    );

    std::sort(local.begin(), local.end());

    for (int stride = 1; stride < size; stride *= 2) {
        if (rank % (2 * stride) == 0) {
            int partner = rank + stride;
            if (partner < size) {
                int recv_size = local.size();
                std::vector<int> recv_buf(recv_size);
                MPI_Recv(recv_buf.data(), recv_size, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<int> merged(local.size() + recv_buf.size());
                std::merge(local.begin(), local.end(), recv_buf.begin(), recv_buf.end(), merged.begin());
                local = merged;
            }
        } else if (rank % (2 * stride) == stride) {
            int partner = rank - stride;
            MPI_Send(local.data(), local.size(), MPI_INT, partner, 0, MPI_COMM_WORLD);
            break;
        }
    }

    if (rank == 0) {
        std::cout << "Sorted array: ";
        for (int x : local)
            std::cout << x << " ";
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}