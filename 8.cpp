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

    if (size != 3) {
        if (rank == 0)
            std::cout << "Need at least three processes to run this program." << std::endl;
        MPI_Finalize();
        return 1;
    }

    if (argc < 2) {
        if (rank == 0)
            std::cout << "Usage: " << argv[0] << " <array_size>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int N = std::stoi(argv[1]);

    int p1 = 333;
    int p2 = 666;
    std::vector<int> array;
    if (rank == 0) {
        array.resize(N);
        for (int i = 0; i < N; i++)
            array[i] = rand() % 1000;
    }    

    std::vector<int> bucket0, bucket1, bucket2;
    if (rank == 0) {
        for (int x : array) {
            if (x < p1)
                bucket0.push_back(x);
            else if (x < p2)
                bucket1.push_back(x);
            else
                bucket2.push_back(x);
        }
    }

    int local_size;
    if (rank == 0) {
        int s1 = bucket1.size();
        int s2 = bucket2.size();
        MPI_Send(&s1, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Send(&s2, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
        local_size = bucket0.size();
    } else if (rank == 1) {
        MPI_Recv(&local_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 2) {
        MPI_Recv(&local_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    std::vector<int> local(local_size);
    if (rank == 0) {
        MPI_Send(bucket1.data(), bucket1.size(), MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Send(bucket2.data(), bucket2.size(), MPI_INT, 2, 0, MPI_COMM_WORLD);
        local = bucket0;
    } else if (rank == 1) {
        MPI_Recv(local.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 2) {
        MPI_Recv(local.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    std::sort(local.begin(), local.end());

    int s1 = 0, s2 = 0;
    if (rank == 0) {
        s1 = bucket1.size();
        s2 = bucket2.size();
    }

    if (rank == 1) {
        MPI_Send(local.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else if (rank == 2) {
        MPI_Send(local.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else if (rank == 0) {
        std::vector<int> recv1(s1), recv2(s2);
        MPI_Recv(recv1.data(), s1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(recv2.data(), s2, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local.insert(local.end(), recv1.begin(), recv1.end());
        local.insert(local.end(), recv2.begin(), recv2.end());
        std::cout << "Sorted array: ";
        for (int x : local)
            std::cout << x << " ";
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}