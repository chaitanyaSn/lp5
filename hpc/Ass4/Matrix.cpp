#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// Function to perform matrix multiplication in parallel
void parallelMatrixMultiply(const vector<vector<double>> &A, const vector<vector<double>> &B, vector<vector<double>> &C)
{
    int n = A.size();
    int m = B[0].size();
    int k = B.size();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            double sum = 0;
            for (int p = 0; p < k; ++p)
            {
                sum += A[i][p] * B[p][j];
            }
            C[i][j] = sum;
        }
    }
}

int main()
{
    int n;
    cout << "Enter size of square matrices: ";
    cin >> n;

    vector<vector<double>> A(n, vector<double>(n));
    vector<vector<double>> B(n, vector<double>(n));
    vector<vector<double>> C(n, vector<double>(n, 0));

    cout << "Enter elements of Matrix A:" << endl;
    for (auto &row : A)
        for (auto &elem : row)
            cin >> elem;

    cout << "Enter elements of Matrix B:" << endl;
    for (auto &row : B)
        for (auto &elem : row)
            cin >> elem;

    double start = omp_get_wtime();
    parallelMatrixMultiply(A, B, C);
    double end = omp_get_wtime();

    cout << "\nResultant Matrix C:" << endl;
    for (const auto &row : C)
    {
        for (double elem : row)
        {
            cout << elem << " ";
        }
        cout << endl;
    }

    cout << "\nTime taken: " << (end - start) << " seconds" << endl;

    return 0;
}
