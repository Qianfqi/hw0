#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
using namespace std;
namespace py = pybind11;
// A: m x n, B: n x k, C: m x k
// C = A * B
void matmul(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < k; j++)
        {
            C[i * k + j] = 0.0;
            for(int p = 0; p < n; p++)
            {
                C[i * k + j] += A[i * n + p] * B[p * k + j];
            }
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // X: m x n, y: 1 x m, theta: n x k
    int times = (m + batch - 1) / batch;
    for(int i = 0; i < times; i++) 
    {
        const float * xx = &X[i * batch * n];
        float * Z = new float[batch * k];
        matmul(xx, theta, Z, batch, n, k);
        for(int j = 0; j < batch * k; j++)  Z[j] = exp(Z[j]);
        for(int j = 0; j < batch; j++)
        {
            float sum = 0.0;
            for(int p = 0; p < k; p++)  sum += Z[j * k + p];
            for(int p = 0; p < k; p++)  Z[j * k + p] /= sum;
        }
        const unsigned char * yy = &y[i * batch];
        float * Y = new float[batch * k];
        for(int j = 0; j < batch; j++)
        {
            for(int p = 0; p < k; p++)
            {
                if(yy[j] == p)  Y[j * k + p] = 1.0;
                else  Y[j * k + p] = 0.0;
            }
        }
        for(int j = 0; j < batch; j++)
        {
            for(int p = 0; p < k; p++)
            {
                Z[j * k + p] -= Y[j * k + p];
            }
        }
        float * x_T = new float[batch * n];
        for (int i = 0; i < batch; i++) 
            for (int j = 0; j < n; j++) 
                x_T[j * batch + i] = xx[i * n + j];
        float * grad = new float[n * k];
        matmul(x_T, Z, grad, n, batch, k);
        for(int j = 0; j < n * k; j++)  grad[j] /= batch;
        for(int j = 0; j < n * k; j++)  theta[j] -= lr * grad[j];
        delete[] Z;
        delete[] Y;
        delete[] x_T;
        delete[] grad;
    }
}
    
    // END YOUR CODE

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
