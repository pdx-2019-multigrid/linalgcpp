#include <omp.h>
#include <assert.h>
#include "linalgcpp.hpp"

using namespace linalgcpp;

template <typename T>
Vector<T> paraSVM(const SparseMatrix<T>& A, const Vector<T>& x)
{

	std::vector<T> newdata = A.GetData();
	std::vector<int> newindptr = A.GetIndptr();
	std::vector<int> newindices = A.GetIndices();

	int M=A.Rows();
	Vector<T> toreturn(M);
	
#pragma omp parallel for num_threads(2)
	for(int i=0; i<M; ++i)
	{
	
		double sum = 0;
		
		int start = newindptr[i];
		int end = newindptr[i+1];	
		for(int j=start; j<end; ++j)
		{
		
			sum += x[newindices[j]]*newdata[j];

		}

		toreturn[i] = sum;
	}
	
	return toreturn;
}

template <typename T>
Vector<T> SVM(const SparseMatrix<T>& A, const Vector<T>& x)
{

	std::vector<T> newdata = A.GetData();
	std::vector<int> newindptr = A.GetIndptr();
	std::vector<int> newindices = A.GetIndices();

	int M=A.Rows();
	Vector<T> toreturn(M);
	

	for(int i=0; i<M; ++i)
	{
	
		double sum = 0;
		
		int start = newindptr[i];
		int end = newindptr[i+1];	
		for(int j=start; j<end; ++j)
		{
		
			sum += x[newindices[j]]*newdata[j];

		}

		toreturn[i] = sum;
	}
	
	return toreturn;
}

//==================================SparseMatrix-Vector=======================================================

template <typename T>
Vector<double> paraMult(const SparseMatrix<T>& A, const Vector<double>& b){
	int M = A.Rows();
	Vector<double> Ab(M);
	#pragma omp parallel for
	for(int i=0;i<M;i++){
		std::vector<int> indices = A.GetIndices(i);
		std::vector<T> data = A.GetData(i);
		double sum=0.0;
		int N = data.size();
		for(int j=0;j<N;j++){
			sum+=data[j]*b[indices[j]];
		}
		Ab[i]=sum;
	}
	return Ab;
}

template <typename T>
Vector<double> Mult(const SparseMatrix<T>& A, const Vector<double>& b){
	int M = A.Rows();
	Vector<double> Ab(M);
	for(int i=0;i<M;i++){
		std::vector<int> indices = A.GetIndices(i);
		std::vector<T> data = A.GetData(i);
		double sum=0.0;
		int N = data.size();
		for(int j=0;j<N;j++){
			sum+=data[j]*b[indices[j]];
		}
		Ab[i]=sum;
	}
	return Ab;
}

//==================================DenseMatrix-Vector=======================================

Vector<double> ParaMult(const DenseMatrix& A, const Vector<double>& b){
	int M=A.Rows();
	int N=A.Cols();
	Vector<double> Ab(M);
	#pragma omp parallel for
	for(int i=0;i<M;i++){
		double sum=0.0;
		for(int j=0;j<N;j++){
			sum+=A(i,j)*b[j];
		}
		Ab[i]=sum;
	}
	return Ab;
}

Vector<double> Mult(const DenseMatrix& A, const Vector<double>& b){
	int M=A.Rows();
	int N=A.Cols();
	Vector<double> Ab(M);
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			Ab[i]+=A(i,j)*b[j];
		}
	}
	return Ab;
}

//===============SparseMaxtrix-Matrix=====================



template <typename U, typename V>
SparseMatrix<double> paraMult(const SparseMatrix<U>& lhs, const SparseMatrix<V>& rhs)
{
    assert(rhs.Rows() == lhs.Cols());
	
	int cols_=lhs.Cols();
	int rows_=lhs.Rows();
	
	const std::vector<int>& indptr_=lhs.GetIndptr();
    const std::vector<int>& indices_=lhs.GetIndices();
	const std::vector<U>& data_=lhs.GetData();
	
	const std::vector<int>& rhs_indptr = rhs.GetIndptr();
    const std::vector<int>& rhs_indices = rhs.GetIndices();
    const std::vector<V>& rhs_data = rhs.GetData();
	
	std::vector<int> out_indptr(rows_ + 1);
    out_indptr[0] = 0;

    omp_set_num_threads(8);

	#pragma omp parallel
	{

	//printf("using %d threads\n",omp_get_num_threads());
	std::vector<int> marker(rhs.Cols());
    std::fill(begin(marker), end(marker), -1);
	#pragma omp for
    for (int i = 0; i < rows_; ++i){
		int row_nnz=0;
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j){
			for (int k = rhs_indptr[indices_[j]]; k < rhs_indptr[indices_[j] + 1]; ++k){
				if (marker[rhs_indices[k]] != static_cast<int>(i)){//<=================WHY static_cast?
					marker[rhs_indices[k]] = i;
					++row_nnz;
				}
			}
		}
		out_indptr[i + 1] = row_nnz;
	}
	
	}
	
	for(int i = 0; i < rows_; ++i){
		out_indptr[i + 1]+=out_indptr[i];
	}

    
    std::vector<int> out_indices(out_indptr[rows_]);
    std::vector<double> out_data(out_indptr[rows_]);
	
	
	#pragma omp parallel
	{
	
	std::vector<int> marker(rhs.Cols());
	std::fill(begin(marker), end(marker), -1);
    
	int total = 0;
	int zero_ptr = -1;
	#pragma omp for
    for (int i = 0; i < rows_; ++i)
    {
        int row_nnz = total;
		// at this point, all entries in marker <= row_nnz
		if(zero_ptr==-1){
			zero_ptr=out_indptr[i];
		}
		
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            for (int k = rhs_indptr[indices_[j]]; k < rhs_indptr[indices_[j] + 1]; ++k)
            {
                if (marker[rhs_indices[k]] < row_nnz)
                {
                    marker[rhs_indices[k]] = total;
                    out_indices[zero_ptr+total] = rhs_indices[k];
                    out_data[zero_ptr+total] = data_[j] * rhs_data[k];

                    total++;
                }
                else
                {
                    out_data[zero_ptr+marker[rhs_indices[k]]] += data_[j] * rhs_data[k];
                }
            }
        }
    }
	
	}
    return SparseMatrix<double>(std::move(out_indptr), std::move(out_indices), std::move(out_data),
                            rows_, rhs.Cols());
}




template <typename U, typename V>
SparseMatrix<double> Mult(const SparseMatrix<U>& lhs, const SparseMatrix<V>& rhs)
{
    assert(rhs.Rows() == lhs.Cols());

	int cols_=lhs.Cols();
	int rows_=lhs.Rows();
	
	const std::vector<int>& indptr_=lhs.GetIndptr();
    const std::vector<int>& indices_=lhs.GetIndices();
	const std::vector<U>& data_=lhs.GetData();
	
	std::vector<int> marker(rhs.Cols());
    std::fill(begin(marker), end(marker), -1);

    std::vector<int> out_indptr(rows_ + 1);
    out_indptr[0] = 0;

    int out_nnz = 0;

    const std::vector<int>& rhs_indptr = rhs.GetIndptr();
    const std::vector<int>& rhs_indices = rhs.GetIndices();
    const std::vector<V>& rhs_data = rhs.GetData();

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            for (int k = rhs_indptr[indices_[j]]; k < rhs_indptr[indices_[j] + 1]; ++k)
            {
                if (marker[rhs_indices[k]] != static_cast<int>(i))
                {
                    marker[rhs_indices[k]] = i;
                    ++out_nnz;
                }
            }
        }

        out_indptr[i + 1] = out_nnz;
    }
	
	
    std::fill(begin(marker), end(marker), -1);

    std::vector<int> out_indices(out_nnz);
    std::vector<double> out_data(out_nnz);
	
	
    int total = 0;

    for (int i = 0; i < rows_; ++i)
    {
        int row_nnz = total;

        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            for (int k = rhs_indptr[indices_[j]]; k < rhs_indptr[indices_[j] + 1]; ++k)
            {
                if (marker[rhs_indices[k]] < row_nnz)
                {
                    marker[rhs_indices[k]] = total;
                    out_indices[total] = rhs_indices[k];
                    out_data[total] = data_[j] * rhs_data[k];

                    total++;
                }
                else
                {
                    out_data[marker[rhs_indices[k]]] += data_[j] * rhs_data[k];
                }
            }
        }
    }

    return SparseMatrix<double>(std::move(out_indptr), std::move(out_indices), std::move(out_data),
                            rows_, rhs.Cols());
}

/**
TODO:
A^Tx
A^TB
*/
