#include <omp.h>
#include <assert.h>
#include "linalgcpp.hpp"

using namespace linalgcpp;

/**
Vector<double> paraMult(const SparseMatrix<double>& A, const Vector<double>& b){
    const int aRows = A.Rows();
    Vector<double> Ab(aRows);
    using std::vector;
    vector<vector<int>> indArr(aRows);
    vector<vector<double>> datArr(aRows);
    vector<int> dataSizes(aRows);

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < aRows; i++){
        indArr[i] = A.GetIndices(i);
        datArr[i] = A.GetData(i);
        dataSizes[i] = datArr[i].size();
    }
    #pragma omp parallel for schedule(dynamic,1)
    for(int j = 0; j < dataSizes[i]; j++){
		for(int i = 0; i < aRows; i++){
            Ab[i] += datArr[i][j] * b[indArr[i][j]];
        }
    }
    return Ab;
}

Vector<double> paraMult(const SparseMatrix<double>& A, const Vector<double>& b){
	Vector<double> Ab(A.Rows());
	std::vector<int> indArr[A.Rows()];
	std::vector<double> datArr[A.Rows()];
	#pragma omp parallel for schedule(dynamic)
	for(int i=0;i<A.Rows();i++){
		indArr[i] = A.GetIndices(i);
		datArr[i] = A.GetData(i);
	}
	
	#pragma omp parallel for
	for(int i=0;i<A.Rows();i++){
		double sum=0.0;
		for(int j=0;j<datArr[i].size();j++){
			sum+=datArr[i][j]*b[indArr[i][j]];
		}
		Ab[i]=sum;
	}
	return Ab;
}

*/

//==================================SparseMatrix=======================================================


Vector<double> paraSVM(const SparseMatrix<double>& A, const Vector<double>& x)
{

	std::vector<double> newdata = A.GetData();
	std::vector<int> newindptr = A.GetIndptr();
	std::vector<int> newindices = A.GetIndices();

	std::vector<double> newx = x.data();
	std::vector<double> toreturn(newx.size());
	
	int M = A.Rows();
#pragma omp parallel for num_threads(2)
	for(int i=0; i<M; ++i)
	{
	
		double sum = 0;
		
	
		for(int j=newindptr[i]; j<newindptr[i+1]; ++j)
		{
		
			sum += newx[newindices[j]]*newdata[j];

		}

		toreturn[i] = sum;
	}
	
	
	Vector<double> toreturn2(toreturn);
	return toreturn2;
}


Vector<double> SVM(const SparseMatrix<double>& A, const Vector<double>& x)
{

	std::vector<double> newdata = A.GetData();
	std::vector<int> newindptr = A.GetIndptr();
	std::vector<int> newindices = A.GetIndices();

	std::vector<double> newx = x.data();
	std::vector<double> toreturn(newx.size());
	
	int M = A.Rows();

	for(int i=0; i<M; ++i)
	{
	
		double sum = 0;
		
	
		for(int j=newindptr[i]; j<newindptr[i+1]; ++j)
		{
		
			sum += newx[newindices[j]]*newdata[j];

		}

		toreturn[i] = sum;
	}
	
	
	Vector<double> toreturn2(toreturn);
	return toreturn2;
}

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

//==================================DenseMatrix=======================================

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

//===============Maxtrix-matrix=====================



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
A^doubleB
*/
