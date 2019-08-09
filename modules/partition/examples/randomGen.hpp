#include <random>
#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <ctime>
#include "linalgcpp.hpp"

using namespace linalgcpp;

double doubleRand() {
  return double(rand()) / (double(RAND_MAX) + 1.0);
}

/*! @brief Generate a random positive definite matrix in O(N^2)
	Note: the generated matrix is guaranteed to be at least semi-positive definite,
	with a negligible chance of not being positive definite
	
    @param n the demension of generated maxtrix
    @param range the matrix will have entry in (-range,range)
*/
DenseMatrix RandSPD(int n,double range){
	CooMatrix<double> M(n);
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			M.Add(i,j,doubleRand()*range+(i==j? 100:0));
		}
	}
	DenseMatrix MD = M.ToDense();
	return MD.MultAT(MD);
}

/*! @brief Generate a random vector in O(N)

    @param n the demension of generated vector
    @param range the vector will have entry in (-range,range)
*/
Vector<double> RandVect(int n,double range){
	Vector<double> v(n);
	for(int i=0;i<n;i++){
		v[i]=doubleRand()*range;
	}
	return v;
}