/*! @file

    @brief A collection of brief tests to make sure
          things do what I expect them.  None of these
          checks are automated yet, but will be in the near
          future.
*/
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

Vector<double> entrywise_mult(const Vector<double>& a, const Vector<double>& b){
	//assert a.size()==b.size()
	Vector<double> c(a.size());
	for(int k=0;k<a.size();++k){
		c[k]=a[k]*b[k];
	}
	return c;
}

Vector<double> entrywise_inv(const Vector<double>& a){
	//assert a.size()==b.size()
	Vector<double> c(a.size());
	for(int k=0;k<a.size();++k){
		c[k]=1.0/a[k];
	}
	return c;
}


/*! @brief Solve system Mx=r, where M is the symmetric Gauss-Seidel matrix of A, in O(N^2)

    @param A the dense maxtrix from which we generate M
    @param r the right-hand-side of the system
*/
Vector<double> Solve_Gauss_Seidel(const DenseMatrix& A, Vector<double> r){
	//step 1: solve the lower triangular system for y: (D+L)y=r
	int n = A.Cols();
	for(int i=0;i<n;++i){
		r[i]/=A(i,i);
		for(int j=i+1;j<n;++j){
			r[j]-=r[i]*A(j,i);
		}
	}
	
	//step 2: solve the upper triangular system for x: (D+U)x=Dy
	r=entrywise_mult(Vector<double>(&A.GetDiag()[0],n),r);
	
	for(int i=n-1;i>=0;--i){
		r[i]/=A(i,i);
		for(int j=i-1;j>=0;--j){
			r[j]-=r[i]*A(j,i);
		}
	}
	
	return r;
	
}

/*! @brief Solve system Mx=r, where M is the symmetric Gauss-Seidel matrix of A, in O(# of non-zero entries in A)

    @param A the sparse maxtrix from which we generate M
    @param r the right-hand-side of the system
*/
Vector<double> Solve_Gauss_Seidel(const SparseMatrix<double>& A, Vector<double> r){
	//step 1: solve the lower triangular system for y: (D+L)y=r
	//check sortedness
	int n = A.Cols();
	SparseMatrix<double> AT = A.Transpose();
	
	for(int i=0;i<n;++i){
		std::vector<int> indices = AT.GetIndices(i);
		std::vector<double> data = AT.GetData(i);
		for(int j=0;j<data.size();j++){
			if(indices[j]<i){
				continue;
			}
			if(indices[j]==i){
				r[i]/=data[j];
				continue;
			}
			r[indices[j]]-=r[i]*data[j];
		}
	}
	//r.Print("After L elimination");
	
	//step 2: solve the upper triangular system for x: (D+U)x=Dy
	r=entrywise_mult(Vector<double>(&A.GetDiag()[0],n),r);
	//r.Print("After mult diag");
	
	for(int i=n-1;i>=0;--i){
		std::vector<int> indices = AT.GetIndices(i);
		std::vector<double> data = AT.GetData(i);
		for(int j=data.size()-1;j>=0;j--){
			if(indices[j]>i){
				continue;
			}
			if(indices[j]==i){
				r[i]/=data[j];
				continue;
			}
			r[indices[j]]-=r[i]*data[j];
		}
	}
	//r.Print("After U elimination");
	return r;
	
}

Vector<double> Solve_Jacobian(const DenseMatrix& A, Vector<double> r){
	//step 1: solve the lower triangular system for y: (D+L)y=r
	int n = A.Cols();
	for(int i=0;i<n;++i){
		r[i]/=A(i,i);
	}
	return r;
}


Vector<double> PCG(const DenseMatrix& A, const Vector<double>& b, Vector<double>(*Msolver)(const DenseMatrix& , Vector<double>),int max_iter,double tol){
	//level of difficulty: medium				    
	//assert A is s.p.d.
	
    int n = A.Cols();
	
    Vector<double> x(n,0.0);
    Vector<double> r(b);
	Vector<double> pr = Msolver(A,r);
    Vector<double> p(pr);
    Vector<double> g(n);
    double delta0 = r.Mult(pr);
    double delta = delta0, deltaOld, tau, alpha;

    for(int k=0;k<max_iter;k++){
        g = A.Mult(p);
        tau = p.Mult(g);
        alpha = delta / tau;
        x = x + (alpha * p);
        //x.Print("x at iteration: "+std::to_string(k));
        r = r - (alpha * g);
        pr = Msolver(A,r);
		deltaOld = delta;
		delta = r.Mult(pr);
		std::cout<<"delta at iteration "<<k<<" is "<<delta<<std::endl;
        if(delta < tol * tol * delta0){
            std::cout<<"converge at iteration "<<k<<std::endl;
            return x;
        }
        p = pr + ((delta / deltaOld)* p);
    }

    std::cout<<"failed to converge in "<<max_iter<<" iterations"<<std::endl;
    return x;
	
}


Vector<double> PCG(const SparseMatrix<double>& A, const Vector<double>& b, Vector<double>(*Msolver)(const SparseMatrix<double>& , Vector<double>),int max_iter,double tol){
	//level of difficulty: medium				    
	//assert A is s.p.d.
	
    int n = A.Cols();
	
    Vector<double> x(n,0.0);
    Vector<double> r(b);
	Vector<double> pr = Msolver(A,r);
    Vector<double> p(pr);
    Vector<double> g(n);
    double delta0 = r.Mult(pr);
    double delta = delta0, deltaOld, tau, alpha;

    for(int k=0;k<max_iter;k++){
        g = A.Mult(p);
        tau = p.Mult(g);
        alpha = delta / tau;
        x = x + (alpha * p);
        //x.Print("x at iteration: "+std::to_string(k));
        r = r - (alpha * g);
        pr = Msolver(A,r);
		deltaOld = delta;
		delta = r.Mult(pr);
		std::cout<<"delta at iteration "<<k<<" is "<<delta<<std::endl;
        if(delta < tol * tol * delta0){
            std::cout<<"converge at iteration "<<k<<std::endl;
            return x;
        }
        p = pr + ((delta / deltaOld)* p);
    }

    std::cout<<"failed to converge in "<<max_iter<<" iterations"<<std::endl;
    return x;
	
}

/*! @brief The regular condugate gradient method, time complexity O(max_iter*N^2)

    @param A an s.p.d. matrix
    @param b the right-hand-side of system
    @param max_iter maximum number of iteration before exit
	@param tol epsilon
*/
Vector<double> CG(const DenseMatrix& A, const Vector<double>& b, int max_iter,double tol){
    //assert A is s.p.d.
    int n = A.Cols();

    Vector<double> x(n,0.0);
    Vector<double> r(b);
    Vector<double> p(r);
    Vector<double> g(n);
    double delta0 = b.Mult(b);
    double delta = delta0, deltaOld, tau, alpha;

    for(int k=0;k<max_iter;k++){
        g = A.Mult(p);
        tau = p.Mult(g);
        alpha = delta / tau;
        x = x + (alpha * p);
        //x.Print("x at iteration: "+std::to_string(k));
        r = r - (alpha * g);
        deltaOld = delta;
		delta = r.Mult(r);
		std::cout<<"delta at iteration "<<k<<" is "<<delta<<std::endl;
        if(delta < tol * tol * delta0){
            std::cout<<"converge at iteration "<<k<<std::endl;
            return x;
        }
        p = r + ((delta / deltaOld)* p);
    }

    std::cout<<"failed to converge in "<<max_iter<<" iterations"<<std::endl;
    return x;
}

int main(int argc, char** argv)
{
	
	std::cout<<"hello function pointer"<<std::endl;
	
	/**
	CooMatrix<int> L(3, 3);
	L.Add(0,0,2);
	L.Add(1,0,4);
	L.Add(1,1,1);
	L.Add(2,0,1);
	L.Add(2,1,-1);
	L.Add(2,2,3);
	L.Add(0,2,1);
	L.Add(1,2,2);
	
	DenseMatrix LD  =L.ToDense();
	LD.Print();
	
	Vector<double> r(3);
	r[0]=4;
    r[1]=3;
    r[2]=10;
	
	r.Print("r: before");
	
	Solve_Gauss_Seidel(LD,r).Print("sol to L system");
	
	r.Print("r: after");
	*/
	
	
    CooMatrix<double> M(3, 3);
    M.AddSym(0, 0, 53);
	M.AddSym(1, 1, 61);
	M.AddSym(2, 2, 79);
    M.AddSym(0, 1, 2);
	M.AddSym(0, 2, 3);
	//M.AddSym(1, 2, 19);
    M.Print("M:");
    DenseMatrix MD = M.ToDense();
	MD.Print("MD:");
	
	
	//std::vector<double> vect;
	//MD.CopyData(vect);
	//std::cout<<vect<<std::endl;
	
    Vector<double> v(3);
    v[0]=20;
    v[1]=30;
    v[2]=40;
    v.Print("v:");
    
	
	/**
	DenseMatrix MD = RandSPD(100,20);
	//SPD.Print("A:");
	
	Vector<double> v = RandVect(100,100);
	//b.Print("b:");
	*/
	
	std::cout<<"=======solving by regular CG======="<<std::endl;
	CG(MD,v,1000,1e-9);//.Print("sol:");
	
	std::cout<<"=======solving by jacobian PCG======="<<std::endl;
    PCG(MD,v,Solve_Jacobian,1000,1e-9);
    //sol.Print("sol:");
	
	std::cout<<"=======solving by gauss-seidel PCG (Dense)======="<<std::endl;
	PCG(MD,v,Solve_Gauss_Seidel,1000,1e-9);//.Print("sol:");
	
	std::cout<<"=======solving by gauss-seidel PCG (Sparse)======="<<std::endl;
	SparseMatrix<double> MS = M.ToSparse();
	PCG(MS,v,Solve_Gauss_Seidel,1000,1e-9);//.Print("sol:");
	
	
	/**
	CooMatrix<double> M(4, 4);
	M.Add(0,0,1);
	M.Add(1,1,2);
	M.Add(2,1,1);
	M.Add(2,2,3);
	M.Add(3,0,1);
	M.Add(3,3,1);
	
	M.Add(0,1,2);
	M.Add(0,2,2);
	M.Add(0,3,1);
	M.Add(2,3,-1);
	SparseMatrix<double> SM=M.ToSparse();
	
	M.ToDense().Print("M");
	
	
	Vector<double> v(4);
    v[0]=1;
    v[1]=4;
    v[2]=11;
	v[3]=0;
    v.Print("v:");
	
	Solve_Gauss_Seidel(SM,v);
	*/
	/**
	std::cout<<"====PCG dense===="<<std::endl;
	PCG_Dense(M.ToDense(),v,Solve_Gauss_Seidel,20,1e-9);
	std::cout<<"====PCG sparse===="<<std::endl;
	PCG_Sparse(M.ToSparse(),v,Solve_Gauss_Seidel,20,1e-9);
	*/
    return EXIT_SUCCESS;
}
