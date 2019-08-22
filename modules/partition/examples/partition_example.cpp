#include "linalgcpp.hpp"
#include "partition.hpp"
#include "graphIO.hpp"
#include "randomGen.hpp"
#include "lubys_partition.hpp"
#include "condugate_gradient.hpp"
#include <cmath>
#include <chrono>
#include <ctime>

using namespace linalgcpp;



Vector<double> Solve_TL(const SparseMatrix<double>& A,
						const DenseMatrix& Ac, 
						const SparseMatrix<int>& P, 
						const DenseMatrix& AcInverse,
						Vector<double>(*Msolver)(const SparseMatrix<double>& , Vector<double>),
						Vector<double>(*MTsolver)(const SparseMatrix<double>& , Vector<double>),
						Vector<double> b){
	
	//2: "Pre-smooth" solve for x1/3
	Vector<double> x13 = Msolver(A,b);
	//3: compute restrictive residual
	Vector<double> rc = P.MultAT(b-A.Mult(x13));
	//4: solve for xc
	Vector<double> xc = AcInverse.Mult(rc);
	//5: fine-level approximation
	Vector<double> x23 = x13+Mult(P,xc);
	//6: compute and return x
	return MTsolver(A,b-A.Mult(x23))+x23;
}

Vector<double> PCG_TL(const SparseMatrix<double>& A, const Vector<double>& b,int max_iter,double tol,int Ncoarse){
	//level of difficulty: medium				    
	//assert A is s.p.d.
	//assert 1<= Ncoarse < A.Cols()
	Vector<int> partitions = Partition(A,Ncoarse);
	SparseMatrix<int> P = GetUnweightedInterpolator(partitions);
	
	DenseMatrix Ac = P.Transpose().Mult(A.Mult(P)).ToDense();
	
	DenseMatrix AcInverse;
	Ac.Invert(AcInverse);
	
    int n = A.Cols();
	
    Vector<double> x(n,0.0);
    Vector<double> r(b);
	Vector<double> pr = Solve_TL(A,Ac,P,AcInverse,*DLsolver,*DUsolver,r);
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
        pr = Solve_TL(A,Ac,P,AcInverse,*DLsolver,*DUsolver,r);
		deltaOld = delta;
		delta = r.Mult(pr);
		//std::cout<<"delta at iteration "<<k<<" is "<<delta<<std::endl;
        if(delta < tol * tol * delta0){
            //std::cout<<"converge at iteration "<<k<<std::endl;
			std::cout << std::setw(15) << k;
            return x;
        }
        p = pr + ((delta / deltaOld)* p);
    }
	
    std::cout<<"failed to converge in "<<max_iter<<" iterations"<<std::endl;
    return x;
	
}

Vector<double> Solve_ML(const std::vector<SparseMatrix<double>>& A,
						const std::vector<SparseMatrix<int>>& P,
						const DenseMatrix AcInverse,
						Vector<double>(*Msolver)(const SparseMatrix<double>& , Vector<double>),
						Vector<double>(*MTsolver)(const SparseMatrix<double>& , Vector<double>),
						const Vector<double>& b){
							
	int L = A.size() - 1;
	std::vector<Vector<double>> r(L+1);
	std::vector<Vector<double>> x(L+1);
	r[0]=b;
	
	for(int i=0;i<L;i++){
		x[i]=Msolver(A[i],r[i]);
		r[i+1]=P[i].MultAT(r[i]-A[i].Mult(x[i]));
	}
	
	x[L]= AcInverse.Mult(r[L]);
	
	for(int i=L-1;i>=0;i--){
		x[i]=x[i]+P[i].Mult(x[i+1]);
		x[i]=x[i]+MTsolver(A[i],r[i]-A[i].Mult(x[i]));
	}
	
	return x[0];
}

Vector<double> PCG_ML(const SparseMatrix<double>& A0, const Vector<double>& b,int max_iter,double tol, int Lmax, int Ncoarse){
	//assert A0 is s.p.d.
	//assert Lmax >= 1
	//assert A0.Cols()>=Ncoarse >= 1
	
	std::vector<int> N(Lmax+1);
	N[0]=A0.Cols();
	
	double q = std::min(pow(1.0*Ncoarse/N[0],1.0/Lmax),0.5);
	//std::cout<<"q = "<<q<<std::endl;
	
	int L = 0;// the index of last Nk
	for(int i = 1; i<Lmax;i++){
		N[i]=N[i-1]*q;
		//std::cout<<N[i]<<" ";
		if(N[i]!=0) L++;
		if(N[i]<=Ncoarse) break;
	}
	//std::cout<<std::endl;
	
	std::vector<SparseMatrix<int>> P(L);
	std::vector<SparseMatrix<double>> A(L+1);
	A[0]=A0;
	
	for(int i=0;i<L;i++){
		Vector<int> partitions = Partition(A[i],N[i+1]);
		P[i] = GetUnweightedInterpolator(partitions);
		A[i+1]= P[i].Transpose().Mult(A[i].Mult(P[i]));
		//std::cout<<"A["<<i+1<<"]"<<std::endl;
		///A[i+1].PrintDense();
	}
	
	DenseMatrix AcInverse;
	A[L].ToDense().Invert(AcInverse);
	
    Vector<double> x(N[0],0.0);
    Vector<double> r(b);
	Vector<double> pr = Solve_ML(A,P,AcInverse,*DLsolver,*DUsolver,r);
    Vector<double> p(pr);
    Vector<double> g(N[0]);
    double delta0 = r.Mult(pr);
    double delta = delta0, deltaOld, tau, alpha;

    for(int k=0;k<max_iter;k++){
        g = A[0].Mult(p);
        tau = p.Mult(g);
        alpha = delta / tau;
        x = x + (alpha * p);
        //x.Print("x at iteration: "+std::to_string(k));
        r = r - (alpha * g);
        pr = Solve_ML(A,P,AcInverse,*DLsolver,*DUsolver,r);
		deltaOld = delta;
		delta = r.Mult(pr);
		//std::cout<<"delta at iteration "<<k<<" is "<<delta<<std::endl;
        if(delta < tol * tol * delta0){
            //std::cout<<"converge at iteration "<<k<<std::endl;
			std::cout << std::setw(15) << k;
            return x;
        }
        p = pr + ((delta / deltaOld)* p);
    }
	
    std::cout<<"failed to converge in "<<max_iter<<" iterations"<<std::endl;
    return x;
	
}


void test_lubys(){
	
	SparseMatrix<double> ADJ = getWeightedAdjacency("data/wgraph_1.txt",0);
	ADJ.PrintDense("Adj");
	
	SparseMatrix<int> P = getP(ADJ);
	P.PrintDense("P");
	
	SparseMatrix<double> Ac = P.Transpose().Mult(ADJ.Mult(P));
	Ac.PrintDense("Ac");
	
	SparseMatrix<int> P1 = getP(Ac);
	P1.PrintDense("P1");
	
	SparseMatrix<double> Ac1 = P1.Transpose().Mult(Ac.Mult(P1));
	Ac1.PrintDense("Ac1");
}


void solver_test(){
	//get matrix 
	SparseMatrix<double> R = ReadCooList("data/matlabData/mat9.txt");
	int n = R.Cols();
	std::cout<<n<<std::endl;
	std::vector<int> row(n-1);
	for(int i=0;i<n-1;i++){
		row[i]=i+1;
	}
	std::vector<int> col=row;
	R = R.GetSubMatrix(row,col);
	//R.PrintDense();
	
	//generate right-hand-side
	Vector<double> b=RandVect(R.Cols(),1000);
	
	//save right-hand-side
	std::ofstream data;
	data.open("b9.txt",std::ios_base::app);
	for(int i=0;i<n-1;i++)
		data << b[i] <<"\n";
	data.close();
	
	double ini_time;
	double end_time;
	std::cout << std::setw(7) << "solver" << std::setw(15) << "iter" << std::setw(15) << "time" << std::endl << std::endl;
	
	std::cout << std::setw(7) << "CG"; 
	ini_time = omp_get_wtime();
    CG(R,b,10000,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;
	
	std::cout << std::setw(7) << "jacobi"; 
	ini_time = omp_get_wtime();
    PCG(R,b,Solve_Jacobian,10000,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;
	
	std::cout << std::setw(7) << "GS"; 
	ini_time = omp_get_wtime();
    PCG(R,b,Solve_Gauss_Seidel,10000,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;

	std::cout << std::setw(7) << "TL"; 
	ini_time = omp_get_wtime();
    PCG_TL(R, b,10000,1e-9,std::cbrt(R.Cols()));
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;

	std::cout << std::setw(7) << "ML"; 
	ini_time = omp_get_wtime();
    PCG_ML(R,b,10000,1e-9,100,std::cbrt(R.Cols()));
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;
	
}

void solver_test(const SparseMatrix<double>& R){
	//get matrix 
	std::cout << "hello?"<< std::endl;
	int n = R.Cols();
	
	//R.PrintDense();
	
	//generate right-hand-side
	Vector<double> b=RandVect(R.Cols(),10000);
	
	//save right-hand-side
	
	
	double ini_time;
	double end_time;
	std::cout << std::setw(7) << "solver" << std::setw(15) << "iter" << std::setw(15) << "time" << std::endl << std::endl;
	
	std::cout << std::setw(7) << "CG"; 
	ini_time = omp_get_wtime();
    CG(R,b,10000,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;
	
	std::cout << std::setw(7) << "jacobi"; 
	ini_time = omp_get_wtime();
    PCG(R,b,Solve_Jacobian,10000,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;
	
	std::cout << std::setw(7) << "GS"; 
	ini_time = omp_get_wtime();
    PCG(R,b,Solve_Gauss_Seidel,10000,1e-9);
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;

	std::cout << std::setw(7) << "TL"; 
	ini_time = omp_get_wtime();
    PCG_TL(R, b,10000,1e-9,std::cbrt(R.Cols()));
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;

	std::cout << std::setw(7) << "ML"; 
	ini_time = omp_get_wtime();
    PCG_ML(R,b,10000,1e-9,100,std::cbrt(R.Cols()));
	end_time = omp_get_wtime();
	std::cout << std::setw(15) << end_time-ini_time << std::endl;
	
}

int main()
{
    //SparseMatrix<double> fine_adjacency = ReadMTXList("data/simple_graph_1.edges");
	
	SparseMatrix<double> Laplacian = getLaplacian("data/sc-nasasrb.mtx",1,false);
    //Laplacian.PrintDense("LAP");
	//std::cout<<"read"<<std::endl;
	SparseMatrix<double> RLap = getReducedLaplacian(Laplacian);
	//RLap.PrintDense("reduced Laplacian");
	
	solver_test(RLap);
	//solver_test();
	
	
	
	//std::cout<<"=======solving by jacobian PCG======="<<std::endl;
    //PCG(RLap,b,Solve_Jacobian,1000,1e-9);
    //sol.Print("sol:");
	/**
	std::cout<<"=======solving by gauss-seidel PCG======="<<std::endl;
	PCG(RLap,b,Solve_Gauss_Seidel,1000,1e-9);//.Print("sol:");
	
	std::cout<<"=======start executing two-level======="<<std::endl;
	PCG_TL(RLap, b,100,1e-9,std::cbrt(RLap.Cols()));
	
	std::cout<<"=======start executing multi-level======="<<std::endl;
	PCG_ML(RLap,b,100,1e-9,7,std::cbrt(RLap.Cols()));
	*/
	
	//test_lubys();
}