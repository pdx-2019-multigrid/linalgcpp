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


DenseMatrix Ainv;
Vector<double> LUsolver(const DenseMatrix& A, Vector<double> b){//!!inefficient
	//compute A^-1 beforehand
	//A.Invert(Ainv);
	return Ainv.Mult(b);
}



Vector<double> Solve_TL(const SparseMatrix<double>& A,
						const DenseMatrix& Ac, 
						const SparseMatrix<int>& P, 
						Vector<double>(*Asolver)(const DenseMatrix& , Vector<double>),
						Vector<double>(*Msolver)(const SparseMatrix<double>& , Vector<double>),
						Vector<double>(*MTsolver)(const SparseMatrix<double>& , Vector<double>),
						Vector<double> b,
						bool para){
	
	//2: "Pre-smooth" solve for x1/3
	Vector<double> x13 = Msolver(A,b);
	//3: compute restrictive residual
	Vector<double> rc = P.MultAT(b-(para? paraMult(A,x13):Mult(A,x13)));
	//4: solve for xc
	Vector<double> xc = Asolver(Ac,rc);
	//5: fine-level approximation
	Vector<double> x23 = x13+(para? paraMult(P,xc):Mult(P,xc));
	//6: compute and return x
	return MTsolver(A,b-(para? paraMult(A,x23):Mult(A,x23)))+x23;
}

Vector<double> PCG_TL(const SparseMatrix<double>& A, const Vector<double>& b,int max_iter,double tol,int Ncoarse,bool para){
	//level of difficulty: medium				    
	//assert A is s.p.d.
	//assert 1<= Ncoarse < A.Cols()
	Vector<int> partitions = Partition(A,Ncoarse);
	SparseMatrix<int> P = GetUnweightedInterpolator(partitions);
	
	DenseMatrix Ac;
	if(para)
		Ac = paraMult(P.Transpose(),paraMult(A,P)).ToDense();
	else
		Ac = P.Transpose().Mult(A.Mult(P)).ToDense();
	
	Ac.Invert(Ainv);
	
    int n = A.Cols();
	
    Vector<double> x(n,0.0);
    Vector<double> r(b);
	Vector<double> pr = Solve_TL(A,Ac,P,*LUsolver,*DLsolver,*DUsolver,r,para);
    Vector<double> p(pr);
    Vector<double> g(n);
    double delta0 = r.Mult(pr);
    double delta = delta0, deltaOld, tau, alpha;

    for(int k=0;k<max_iter;k++){
        if(para)
			g = paraMult(A,p);
		else
			g = A.Mult(p);
        tau = p.Mult(g);
        alpha = delta / tau;
        x = x + (alpha * p);
        //x.Print("x at iteration: "+std::to_string(k));
        r = r - (alpha * g);
        pr = Solve_TL(A,Ac,P,*LUsolver,*DLsolver,*DUsolver,r,para);
		deltaOld = delta;
		delta = r.Mult(pr);
		//std::cout<<"delta at iteration "<<k<<" is "<<delta<<std::endl;
        if(delta < tol * tol * delta0){
            //std::cout<<"converge at iteration "<<k<<std::endl;
            return x;
        }
        p = pr + ((delta / deltaOld)* p);
    }
	
    std::cout<<"failed to converge in "<<max_iter<<" iterations"<<std::endl;
    return x;
	
}

Vector<double> Solve_ML(const std::vector<SparseMatrix<double>>& A,
						const std::vector<SparseMatrix<int>>& P,
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
	
	DenseMatrix Ac = A[L].ToDense();
	Ac.Invert();
	x[L]= Ac.Mult(r[L]);
	
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
	std::cout<<"Hello World"<<std::endl;
	std::vector<int> N(Lmax+1);
	N[0]=A0.Cols();
	
	double q = std::min(pow(1.0*Ncoarse/N[0],1.0/Lmax),0.5);
	std::cout<<"q = "<<q<<std::endl;
	
	int L = 0;// the index of last Nk
	for(int i = 1; i<Lmax;i++){
		N[i]=N[i-1]*q;
		std::cout<<N[i]<<" ";
		if(N[i]!=0) L++;
		if(N[i]<=Ncoarse) break;
	}
	std::cout<<std::endl;
	
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
	
	
    Vector<double> x(N[0],0.0);
    Vector<double> r(b);
	Vector<double> pr = Solve_ML(A,P,*DLsolver,*DUsolver,r);
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
        pr = Solve_ML(A,P,*DLsolver,*DUsolver,r);
		deltaOld = delta;
		delta = r.Mult(pr);
		//std::cout<<"delta at iteration "<<k<<" is "<<delta<<std::endl;
        if(delta < tol * tol * delta0){
            std::cout<<"converge at iteration "<<k<<std::endl;
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

void test_paraMult(){
	SparseMatrix<double> Laplacian = getLaplacian("data/6473.edges",0,false);
	SparseMatrix<double> LT = Laplacian.Transpose();
	
	std::cout<<"=======Matrix-Matrix Mult======="<<std::endl;
	std::cout << std::setw(7) << "trial#" << std::setw(15) << "para" << std::setw(15) << "no para" << std::endl << std::endl;
    double ini_time;
	double end_time;
	for(int i=1;i<=4;i++){
		ini_time = omp_get_wtime();
		paraMult(LT,Laplacian);
		end_time = omp_get_wtime();
		std::cout << std::setw(7) << i << std::setw(15) << end_time-ini_time;
		
		ini_time = omp_get_wtime();
		//Mult<double,double,double>(LT,Laplacian);
		LT.Mult(Laplacian);
		end_time = omp_get_wtime();
		std::cout << std::setw(15) << end_time-ini_time << std::endl;
    
	}
	
}

void test_paraCG(const SparseMatrix<double>& RLap){
	
	Vector<double> b=RandVect(RLap.Cols(),1000);
	//b.Print("b");
	std::cout<<"=======solving by regular CG======="<<std::endl;
	std::cout << std::setw(7) << "trial#" << std::setw(30) << "para(wtime/systemtime)" << std::setw(30) << "no para(wtime/systemtime)" << std::endl << std::endl;
    
	double ini_time;
	double end_time;
	
	std::chrono::high_resolution_clock::time_point t1;
	std::chrono::high_resolution_clock::time_point t2;
	std::chrono::duration<double> time_span;
	
	for(int i=1;i<=4;i++){
		ini_time = omp_get_wtime();
		t1 = std::chrono::high_resolution_clock::now();
		omp_set_num_threads(2);
		CG(RLap,b,1000,1e-9,true);//.Print("sol:");
		
		end_time = omp_get_wtime();
		t2 = std::chrono::high_resolution_clock::now();
		
		time_span = t2-t1;
		std::cout << std::setw(7) << i << std::setw(20) << end_time-ini_time<<'/'<<time_span.count();
		
		ini_time = omp_get_wtime();
		t1 = std::chrono::high_resolution_clock::now();
		
		CG(RLap,b,1000,1e-9,false);//.Print("sol:");
		
		end_time = omp_get_wtime();
		t2 = std::chrono::high_resolution_clock::now();
		
		time_span = t2-t1;
		std::cout << std::setw(20) << end_time-ini_time <<'/'<<time_span.count()<< std::endl;
    
	}
	
	
}

void test_paraTL(const SparseMatrix<double>& RLap){
	
	
	Vector<double> b=RandVect(RLap.Cols(),300);
	
	std::cout<<"=======Two level solver======="<<std::endl;
	std::cout << std::setw(7) << "trial#" << std::setw(15) << "para" << std::setw(15) << "no para" << std::endl << std::endl;
    
	double ini_time;
	double end_time;
	
	for(int i=1;i<=4;i++){
		ini_time = omp_get_wtime();
		PCG_TL(RLap, b,100,1e-9,std::cbrt(RLap.Cols()),true);
		end_time = omp_get_wtime();
		std::cout << std::setw(7) << i << std::setw(15) << end_time-ini_time;
		
		ini_time = omp_get_wtime();
		//Mult<double,double,double>(LT,Laplacian);
		PCG_TL(RLap, b,100,1e-9,std::cbrt(RLap.Cols()),false);
		end_time = omp_get_wtime();
		std::cout << std::setw(15) << end_time-ini_time << std::endl;
    
	}
	
}

int main()
{
    //SparseMatrix<double> fine_adjacency = ReadMTXList("data/simple_graph_1.edges");
	
	SparseMatrix<double> Laplacian = getLaplacian("data/6473.edges",0,false);
    //Laplacian.PrintDense("LAP");
	//std::cout<<"read"<<std::endl;
	SparseMatrix<double> RLap = getReducedLaplacian(Laplacian);
	//RLap.PrintDense("reduced Laplacian");
	
	/**
	Vector<int> partitions = Partition(RLap,3);
    
    
	std::cout << "Partition Vector: ";
    for (int part: partitions.data()) std::cout << part << ' ';
    std::cout << std::endl;
    
    SparseMatrix<int> interpolation = GetUnweightedInterpolator(partitions);
    SparseMatrix<double> coarse_mat = interpolation.Transpose().Mult(RLap.Mult(interpolation));

    coarse_mat.PrintDense();
	*/
	
	/**
	DenseMatrix A = RandSPD(100,10);
	Vector<double> b=RandVect(100,300);
	
	
	std::cout << std::setw(7) << "trial#" << std::setw(15) << "para" << std::setw(15) << "no para" << std::endl << std::endl;
    double ini_time;
	double end_time;
	for(int i=1;i<=4;i++){
		ini_time = omp_get_wtime();
		ParaMult(A,b);
		end_time = omp_get_wtime();
		std::cout << std::setw(7) << i << std::setw(15) << end_time-ini_time;
		
		ini_time = omp_get_wtime();
		Mult(A,b);
		end_time = omp_get_wtime();
		std::cout << std::setw(15) << end_time-ini_time << std::endl;
    
	}
	*/
	
	
	/**
	Vector<double> b=RandVect(RLap.Cols(),300);
	//b.Print("b");
	//Timer Timer1, Timer2;
	std::cout<<"=======solving by regular CG======="<<std::endl;
	std::cout << std::setw(7) << "trial#" << std::setw(15) << "para" << std::setw(15) << "no para" << std::endl << std::endl;
    std::chrono::high_resolution_clock::time_point ini_time;
	std::chrono::high_resolution_clock::time_point end_time;
	std::chrono::duration<double> time_span;
	for(int i=1;i<=4;i++){
		ini_time = std::chrono::high_resolution_clock::now();
		CG(RLap,b,1000,1e-9,true);//.Print("sol:");
		end_time = std::chrono::high_resolution_clock::now();
		time_span = end_time-ini_time;
		std::cout << std::setw(7) << i << std::setw(15) << time_span.count();
		
		ini_time = std::chrono::high_resolution_clock::now();
		CG(RLap,b,1000,1e-9,false);//.Print("sol:");
		end_time = std::chrono::high_resolution_clock::now();
		time_span = end_time-ini_time;
		std::cout << std::setw(15) << time_span.count() << std::endl;
    
	}
	*/
	
	
	
	//test_paraMult();
	test_paraCG(RLap);
	
	/**
	std::cout<<"=======solving by jacobian PCG======="<<std::endl;
    PCG(RLap,b,Solve_Jacobian,1000,1e-9);
    //sol.Print("sol:");
	
	std::cout<<"=======solving by gauss-seidel PCG======="<<std::endl;
	PCG(RLap,b,Solve_Gauss_Seidel,1000,1e-9);//.Print("sol:");
	
	std::cout<<"=======start executing two-level======="<<std::endl;
	PCG_TL(RLap, b,100,1e-9,std::cbrt(RLap.Cols()));
	
	std::cout<<"=======start executing multi-level======="<<std::endl;
	PCG_ML(RLap,b,100,1e-9,7,std::cbrt(RLap.Cols()));
	*/
	
	//test_lubys();
}