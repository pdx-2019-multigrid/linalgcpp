#define watch(x) std::cout << (#x) << " is " << (x) << std::endl
#define log(x) std::cout << x << std::endl
#include "linalgcpp.hpp"
#include <omp.h>

using namespace linalgcpp;


void print(std::pair<int,int> pr){
	std::cout<<pr.first<<","<<pr.second<<std::endl;
}

SparseMatrix<int> getP(const SparseMatrix<double>& A){
	int n = A.Cols();
	std::vector<std::pair<int,int>> max_edge(n);
	for(int i=0;i<n;i++){
		//watch(i);
		//log("==============");
		std::vector<int> indices = A.GetIndices(i);
		std::vector<double> data = A.GetData(i);
		int max_index = 0;
		double max_weight = 0.0;
		
		for(int j=0;j<indices.size();++j){
			//watch(data[j]);
			//watch(indices[j]);
			if(indices[j]==i) continue;
			if(data[j]>max_weight){
				max_index=indices[j];
				max_weight=data[j];
			}
		}
		
		//watch(max_weight);
		//watch(max_index);
		max_edge[i]=std::make_pair(std::min(i,max_index),std::max(i,max_index));
	}
	/**
	for(int i=0;i<n;i++){
		print(max_edge[i]);
	}
	*/
	CooMatrix<int> P;
	std::vector<bool> registered(n);
	int col=0;
	for(int i=0;i<n;i++){
		if(registered[i]) continue;
		if(max_edge[i].second!=i&&max_edge[max_edge[i].second]==max_edge[i]){ 
			P.Add(max_edge[i].first,col,1);
			P.Add(max_edge[i].second,col++,1);
			registered[max_edge[i].second] =true;
		}else{
			P.Add(i,col++,1);
		}
	}
	
	return P.ToSparse();
	
}

SparseMatrix<int> getP(const SparseMatrix<double>& A, int Ncoarse){
	//assert Ncoarse<A.cols()
	
	int n = A.Cols();
	std::vector<std::pair<int,int>> max_edge(n);
	for(int i=0;i<n;i++){
		//watch(i);
		//log("==============");
		std::vector<int> indices = A.GetIndices(i);
		std::vector<double> data = A.GetData(i);
		int max_index = 0;
		double max_weight = 0.0;
		
		for(int j=0;j<indices.size();++j){
			//watch(data[j]);
			//watch(indices[j]);
			if(indices[j]==i) continue;
			if(data[j]>max_weight){
				max_index=indices[j];
				max_weight=data[j];
			}
		}
		
		//watch(max_weight);
		//watch(max_index);
		max_edge[i]=std::make_pair(std::min(i,max_index),std::max(i,max_index));
	}
	/**
	for(int i=0;i<n;i++){
		print(max_edge[i]);
	}
	*/
	CooMatrix<int> P;
	std::vector<bool> registered(n);
	int col=0;
	for(int i=0;i<n;i++){
		if(registered[i]) continue;
		if(max_edge[i].second!=i&&max_edge[max_edge[i].second]==max_edge[i]){ 
			P.Add(max_edge[i].first,col,1);
			P.Add(max_edge[i].second,col++,1);
			registered[max_edge[i].second] =true;
		}else{
			P.Add(i,col++,1);
		}
	}
	double x=omp_get_wtime();
	SparseMatrix<int> Ps = P.ToSparse();
	if(col<=Ncoarse){
		return Ps;
	}else{
		return Ps.Mult(getP(Ps.Transpose().Mult(A.Mult(Ps))));
	}
	
}