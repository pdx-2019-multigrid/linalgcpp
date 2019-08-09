#include <stdio.h>
#include <cstdlib>

using namespace linalgcpp;

SparseMatrix<double> getLaplacian(std::string fileName, int index, bool edgeRepeat){
	std::fstream file(fileName);
	int a,b,c;
	file>>a>>b>>c;
	
	CooMatrix<double> ADJ(a,b);
	for(int i=0;i<a;i++){
		ADJ.Add(i,i,0);//otherwise AddDiag() would not work
	}
	
	std::vector<double> degree(a);
	for(int i=0;i<c;i++){
		file >>a>>b;
		a-=index;
		b-=index;
		if(edgeRepeat){
			ADJ.Add(a,b,-1);
			degree[a]++;
		}else{
			ADJ.AddSym(a,b,-1);
			degree[a]++;
			degree[b]++;
		}
	}
	
	//std::cout<<degree<<std::endl;
	//L=D-A
	SparseMatrix<double> LAP = ADJ.ToSparse();
	LAP.AddDiag(degree);
	return LAP;
}

SparseMatrix<double> getWeightedAdjacency(std::string fileName, int index){
	std::fstream file(fileName);
	int a,b;
	file >>a>>b;
	CooMatrix<double> ADJ(a,a);
	
	int u,v;
	double w;
	for(int i=0;i<b;i++){
		file >>u>>v>>w;
		u-=index;
		v-=index;
		ADJ.AddSym(u,v,w);
	}
	
	return ADJ.ToSparse();
}

SparseMatrix<double> getReducedLaplacian(SparseMatrix<double> lap){
	int n = lap.Cols();
	std::vector<int> row(n-1);
	for(int i=0;i<n-1;i++){
		row[i]=i;
	}
	std::vector<int> col=row;
	return lap.GetSubMatrix(row,col);
}