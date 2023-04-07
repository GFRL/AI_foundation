#include<algorithm>
#include<iostream>
#include<cstdio>
#include<ctime>
//#include<filesystem> 
#include<fstream>
//using namespace std::filesystem;
using namespace std;
static char stack[12]={'0','0','0','0','0','0','0','0','0','0','0','0'};
static int top = 0;int x=0;

inline void put(int mm) {
	std::streambuf *iutbuf = cout.rdbuf();
	top=0;
    while (mm) {
        stack[++top] = mm % 10 + '0';
        mm /= 10;
    }
    x=top;
    while (x)iutbuf -> sputc(stack[x--]);
    iutbuf -> sputc(' ');
}
int main(){
	ios::sync_with_stdio(0);
	cin.tie(0);cout.tie(0);
	std::streambuf *outbuf = cout.rdbuf();
    
	int n;cin>>n;
	clock_t t0=clock();
	if(n%6==2||n%6==3){
		int m=n/2;int i=m;
		if(n%2==0){	
		    if(m%2==0){	
				i = m + 2; put(m);
				for (; i <= n; i += 2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
				i = 4; put(2);
				for (; i < m; i += 2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
				i = m + 5; put(m + 3);
				for (; i <= n; i += 2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
			    for(i=1;i<m;i+=2)cout<<i<<' ';
			    cout<<m+1<<endl;
		    }
		    else{
				i = m + 2; put(m);
				for (; i < n; i += 2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
				i = 4; put(2);
				for (; i < m; i += 2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
				i = m + 5; put(m + 3);
				for (; i <= n; i += 2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
			    i = 3; put(1);
				for (; i < m; i += 2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
			    cout<<m+1<<' '<<n<<endl;
			}
		}
		else {
			if(m%2==0){
				i = m + 2; put(m);
				for(;i<n;i+=2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
				i = 3; put(1);
				for(;i<m;i+=2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
				i = m + 5; put(m + 3);
				for(;i<=n;i+=2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
				i = 4; put(2);
			    for(;i<m;i+=2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
			    cout<<m+1<<endl;
			} 
			else{
				i = m + 2; put(m);
				for (; i < n; i += 2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
				i = 3; put(1);
				for (; i < m; i += 2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
				i = m + 5; put(m + 3);
				for (; i < n; i += 2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
				i = 4; put(2);
				for (; i < m; i += 2) {
					stack[1] += 2;
					x++;
					while (stack[x] > '9')stack[x] -= 10, stack[++x]++;
					if (top < x)top++; x = top;
					while (x)outbuf->sputc(stack[x--]);
					outbuf->sputc(' ');
				}
				cout << m + 1 << endl;
			    cout<<m+1<<' '<<n<<endl;
			}
		}	
	}
	else{
		int i=4;put(2);
		for(;i<=n;i+=2){
		    stack[1]+=2;
			x++;
		    while (stack[x]>'9')stack[x]-=10,stack[++x]++;
            if(top<x)top++;x=top;
            while (x)outbuf -> sputc(stack[x--]);
            outbuf -> sputc(' ');
       }
       i=3;put(1);
	   for(;i<=n;i+=2){
			stack[1]+=2;
			x++;
		    while (stack[x]>'9')stack[x]-=10,stack[++x]++;
            if(top<x)top++;x=top;
            while (x)outbuf -> sputc(stack[x--]);
            outbuf -> sputc(' ');
       }
		cout<<endl;
	}
	ofstream ofs;
	ofs.open("try1.txt");
	ofs<<"time : "<<clock()-t0;
	//cout<<"time : "<<clock()-t0<<endl;
	return 0;
}
