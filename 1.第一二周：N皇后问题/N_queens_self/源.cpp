#include<iostream>
#include<cstdio>
#include<algorithm>
#include<string>
#include<cstring>
#include<string.h>
#include<ctime>
#include<chrono>
#include<vector>
#include<queue>
#include"random_variables.h"
int L;
class queenstate {
private:
	int N;
	std::vector<int>queenplace;
	std::vector<int>check_place1;
	std::vector<int>check_place2;
	long long check_place1_value, check_place2_value;
	unsigned long long total;
	double propability(int s,int t) {
		int ss = queenplace[s], tt = queenplace[t]; long long delta1 = 0, delta2 = 0;
		delta1 -= check_place1[ss + s]; check_place1[ss + s]--;
		delta1 -= check_place1[tt + t]; check_place1[tt + t]--;
		delta1 += check_place1[ss + t]; check_place1[ss + t]++;
		delta1 += check_place1[tt + s]; check_place1[tt + s]++;
		delta1 = (delta1 + 2) << 1;
		check_place1[ss + s]++; check_place1[tt + t]++; check_place1[ss + t]--; check_place1[tt + s]--;
		delta2 -= check_place2[ss - s + N]; check_place2[ss - s + N]--;
		delta2 -= check_place2[tt - t + N]; check_place2[tt - t + N]--;
		delta2 += check_place2[ss - t + N]; check_place2[ss - t + N]++;
		delta2 += check_place2[tt - s + N]; check_place2[tt - s + N]++;
		delta2 = (delta2 + 2) << 1;
		check_place2[ss - s + N]++; check_place2[tt - t + N]++; check_place2[ss - t + N]--; check_place2[tt - s + N]--;
		long long now = (delta1 * check_place2_value + delta2 * check_place1_value + delta1 * delta2);
		return std::max(-now*100000/ std::sqrt(std::min(N,int(total))),1.0/(total+now));
		//return -now * 1.0 / std::min(N, int(total));
		
	}
	unsigned long long change(int s,int t) {
		int ss = queenplace[s], tt = queenplace[t]; long long delta1 = 0, delta2 = 0;
		delta1 -= check_place1[ss + s]; check_place1[ss + s]--;
		delta1 -= check_place1[tt + t]; check_place1[tt + t]--;
		delta1 += check_place1[ss + t]; check_place1[ss + t]++;
		delta1 += check_place1[tt + s]; check_place1[tt + s]++;
		delta1 = (delta1 + 2) << 1;
		delta2 -= check_place2[ss - s + N]; check_place2[ss - s + N]--;
		delta2 -= check_place2[tt - t + N]; check_place2[tt - t + N]--;
		delta2 += check_place2[ss - t + N]; check_place2[ss - t + N]++;
		delta2 += check_place2[tt - s + N]; check_place2[tt - s + N]++;
		delta2 = (delta2 + 2) << 1;
		std::swap(queenplace[s], queenplace[t]);
		long long now = (delta1 * check_place2_value + delta2 * check_place1_value + delta1 * delta2);
		check_place1_value += delta1; check_place2_value += delta2;
		total += now;
		return total;
	}
public:
	void bulid(int size) {
		N = size; queenplace.clear(); check_place1.clear(); check_place2.clear();
		queenplace = RandomVariables::uniform_permutation(N);
		for (int i = 1; i <=2 * N ; i++)check_place1.push_back(0),check_place2.push_back(0);
		for (int i = 0; i < N; i++) {
			check_place1[queenplace[i] + i]++;
			check_place2[queenplace[i] - i + N]++;
		}
		for (int i = 0; i < 2 * N ; i++) {
			check_place1_value += check_place1[i] * check_place1[i];
			check_place2_value += check_place2[i] * check_place2[i];
		}
		check_place1_value -= N-1; check_place2_value -= N-1;
		total = check_place1_value * check_place2_value;
	}
	unsigned long long swap() {
		int s, t;
		while (1) {
			L++;
			s = RandomVariables::uniform_int() % N; t = RandomVariables::uniform_int() % N;
			if (RandomVariables::uniform_real() < propability(s, t))
				break;
		}
		return change(s, t);
	}


};
int main() {
	std::ios::sync_with_stdio(false);
	/*int s = 1;
	for (int i = 1; i <= 7; i++) {
		s *= 10;
		for (int j = 1; j <= 9; j++) {
			clock_t t0 = clock();
			queenstate A; A.bulid(j*s);
			std::cout << "NOW " << j * s << "  " << clock() - t0 << "ms\n";
		}
	}
	*/
	int S;
	while (std::cin >> S) {
		clock_t t0 = clock();
		int s = 0; int M = 0;
		queenstate AAA;
		AAA.bulid(S);
		while (1) {
			int m = AAA.swap(); s++;
			if (m == 1)break;
			if (clock() - t0 > 5000) {
				std::cout << "fail\n"; break;
			}
			if (s > 10000) {
				s = 0; AAA.bulid(S); M++;
			}
		}
		std::cout << "NOW " << "using "<<10000*M+s<<"times!"<<L<<"uasdhk!!" << clock() - t0 << "ms\n";
	}

}