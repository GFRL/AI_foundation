#include<iostream>
#include<random>
#include<cstdio>
#include<algorithm>
#include<string>
#include<cmath>
#include<cstring>
#include<map>
#include<vector>
#include<queue>
#include<ctime>
#include<string.h>
#include "jsoncpp/json.h"
#define MTCSXS 0.1
#define N 11
using namespace std;
int TIME_LIMITATION = 1000;

class RandomVariables {
private:

	// 固定随机种子
	RandomVariables() = default;

	// 非固定随机种子
	// RandomVariables() : random_engine(time(nullptr)) {}

	~RandomVariables() = default;

	std::default_random_engine random_engine;
	std::uniform_real_distribution<double> uniform_dist;
	std::uniform_int_distribution<int> uniform_int_dist;

	static RandomVariables rv;

public:

	// 均匀分布的正整数
	static int uniform_int() {
		return rv.uniform_int_dist(rv.random_engine);
	}

	// [0,1)均匀分布的实数
	static double uniform_real() {
		return rv.uniform_dist(rv.random_engine);
	}

	// 等概率分布的{0,1,2,n-1}排列
	static std::vector<int> uniform_permutation(int n) {
		std::vector<int> permutation(n);
		for (int i = 0; i < n; ++i) {
			permutation[i] = i;
		}

		for (int i = 0, j; i < n; ++i) {
			j = uniform_int() % (n - i) + i;
			std::swap(permutation[i], permutation[j]);
		}

		return permutation;
	}
};

RandomVariables RandomVariables::rv;

//global values statetion
char symbol[3] = { 'X','?','Y' };

int GRID[N + 2][N + 2];//-1红,1蓝;
int xx[6] = { 0,1,1,0,-1,-1 }, yy[6] = { -1,-1,0,1,1,0 };
int VIS[N + 2][N + 2];//分别用于两种DFS VIS=-1未访问,VIS=-100已经访问
int bs;
int now_camp;
clock_t t0;
inline int Min(int a, int b) { return a < b ? a : b; }
int Tnode0_expand_times;

double evaluation[14] = { 1e9,100000,20000,5000,1111,370,123,41,13,4,1,0.033,0.011,0.003 };
//返回x是否大于y;
bool compare(int x, int y) {
	if (x == -100) {
		return y != -100;
	}
	else if (x != -1) { return y == -1; }
	else return false;
}
struct Node {
	int x, y, value, lastfather_position;
	Node(int x, int y, int value, int lastfather_position = -100) :x(x), y(y), value(value), lastfather_position(lastfather_position) {};
};
struct cmp {
	bool operator()(Node A, Node B) { return A.value > B.value; }
};
inline int pos(int X, int Y) { return X * N + Y - N - 1; }
struct Treenode {
	int x, y, camp, tree_father;//描述刚刚走完的棋的性质,注意camp与该状态下要走棋的人camp相反;
	vector<int>choice;//描述儿子节点编号
	vector<int>choice_position;//描述所有儿子所在的位置
	vector<int>Father; int expand_times;
	//vector<vector<int>>Q;
	vector<int>Value0[2], Value1[2];//0描述较近的,1描述较远的;1是红,0是蓝
	double GZ[3];//0是红,2是蓝,GZ[2]*camp越大说明现在对要下棋人的人越好，超过1000000基本判定获胜;
	int choice_num, choice_all_num;
	//map<int, int>paths;
	double UTB;
	double expand_all_values;
	//bool empty_node_surround[6];//记述某点周围环境,用于双桥特判
	//find father,return father;
	int GetFather(int node_NO) {
		if (Father[node_NO] == node_NO)return node_NO;
		else return Father[node_NO] = GetFather(Father[node_NO]);
	}
	//join two nodes,return father;
	inline int Join(int node1, int node2, int camp = 0) {
		int s = GetFather(node1), t = GetFather(node2);
		return Father[s] = t;
	}
	//find the distance;3 types:up,down,both;
	inline int Getdistance(int node, int camp) { return Value0[camp][node] + Value1[camp][node]; }
	//several special check,node方向,m与m+1之间,请保证GRID[x][y]非空;
	inline bool judge(int x, int y, int m) {
		if (m == 5)
			return!GRID[x + xx[5]][y + yy[5]] && !GRID[x + xx[0]][y + yy[0]] && GRID[x][y] == GRID[x + xx[5] + xx[0]][y + yy[5] + yy[0]];
		else
			return !GRID[x + xx[m]][y + yy[m]] && !GRID[x + xx[m + 1]][y + yy[m + 1]] && GRID[x][y] == GRID[x + xx[m] + xx[m + 1]][y + yy[m] + yy[m + 1]];
	}
	//return GZ1;
	double Path_update() {
		//Father_update
		if (x > 0 && y > 0) {
			for (int i = 0; i < 6; i++) {
				int X0 = x + xx[i], Y0 = y + yy[i];
				if (GRID[X0][Y0] == camp)Join(pos(X0, Y0), pos(x, y));
			}
		}
		for (int i = 0; i < N * N; i++)GetFather(i);
		deque<Node>Q;
		Node tmp(0, 0, 0);

		//Value0[0];蓝方较近线
		{
			memset(VIS, -1, sizeof(VIS));
			for (int i = 1; i <= N; i++) { if (GRID[i][1] == 1)Q.push_front(Node(i, 1, 0)), Value0[0][pos(i, 1)] = 0; else if (!GRID[i][1])Q.push_back(Node(i, 1, 1)), Value0[0][pos(i, 1)] = 1; }
			while (!Q.empty()) {
				Node tmp = Q.front(); Q.pop_front();
				if (tmp.lastfather_position < 0)VIS[tmp.x][tmp.y] = -100;
				else if (VIS[tmp.x][tmp.y] == -100)continue;
				else {
					int MX = tmp.lastfather_position / N, MY = tmp.lastfather_position - N * MX; MX++, MY++;
					//假设是有色节点蓝色;
					if (GRID[MX][MY] == 1 && !GRID[tmp.x][tmp.y]) { VIS[tmp.x][tmp.y] = -100, Value0[0][pos(tmp.x, tmp.y)] = tmp.value; }
					else if (Father[pos(tmp.x, tmp.y)] == Father[tmp.lastfather_position]) {//判断是否为同阵营;若有效，这样的点需要进行同阵营扩展;
						if (compare(VIS[MX][MY], VIS[tmp.x][tmp.y]))VIS[tmp.x][tmp.y] = VIS[MX][MY], Value0[0][pos(tmp.x, tmp.y)] = tmp.value;
						else continue;
					}
					else {//存在一个空气的节点;
						if (VIS[tmp.x][tmp.y] >= 0) {//说明已经被访问了
							if (VIS[tmp.x][tmp.y] == tmp.lastfather_position) continue;
							else VIS[tmp.x][tmp.y] = -100, Value0[0][pos(tmp.x, tmp.y)] = tmp.value;
						}
						else VIS[tmp.x][tmp.y] = tmp.lastfather_position;
					}
				}
				int tmp_pos = pos(tmp.x, tmp.y);
				if (VIS[tmp.x][tmp.y] == -100) {//更新后已经可以全面更新
					/*
					if (GRID[tmp.x][tmp.y] == 1) {//双桥特判
						for (int i = 0; i < 6; i++)
							if (judge(tmp.x, tmp.y, i)) {
								int s = tmp.x + xx[i] + xx[(i + 1) % 6], t = tmp.y + yy[i] + yy[(i + 1) % 6];
								if (VIS[s][t] != -100)Q.push_front(Node(s, t, tmp.value, tmp_pos));
							}
					}
					*/
					for (int i = 0; i < 6; i++) {//全面拓展;
						int X0 = tmp.x + xx[i], Y0 = tmp.y + yy[i];
						if (GRID[X0][Y0] != -100 && VIS[X0][Y0] != -100) {
							if (!GRID[X0][Y0])Q.push_back(Node(X0, Y0, tmp.value + 1, tmp_pos));
							else if (GRID[X0][Y0] > 0 && compare(VIS[tmp.x][tmp.y], VIS[X0][Y0]))Q.push_front(Node(X0, Y0, tmp.value, tmp_pos));
						}
					}
				}
				else if (GRID[tmp.x][tmp.y] == 1) {//可以内部拓展
					for (int i = 0; i < 6; i++) {//小部分拓展;
						int X0 = tmp.x + xx[i], Y0 = tmp.y + yy[i];
						if (GRID[X0][Y0] == 1 && compare(VIS[tmp.x][tmp.y], VIS[X0][Y0]))Q.push_front(Node(X0, Y0, tmp.value, tmp_pos));
					}
				}
			}
		}
		//Value0[1];红方较近线
		{
			memset(VIS, -1, sizeof(VIS));
			for (int i = 1; i <= N; i++) { if (GRID[1][i] == -1)Q.push_front(Node(1, i, 0)), Value0[1][pos(1, i)] = 0; else if (!GRID[1][i])Q.push_back(Node(1, i, 1)), Value0[1][pos(1, i)] = 1; }
			while (!Q.empty()) {
				Node tmp = Q.front(); Q.pop_front();
				if (tmp.lastfather_position < 0)VIS[tmp.x][tmp.y] = -100;
				else if (VIS[tmp.x][tmp.y] == -100)continue;
				else {
					int MX = tmp.lastfather_position / N, MY = tmp.lastfather_position - N * MX; MX++, MY++;
					//假设是有色节点红色;
					if (GRID[MX][MY] == -1 && !GRID[tmp.x][tmp.y]) { VIS[tmp.x][tmp.y] = -100, Value0[1][pos(tmp.x, tmp.y)] = tmp.value; }
					else if (Father[pos(tmp.x, tmp.y)] == Father[tmp.lastfather_position]) {//判断是否为同阵营;若有效，这样的点需要进行同阵营扩展;
						if (compare(VIS[MX][MY], VIS[tmp.x][tmp.y]))VIS[tmp.x][tmp.y] = VIS[MX][MY], Value0[1][pos(tmp.x, tmp.y)] = tmp.value;
						else continue;
					}
					else {//存在一个空气的节点;
						if (VIS[tmp.x][tmp.y] >= 0) {//说明已经被访问了
							if (VIS[tmp.x][tmp.y] == tmp.lastfather_position) continue;
							else VIS[tmp.x][tmp.y] = -100, Value0[1][pos(tmp.x, tmp.y)] = tmp.value;
						}
						else VIS[tmp.x][tmp.y] = tmp.lastfather_position;
					}
				}
				int tmp_pos = pos(tmp.x, tmp.y);
				if (VIS[tmp.x][tmp.y] == -100) {//更新后已经可以全面更新
					/*
					if (GRID[tmp.x][tmp.y] == -1) {//双桥特判
						for (int i = 0; i < 6; i++)
							if (judge(tmp.x, tmp.y, i)) {
								int s = tmp.x + xx[i] + xx[(i + 1) % 6], t = tmp.y + yy[i] + yy[(i + 1) % 6];
								if (VIS[s][t] != -100)Q.push_front(Node(s, t, tmp.value, tmp_pos));
							}
					}
					*/
					for (int i = 0; i < 6; i++) {//全面拓展;
						int X0 = tmp.x + xx[i], Y0 = tmp.y + yy[i];
						if (GRID[X0][Y0] != -100 && VIS[X0][Y0] != -100) {
							if (!GRID[X0][Y0])Q.push_back(Node(X0, Y0, tmp.value + 1, tmp_pos));
							else if (GRID[X0][Y0] < 0 && compare(VIS[tmp.x][tmp.y], VIS[X0][Y0]))Q.push_front(Node(X0, Y0, tmp.value, tmp_pos));
						}
					}
				}
				else if (GRID[tmp.x][tmp.y] == -1) {//可以内部拓展
					for (int i = 0; i < 6; i++) {//小部分拓展;
						int X0 = tmp.x + xx[i], Y0 = tmp.y + yy[i];
						if (GRID[X0][Y0] == -1 && compare(VIS[tmp.x][tmp.y], VIS[X0][Y0]))Q.push_front(Node(X0, Y0, tmp.value, tmp_pos));
					}
				}
			}
		}
		//Value1[0];蓝方较远线
		{
			memset(VIS, -1, sizeof(VIS));
			for (int i = 1; i <= N; i++) { if (GRID[i][N] == 1)Q.push_front(Node(i, N, 0)), Value1[0][pos(i, N)] = 0; else if (!GRID[i][N])Q.push_back(Node(i, N, 1)), Value1[0][pos(i, N)] = 1; }
			while (!Q.empty()) {
				Node tmp = Q.front(); Q.pop_front();
				if (tmp.lastfather_position < 0)VIS[tmp.x][tmp.y] = -100;
				else if (VIS[tmp.x][tmp.y] == -100)continue;
				else {
					int MX = tmp.lastfather_position / N, MY = tmp.lastfather_position - N * MX; MX++, MY++;
					//假设是有色节点蓝色;
					if (GRID[MX][MY] == 1 && !GRID[tmp.x][tmp.y]) { VIS[tmp.x][tmp.y] = -100, Value1[0][pos(tmp.x, tmp.y)] = tmp.value; }
					else if (Father[pos(tmp.x, tmp.y)] == Father[tmp.lastfather_position]) {//判断是否为同阵营;若有效，这样的点需要进行同阵营扩展;
						if (compare(VIS[MX][MY], VIS[tmp.x][tmp.y]))VIS[tmp.x][tmp.y] = VIS[MX][MY], Value1[0][pos(tmp.x, tmp.y)] = tmp.value;
						else continue;
					}
					else {//存在一个空气的节点;
						if (VIS[tmp.x][tmp.y] >= 0) {//说明已经被访问了
							if (VIS[tmp.x][tmp.y] == tmp.lastfather_position) continue;
							else VIS[tmp.x][tmp.y] = -100, Value1[0][pos(tmp.x, tmp.y)] = tmp.value;
						}
						else VIS[tmp.x][tmp.y] = tmp.lastfather_position;
					}
				}
				int tmp_pos = pos(tmp.x, tmp.y);
				if (VIS[tmp.x][tmp.y] == -100) {//更新后已经可以全面更新
					/*
					if (GRID[tmp.x][tmp.y] == 1) {//双桥特判
						for (int i = 0; i < 6; i++)
							if (judge(tmp.x, tmp.y, i)) {
								int s = tmp.x + xx[i] + xx[(i + 1) % 6], t = tmp.y + yy[i] + yy[(i + 1) % 6];
								if (VIS[s][t] != -100)Q.push_front(Node(s, t, tmp.value, tmp_pos));
							}
					}
					*/
					for (int i = 0; i < 6; i++) {//全面拓展;
						int X0 = tmp.x + xx[i], Y0 = tmp.y + yy[i];
						if (GRID[X0][Y0] != -100 && VIS[X0][Y0] != -100) {
							if (!GRID[X0][Y0])Q.push_back(Node(X0, Y0, tmp.value + 1, tmp_pos));
							else if (GRID[X0][Y0] > 0 && compare(VIS[tmp.x][tmp.y], VIS[X0][Y0]))Q.push_front(Node(X0, Y0, tmp.value, tmp_pos));
						}
					}
				}
				else if (GRID[tmp.x][tmp.y] == 1) {//可以内部拓展
					for (int i = 0; i < 6; i++) {//小部分拓展;
						int X0 = tmp.x + xx[i], Y0 = tmp.y + yy[i];
						if (GRID[X0][Y0] == 1 && compare(VIS[tmp.x][tmp.y], VIS[X0][Y0]))Q.push_front(Node(X0, Y0, tmp.value, tmp_pos));
					}
				}
			}
		}
		//Value1[1];红方较远线
		{
			memset(VIS, -1, sizeof(VIS));
			for (int i = 1; i <= N; i++) { if (GRID[N][i] == -1)Q.push_front(Node(N, i, 0)), Value1[1][pos(N, i)] = 0; else if (!GRID[N][i])Q.push_back(Node(N, i, 1)), Value1[1][pos(N, i)] = 1; }
			while (!Q.empty()) {
				Node tmp = Q.front(); Q.pop_front();
				if (tmp.lastfather_position < 0)VIS[tmp.x][tmp.y] = -100;
				else if (VIS[tmp.x][tmp.y] == -100)continue;
				else {
					int MX = tmp.lastfather_position / N, MY = tmp.lastfather_position - N * MX; MX++, MY++;
					//假设是有色节点红色;
					if (GRID[MX][MY] == -1 && !GRID[tmp.x][tmp.y]) { VIS[tmp.x][tmp.y] = -100, Value1[1][pos(tmp.x, tmp.y)] = tmp.value; }
					else if (Father[pos(tmp.x, tmp.y)] == Father[tmp.lastfather_position]) {//判断是否为同阵营;若有效，这样的点需要进行同阵营扩展;
						if (compare(VIS[MX][MY], VIS[tmp.x][tmp.y]))VIS[tmp.x][tmp.y] = VIS[MX][MY], Value1[1][pos(tmp.x, tmp.y)] = tmp.value;
						else continue;
					}
					else {//存在一个空气的节点;
						if (VIS[tmp.x][tmp.y] >= 0) {//说明已经被访问了
							if (VIS[tmp.x][tmp.y] == tmp.lastfather_position) continue;
							else VIS[tmp.x][tmp.y] = -100, Value1[1][pos(tmp.x, tmp.y)] = tmp.value;
						}
						else VIS[tmp.x][tmp.y] = tmp.lastfather_position;
					}
				}
				int tmp_pos = pos(tmp.x, tmp.y);
				if (VIS[tmp.x][tmp.y] == -100) {//更新后已经可以全面更新
					/*
					if (GRID[tmp.x][tmp.y] == -1) {//双桥特判
						for (int i = 0; i < 6; i++)
							if (judge(tmp.x, tmp.y, i)) {
								int s = tmp.x + xx[i] + xx[(i + 1) % 6], t = tmp.y + yy[i] + yy[(i + 1) % 6];
								if (VIS[s][t] != -100)Q.push_front(Node(s, t, tmp.value, tmp_pos));
							}
					}
					*/
					for (int i = 0; i < 6; i++) {//全面拓展;
						int X0 = tmp.x + xx[i], Y0 = tmp.y + yy[i];
						if (GRID[X0][Y0] != -100 && VIS[X0][Y0] != -100) {
							if (!GRID[X0][Y0])Q.push_back(Node(X0, Y0, tmp.value + 1, tmp_pos));
							else if (GRID[X0][Y0] < 0 && compare(VIS[tmp.x][tmp.y], VIS[X0][Y0]))Q.push_front(Node(X0, Y0, tmp.value, tmp_pos));
						}
					}
				}
				else if (GRID[tmp.x][tmp.y] == -1) {//可以内部拓展
					for (int i = 0; i < 6; i++) {//小部分拓展;
						int X0 = tmp.x + xx[i], Y0 = tmp.y + yy[i];
						if (GRID[X0][Y0] == -1 && compare(VIS[tmp.x][tmp.y], VIS[X0][Y0]))Q.push_front(Node(X0, Y0, tmp.value, tmp_pos));
					}
				}
			}
		}

		//GZ运算
		memset(VIS, 0, sizeof(VIS));
		for (int i = 0; i < N * N; i++) {
			int X0 = (Father[i] / N), Y0 = (Father[i] - N * X0);
			X0++, Y0++;
			if (!VIS[X0][Y0] && (GRID[X0][Y0] == 1 || GRID[X0][Y0] == -1)) {
				int s = Getdistance(Father[i], (1 - GRID[X0][Y0]) / 2);
				VIS[X0][Y0] = true;
				if (s < 14)
					GZ[GRID[X0][Y0] + 1] += evaluation[s];
			}
		}
		GZ[1] = (GZ[0] - GZ[2]) * camp; return GZ[1];
	}
	double UTB_update() {
		return UTB = expand_all_values / expand_times + MTCSXS * bs * pow(N * N * expand_times / Tnode0_expand_times, 2);
	}
}Tnode[60001];
int Tnode_cnt = 0;
vector<int>Value_standrad;
Treenode Root;
int game_mode;//mode 0 正常 mode 1 下一步必胜 mode 2 已经获胜
int best_choice = 0;



void Reset() {
	fill(Tnode, Tnode + 60000, Treenode{});
	Tnode_cnt = 0;
}
//进入时自动改变GRID;退出时会复原GRID special传入指定儿子编号;
int Treenode_bulit(int position, int camp, int father, int special = -1) {
	if (special >= 0)Tnode[father].choice_num = special;
	int x = (position / N), y = (position - N * x);
	x++; y++;
	Tnode[father].choice.push_back(++Tnode_cnt); Tnode[father].choice_num++;
	Tnode[Tnode_cnt].x = x; Tnode[Tnode_cnt].y = y; Tnode[Tnode_cnt].camp = camp; Tnode[Tnode_cnt].tree_father = father;
	Tnode[Tnode_cnt].Father = Tnode[father].Father;
	Tnode[Tnode_cnt].GZ[0] = Tnode[Tnode_cnt].GZ[2] = 0;
	Tnode[Tnode_cnt].Value0[0] = Value_standrad; Tnode[Tnode_cnt].Value0[1] = Value_standrad; Tnode[Tnode_cnt].Value1[0] = Value_standrad; Tnode[Tnode_cnt].Value1[1] = Value_standrad;
	GRID[x][y] = camp;
	double tmppp = Tnode[Tnode_cnt].Path_update();
	//此处暂时去掉必败特判
	Tnode[Tnode_cnt].choice_num = 0; Tnode[Tnode_cnt].choice_all_num = Tnode[father].choice_all_num - 1;
	for (int i = 0; i < Tnode[father].choice_num - 1; i++) {
		Tnode[Tnode_cnt].choice_position.push_back(Tnode[father].choice_position[i]);
	}
	for (int i = Tnode[father].choice_num; i < Tnode[father].choice_all_num; i++)Tnode[Tnode_cnt].choice_position.push_back(Tnode[father].choice_position[i]);
	//for (int i = 0; i <= Tnode[Tnode_cnt].choice_all_num; i++)
		//if (Tnode[father].choice_position[i] != position)Tnode[Tnode_cnt].choice_position.push_back(Tnode[father].choice_position[i]);

	GRID[x][y] = 0;
	return Tnode_cnt;
}
int Treenode_expand(int father) {
	if (Tnode[father].choice_num != Tnode[father].choice_all_num) {
		if (father != 0)
			GRID[Tnode[father].x][Tnode[father].y] = Tnode[father].camp;
		return Treenode_bulit(Tnode[father].choice_position[Tnode[father].choice_num], -Tnode[father].camp, father);
	}
	else {
		if (!Tnode[father].choice_all_num)return father;
		vector<int>BESTSON; double Bestvalue = -1e11;
		for (int i = 0; i < Tnode[father].choice_all_num; i++) {
			double tmp = -Tnode[Tnode[father].choice[i]].UTB_update();
			if (tmp > Bestvalue) { BESTSON.clear(); BESTSON.push_back(i); Bestvalue = tmp; }
			else if (tmp == Bestvalue) { BESTSON.push_back(i); }
		}
		GRID[Tnode[father].x][Tnode[father].y] = Tnode[father].camp;
		int s = Treenode_expand(Tnode[father].choice[BESTSON[RandomVariables::uniform_int() % BESTSON.size()]]);
		return s;
	}
}
void value_update(int now, double value) {
	Tnode[now].expand_times++; Tnode[now].expand_all_values += value;
	GRID[Tnode[now].x][Tnode[now].y] = 0;
	while (Tnode[now].tree_father != 0) {
		value *= -1; now = Tnode[now].tree_father; if (value < -10000000)value = -500000;
		Tnode[now].expand_times++; Tnode[now].expand_all_values += value;
		GRID[Tnode[now].x][Tnode[now].y] = 0;
	}
}


int special_father[N + 2][N + 2];
int special_mtcs(int Mode) {
	if (Mode == 1) {//即将获胜环节
		while (clock() - t0 < TIME_LIMITATION) {
			int m = Treenode_expand(best_choice); value_update(m, Tnode[m].GZ[1]); Tnode[0].expand_times++; Tnode0_expand_times = Tnode[0].expand_times;
		}
		return best_choice;
	}
	else {//搜索胜利之路
		for (int i = 0; i <= N + 1; i++)
			for (int j = 0; j <= N + 1; j++)
				special_father[i][j] = -10000;
		priority_queue<Node, vector<Node>, cmp>Q;
		int best_x = -1, best_y = -1;
		memset(VIS, 0, sizeof(VIS));
		int NOW_your_camp = -Tnode[0].camp;
		if (NOW_your_camp == -1) {//当上一个人走的是蓝，意味着你现在将走红;
			//从底边开始搜索;
			for (int i = 1; i <= N; i++) {
				if (Tnode[0].Getdistance(pos(1, i), 1) == 0 && GRID[1][i] == -1) {
					Q.push(Node(1, i, 0, -1));
				}
				special_father[1][i] = -1;
			}
			while (!Q.empty()) {
				Node tmp = Q.top(); Q.pop();
				if (VIS[tmp.x][tmp.y])continue;
				VIS[tmp.x][tmp.y] = true; special_father[tmp.x][tmp.y] = tmp.lastfather_position;
				if (tmp.x == N) {
					int tmpx = tmp.x, tmpy = tmp.y; int tmppx = -1, tmppy = -1;
					while (special_father[tmpx][tmpy] != -1) {
						tmppx = special_father[tmpx][tmpy] / N; tmppy = special_father[tmpx][tmpy] - N * tmppx;
						tmppx++; tmppy++;
						if (Tnode[0].Father[pos(tmpx, tmpy)] == Tnode[0].Father[pos(tmppx, tmppy)])
							tmpx = tmppx, tmpy = tmppy;
						else break;
					}
					if (special_father[tmpx][tmpy] != -1) {
						for (int i = 0; i < 6; i++) {
							if (GRID[tmpx + xx[i]][tmpy + yy[i]] == 0 && tmpx + xx[i] + xx[(i + 1) % 6] == tmppx && tmpy + yy[i] + yy[(i + 1) % 6] == tmppy) {
								best_x = tmpx + xx[i], best_y = tmpy + yy[i]; break;
							}
						}
					}
					break;
				}
				else {
					for (int i = 0; i < 6; i++) {
						if (GRID[tmp.x][tmp.y] == GRID[tmp.x + xx[i]][tmp.y + yy[i]] && !VIS[tmp.x + xx[i]][tmp.y + yy[i]])Q.push(Node(tmp.x + xx[i], tmp.y + yy[i], tmp.value, pos(tmp.x, tmp.y)));
					}

					for (int i = 0; i < 6; i++) {
						if (Tnode[0].judge(tmp.x, tmp.y, i) && !VIS[tmp.x + xx[i] + xx[(i + 1) % 6]][tmp.y + yy[i] + yy[(i + 1) % 6]]) {
							Q.push(Node(tmp.x + xx[i] + xx[(i + 1) % 6], tmp.y + yy[i] + yy[(i + 1) % 6], tmp.value + 1, pos(tmp.x, tmp.y)));
						}
					}

				}
			}
			if (best_x == -1)return 1;//返回第一个可走的棋子;
			else {
				int POS2 = pos(best_x, best_y);
				int l = 0, r = Tnode[0].choice_all_num - 1;
				while (l != r) {
					int mid = (l + r) >> 1;
					if (Tnode[0].choice_position[mid] > POS2)r = mid - 1;
					else if (Tnode[0].choice_position[mid] < POS2)l = mid + 1;
					else l = r = mid;
				}
				return l;
			}
		}
		else if (NOW_your_camp == 1) {//当上一个人走的是红，意味着你现在将走蓝;
			//从底边开始搜索;
			for (int i = 1; i <= N; i++) {
				if (Tnode[0].Getdistance(pos(i, 1), 0) == 0 && GRID[i][1] == 1) {
					Q.push(Node(i, 1, 0, -1));
				}
				special_father[i][1] = -1;
			}
			while (!Q.empty()) {
				Node tmp = Q.top(); Q.pop();
				if (VIS[tmp.x][tmp.y])continue;
				VIS[tmp.x][tmp.y] = true; special_father[tmp.x][tmp.y] = tmp.lastfather_position;
				if (tmp.x == N) {
					int tmpx = tmp.x, tmpy = tmp.y; int tmppx = -1, tmppy = -1;
					while (special_father[tmpx][tmpy] != -1) {
						tmppx = special_father[tmpx][tmpy] / N; tmppy = special_father[tmpx][tmpy] - N * tmppx;
						tmppx++; tmppy++;
						if (Tnode[0].Father[pos(tmpx, tmpy)] == Tnode[0].Father[pos(tmppx, tmppy)])
							tmpx = tmppx, tmpy = tmppy;
						else break;
					}
					if (special_father[tmpx][tmpy] != -1) {
						for (int i = 0; i < 6; i++) {
							if (GRID[tmpx + xx[i]][tmpy + yy[i]] == 0 && tmpx + xx[i] + xx[(i + 1) % 6] == tmppx && tmpy + yy[i] + yy[(i + 1) % 6] == tmppy) {
								best_x = tmpx + xx[i], best_y = tmpy + yy[i]; break;
							}
						}
					}
					break;
				}
				else {
					for (int i = 0; i < 6; i++) {
						if (GRID[tmp.x][tmp.y] == GRID[tmp.x + xx[i]][tmp.y + yy[i]] && !VIS[tmp.x + xx[i]][tmp.y + yy[i]])Q.push(Node(tmp.x + xx[i], tmp.y + yy[i], tmp.value, pos(tmp.x, tmp.y)));
					}
					/*
					for (int i = 0; i < 6; i++) {
						if (Tnode[0].judge(tmp.x, tmp.y, i) && !vis[tmp.x + xx[i] + xx[(i + 1) % 6]][tmp.y + yy[i] + yy[(i + 1) % 6]]) {
							Q.push(Node(tmp.x + xx[i] + xx[(i + 1) % 6], tmp.y + yy[i] + yy[(i + 1) % 6], tmp.value + 1, pos(tmp.x, tmp.y)));
						}
					}
					*/
				}
			}
			if (best_x == -1)return 1;//返回第一个可走的棋子;
			else {
				int POS2 = pos(best_x, best_y);
				int l = 0, r = Tnode[0].choice_all_num - 1;
				while (l != r) {
					int mid = (l + r) >> 1;
					if (Tnode[0].choice_position[mid] > POS2)r = mid - 1;
					else if (Tnode[0].choice_position[mid] < POS2)l = mid + 1;
					else l = r = mid;
				}
				return l;
			}
		}
	}
}
//对被破坏单桥的搜索;请保证已经完成建树
int special_search(int node) {
	GRID[Tnode[node].x][Tnode[node].y] = Tnode[node].camp;//保证已经完成;
	int MINVALUE = 10000; int best_pos = -1;
	for (int tmpx = 1; tmpx <= N; tmpx++)
		for (int tmpy = 1; tmpy <= N; tmpy++)
			if (GRID[tmpx][tmpy] == -Tnode[node].camp) {
				for (int i = 0; i < 3; i++) {
					int sx = tmpx + xx[i], tx = tmpx + xx[i + 1], sy = tmpy + yy[i], ty = tmpy + yy[i + 1], stx = tx + xx[i], sty = ty + yy[i];
					if (GRID[sx][sy] == 0 && GRID[tx][ty] == Tnode[node].camp && GRID[stx][sty] == GRID[tmpx][tmpy] && Tnode[node].Father[pos(tmpx, tmpy)] != Tnode[node].Father[pos(stx, sty)]) {
						int W = min(Tnode[node].Getdistance(pos(tmpx, tmpy), (1 + Tnode[node].camp) / 2), Tnode[node].Getdistance(pos(stx, sty), (1 + Tnode[node].camp) / 2));
						if (W < MINVALUE)best_pos = pos(sx, sy), MINVALUE = W;
					}
					else if (GRID[tx][ty] == 0 && GRID[sx][sy] == Tnode[node].camp && GRID[stx][sty] == GRID[tmpx][tmpy] && Tnode[node].Father[pos(tmpx, tmpy)] != Tnode[node].Father[pos(stx, sty)]) {
						int W = min(Tnode[node].Getdistance(pos(tmpx, tmpy), (1 + Tnode[node].camp) / 2), Tnode[node].Getdistance(pos(stx, sty), (1 + Tnode[node].camp) / 2));
						if (W < MINVALUE)best_pos = pos(tx, ty), MINVALUE = W;
					}
				}
			}
	if (best_pos == -1)return -1;
	else {
		int POS2 = best_pos;
		int l = 0, r = Tnode[node].choice_all_num - 1;
		while (l != r) {
			int mid = (l + r) >> 1;
			if (Tnode[node].choice_position[mid] > POS2)r = mid - 1;
			else if (Tnode[node].choice_position[mid] < POS2)l = mid + 1;
			else l = r = mid;
		}
		return l + 1;
	}
}
int MTCS(int camp) {
	t0 = clock();
	game_mode = 0;
	Reset();
	Tnode[0] = Root; Tnode[0].choice.clear(); Tnode[0].choice_num = 0; Tnode[0].expand_times = 1; Tnode[0].expand_all_values = 0;
	Tnode[0].tree_father = 0; Tnode0_expand_times = 1;
	if (Tnode[0].GZ[Tnode[0].camp + 1] > 8e8) {//若为-1;则应当0最大;
		game_mode = 2;
	}
	else {
		//
		//int TMP = special_search(0);
		//if (TMP != -1)game_mode = 1, best_choice = TMP;
	}
	//高速刷一遍根节点
	for (int i = 0; i < Tnode[0].choice_all_num; i++) {
		int m = Treenode_expand(0); value_update(m, Tnode[m].GZ[1]); Tnode[0].expand_times++; Tnode0_expand_times = Tnode[0].expand_times;
		if (game_mode != 2 && Tnode[m].GZ[Tnode[0].camp + 1] > 8e8) {
			game_mode = 1; best_choice = m;
		}
	}
	if (game_mode)return special_mtcs(game_mode);
	while (clock() - t0 < TIME_LIMITATION) {
		int m = Treenode_expand(0); value_update(m, Tnode[m].GZ[1]); Tnode[0].expand_times++; Tnode0_expand_times = Tnode[0].expand_times;
	}
	vector<int>BESTSON; double Bestvalue = -1e11;
	for (int i = 0; i < Tnode[0].choice_all_num; i++) {
		double tmp = -Tnode[Tnode[0].choice[i]].expand_all_values / Tnode[Tnode[0].choice[i]].expand_times;
		if (tmp > Bestvalue) { BESTSON.clear(); BESTSON.push_back(i); Bestvalue = tmp; }
		else if (tmp == Bestvalue) { BESTSON.push_back(i); }
	}
	if (bs == 1)return N * 1 + 3;
	//cout << Tnode[0].expand_times << endl;
	return Tnode[0].choice[BESTSON[RandomVariables::uniform_int() % BESTSON.size()]];
}
;

string str;
Json::Reader reader;
Json::Value input;


void init() {
	TIME_LIMITATION = 99 * CLOCKS_PER_SEC / 100;
	int turn_id = input["responses"].size();
	if (input["requests"][0].isMember("forced_x")) {//特判
		bs = 1; now_camp = -1;
	}
	else {
		bs = 2; GRID[input["requests"][0]["x"].asInt() + 1][input["requests"][0]["y"].asInt() + 1] = -1;
		now_camp = 1;
	}
	for (int i = 0; i < turn_id; i++) {
		bs += 2;
		GRID[input["responses"][i]["x"].asInt() + 1][input["responses"][i]["y"].asInt() + 1] = now_camp;
		GRID[input["requests"][i + 1]["x"].asInt() + 1][input["requests"][i + 1]["y"].asInt() + 1] = -now_camp;
	}
	//棋盘预处理
	for (int i = 0; i <= N + 1; i++)GRID[i][0] = GRID[0][i] = GRID[N + 1][i] = GRID[i][N + 1] = -100;
	for (int i = 0; i < N * N; i++)Value_standrad.push_back(1e9), Root.Father.push_back(i);
	Root.camp = -now_camp;
	Root.x = input["requests"][turn_id]["x"].asInt() + 1; Root.y = input["requests"][turn_id]["y"].asInt() + 1;
	Root.choice_all_num = Root.choice_num = 0; Root.choice_position = {}; Root.choice = {};

	for (int i = 1; i <= N; i++)
		for (int j = 1; j <= N; j++) {
			if (!GRID[i][j])Root.choice_position.push_back(pos(i, j)), Root.choice_all_num++;
			else for (int s = 0; s < 3; s++) {
				if (GRID[i + xx[s]][j + yy[s]] == GRID[i][j])Root.Join(pos(i, j), pos(i + xx[s], j + yy[s]));
			}
		}
	Root.Value0[0] = Value_standrad; Root.Value0[1] = Value_standrad; Root.Value1[0] = Value_standrad; Root.Value1[1] = Value_standrad;
	Root.Path_update();
	Root.tree_father = -1;
	Root.expand_times = 1; Root.expand_all_values = Root.GZ[1]; Root.tree_father = -1;
}
void Renew(int s, int X2, int Y2) {//请传递内部实现的X，Y，也就是加了1的；
	int POS2 = pos(X2, Y2);
	int l = 0, r = Tnode[s].choice_all_num - 1;
	while (l != r) {
		int mid = (l + r) >> 1;
		if (Tnode[s].choice_position[mid] > POS2)r = mid - 1;
		else if (Tnode[s].choice_position[mid] < POS2)l = mid + 1;
		else l = r = mid;
	}
	if (l >= Tnode[s].choice_num)Root = Tnode[Treenode_bulit(POS2, Tnode[s].camp * -1, s, l)];
	else Root = Tnode[Tnode[s].choice[l]];
	GRID[X2][Y2] = Root.camp;
}

void output() {
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N - i + 1; j++)cout << " ";
		for (int j = 1; j <= i; j++)cout << symbol[GRID[i - j + 1][j] + 1] << ' ';
		cout << endl;
	}
	for (int i = 1; i < N; i++) {
		for (int j = 1; j <= i + 1; j++)cout << " ";
		for (int j = 1; j <= N - i; j++) {
			cout << symbol[GRID[N - j + 1][i + j] + 1] << ' ';
		}
		cout << endl;
	}
}

void JSON_output(int u, int v) {//请传入内部X，Y;
	Json::Value result;
	Json::Value action;
	action["x"] = u - 1; action["y"] = v - 1;
	action["debug"] = "Expand times: " + to_string(Tnode0_expand_times) + " expand Treenodes: " + to_string(Tnode_cnt) + " using times :" + to_string(clock() - t0);
	result["response"] = action;

	Json::FastWriter writer;

	cout << writer.write(result) << endl;
	cout << "\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<" << endl;
}
int main() {
	bs = 0;
	string str;
	getline(std::cin, str);
	reader.parse(str, input);
	int X1, X2, Y1, Y2; int s;

	t0 = clock();
	init();
	s = MTCS(now_camp);
	X1 = Tnode[s].x, Y1 = Tnode[s].y;
	GRID[X1][Y1] = now_camp;
	//output();
	JSON_output(X1, Y1);
	while (1) {
		bs += 2;
		getline(std::cin, str);
		reader.parse(str, input);
		X2 = input["x"].asInt() + 1;
		Y2 = input["y"].asInt() + 1;
		GRID[X2][Y2] = -now_camp;
		Renew(s, X2, Y2);
		s = MTCS(now_camp);
		if (s == 0)break;
		//cin >> s; char t = getchar();
		X1 = Tnode[s].x, Y1 = Tnode[s].y;
		GRID[X1][Y1] = now_camp;

		JSON_output(X1, Y1);
		//output();
	}
	return 0;
}