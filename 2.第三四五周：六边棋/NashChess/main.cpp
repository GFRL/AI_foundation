#include<iostream>
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
#define MTCSXS 0.04
using namespace std;

char qp[3]={'X','?','Y'};

int GRID[11][11];//red -1 blue 1;
int xx[6] = { 0,1,-1,0,1,-1 }; int yy[6] = { 1,0,0,-1,-1,1 };
int v[11][11];
int vv[11][11];
int bs;
int now_camp;
clock_t t0;
inline int Min(int a, int b) { return a < b ? a : b; }
struct Node{
	int node,value;
	Node(int node, int value) :node(node), value(value) {};
};
struct cmp {
	bool operator()(Node A, Node B) { return A.value > B.value; }
};
struct Treenode {
	int x, y, camp,tree_father;//描述刚刚走完的棋的性质;
	vector<int>choice;//描述儿子节点编号
	vector<int>choice_position;//描述所有儿子所在的位置
	vector<int>Father; int expand_times;
	//vector<vector<int>>Q;
	vector<int>Value0[2],Value1[2];//0描述较近的,1描述较远的;1是红,0是蓝
	int GZ[3];
	int choice_num,choice_all_num;
	//map<int, int>paths;
	double UTB; double gz;
	double expand_all_values;
	//find father,return father;
	int GetFather(int node_NO) {
		if (Father[node_NO] == node_NO)return node_NO;
		else return Father[node_NO] = GetFather(Father[node_NO]);
	}
	//join two nodes,return father;
	int Join(int node1, int node2,int camp) {
		if (node1 > node2)swap(node1,node2);
		int s = GetFather(node1), t = GetFather(node2);
		Value0[camp][t] = Min(Value0[camp][t], Value0[camp][s]); Value1[camp][t] = Min(Value1[camp][s], Value1[camp][t]);
		return Father[s] = t;
	}
	//find the distance;3 types:up,down,both;
	int Getdistance(int node,int camp) {
		return Value0[camp][node] + Value1[camp][node];
	}
	int DFS(int X, int Y,int camp) {
		memset(vv, 0, sizeof(vv)); bool VALUE0 = true, VALUE1=true;
		priority_queue<Node, vector<Node>, cmp>QQ;
		QQ.push(Node(X*11+Y,0));
		while (!QQ.empty()) {
			Node A = QQ.top(); int X0 = A.node / 11; int Y0 = A.node - 11 * X0;
			if (vv[X0][Y0])continue;
			if (!VALUE1 && !VALUE0) {return VALUE1 + VALUE0;}
			if (Value0[camp][Father[X0 * 11 + Y0]] == 0) { 
				VALUE0 = false; Value0[11 * X + Y][camp] = Min(Value0[11 * X + Y][camp], A.value);
		    }
			if (Value1[camp][Father[X0 * 11 + Y0]] == 0) {
				VALUE1 = false; Value1[camp][11 * X + Y] = Min(Value1[camp][11 * X + Y], A.value);
			}
			for (int i = 0; i < 6; i++) {
				int newX = X0 + xx[i], newY = Y0 + yy[i];
				if (newX<0||newX>11||newY<0||newY>0||GRID[newX][newY] * GRID[X0][Y0] < 0 || !vv[newX][newY]) {
					QQ.push(Node(newX * 11 + newY, A.value + (Father[newX * 11 + newY] == Father[X0 * 11 + Y0] ? 1 : 0)));
				}
			}
		}
	}
	int Path_update() {
		int tempp = pos();
		if (camp == -1)Value0[1][tempp] = x, Value1[1][tempp] = 10 - x, Value0[0][tempp] = 100000, Value1[0][tempp] = 100000;
		else Value0[1][tempp] = 100000, Value1[1][tempp] = 100000, Value0[0][tempp] = y, Value1[0][tempp] = 10 - y;
		for (int i = 0; i < 6; i++) {
			int s = x + xx[i], t = y + yy[i];
			if (s >= 0 && s <= 10 && t >= 0 && t <= 10 && GRID[s][t] == GRID[x][y])Join(pos(), s * 11 + t,(1-camp)/2);
		}
		memset(v, 0, sizeof(v));
		for (int i = 0; i < 121; i++) {
			GetFather(i);
		}
		for (int i = 0; i < 121; i++) {
			int s = GetFather(i); int dx = s / 11, dy = s - 11 * dx;
			if (GRID[dx][dy]) { if (v[dx][dy])GZ[GRID[dx][dy]+1]=Min(GZ[GRID[dx][dy]+1],DFS(dx, dy,(1-GRID[dx][dy])/2)); }
		}
		if (GZ[camp + 1] == 0)gz= 10; else if (GZ[-camp + 1] == 0)gz = -10;
		else gz = 10.0/GZ[camp + 1] - 10.0/GZ[-camp + 1];
		//expand_times = 1; expand_all_values = gz;
		return gz;
	}
	double UTB_update() {
		return UTB = expand_all_values / expand_times + sqrt(MTCSXS * log(Tnode[0].expand_times / expand_times));
	}
	int pos() { return x * 11 + y; }
}Tnode[100001];
int Tnode_cnt = 0;
Treenode Root;
//清空Tnode;
void Reset() {
	fill(Tnode, Tnode + 100000, Treenode{});
	Tnode_cnt = 0;
}
void init() {
	Root.camp = -1; Root.choice_all_num = 121; for (int i = 0; i < 121; i++) {
		int x = i / 11, y = i - 11 * x;
		Root.Father.push_back(i); Root.choice_position.push_back(i); Root.Value0[0].push_back(x); Root.Value0[0].push_back(0);
	}
		Root.expand_times = 1; Root.expand_all_values = 0; Root.gz = 0; Root.tree_father = -1;
}
//新建Treenode;x,y,camp表示从father到此的选择;
int Treenode_bulit(int position,int camp,int father) {
	int x = position / 11, y = position - 11 * x; Tnode[father].choice.push_back(++Tnode_cnt); Tnode[father].choice_num++;
	Tnode[Tnode_cnt].x = x; Tnode[Tnode_cnt].y = y; Tnode[Tnode_cnt].camp = camp; Tnode[Tnode_cnt].tree_father = father;
	Tnode[Tnode_cnt].Father = Tnode[father].Father;
	Tnode[Tnode_cnt].GZ[0] = Tnode[Tnode_cnt].GZ[2] = 1e9;
	Tnode[Tnode_cnt].Value0 = Tnode[father].Value0; Tnode[Tnode_cnt].Value1 = Tnode[father].Value1;
	Tnode[Tnode_cnt].choice_num = 0; Tnode[Tnode_cnt].choice_all_num = Tnode[father].choice_all_num-1;
	GRID[x][y] = camp;
	for (int i = 0; i <= Tnode[Tnode_cnt].choice_all_num; i++)
		if (Tnode[father].choice_position[i] != position)Tnode[Tnode_cnt].choice_position.push_back(Tnode[father].choice_position[i]);
	Tnode[Tnode_cnt].Path_update();
	GRID[x][y] = 0;
	return Tnode_cnt;
}
int Treenode_expand(int father) {
	if(Tnode[father].choice_num!=Tnode[father].choice_all_num){
		return Treenode_bulit(Tnode[father].choice_position[Tnode[father].choice_num],-Tnode[father].camp,father);
	}
	else {
		if (!Tnode[father].choice_all_num)return father;
		vector<int>BESTSON; double Bestvalue=-100000;
		for (int i = 0; i < Tnode[father].choice_all_num; i++) {
			double tmp = Tnode[Tnode[father].choice[i]].UTB_update();
			if (tmp > Bestvalue) { BESTSON.clear(); BESTSON.push_back(i); Bestvalue = tmp; }
			else if (tmp == Bestvalue) { BESTSON.push_back(i); }
		}
		GRID[Tnode[father].x][Tnode[father].y] = Tnode[father].camp;
		int s=Treenode_expand(Tnode[father].choice[BESTSON[rand() % BESTSON.size()]]);
		GRID[Tnode[father].x][Tnode[father].y] = 0;
		return s;
	}
}
void value_update(int now, double value) {
	Tnode[now].expand_times++; Tnode[now].expand_all_values += value;
	while (Tnode[now].tree_father!=-1) {
		value *= -1; now = Tnode[now].tree_father;
		Tnode[now].expand_times++; Tnode[now].expand_all_values += value;
	}
}
int MTCS(int camp) {
	t0 = clock();
	Reset();
	Tnode[0] = Root; Tnode[0].choice.clear(); Tnode[0].choice_num = 0; Tnode[0].expand_times = 1; Root.expand_all_values = 0;
	if (Tnode[0].gz > 5 || Tnode[0].gz < -5)return 0;
	while (clock() - t0 < 9500) {
		int m = Treenode_expand(0); value_update(m, Tnode[m].gz);
	}
	vector<int>BESTSON; double Bestvalue = -100000;
	for (int i = 0; i < Tnode[0].choice_all_num; i++) {
		double tmp = Tnode[Tnode[0].choice[i]].UTB_update();
		if (tmp > Bestvalue) { BESTSON.clear(); BESTSON.push_back(i); Bestvalue = tmp; }
		else if (tmp == Bestvalue) { BESTSON.push_back(i); }
	}
	if (bs == 1)return 14;
	return BESTSON[rand() % BESTSON.size()];
}

void output() {
	for (int i = 0; i <= 10; i++) {
		for (int j = 1; j <= 10 - i; j++)cout << " ";
		for (int j = 0; j <= i; j++)cout << qp[GRID[i - j][j]+1] << ' '; 
		cout << endl;
	}
	for (int i = 1; i <= 10; i++) {
		for (int j = 1; j <= i; j++)cout << " ";
		for (int j = 0; j <= 10 - i; j++) {
			cout << qp[GRID[10 - j][i + j]+1] << ' ';
		}
		cout << endl;
	}
}

int main()
{
	int now_camp = -1;
	bs = 0;
	init();
	while (1) {
		if (bs % 2 == 0) {
			int s=MTCS(now_camp);
			if (s <= 0)cout << "OVER!";
			Root = Tnode[Treenode_bulit(Tnode[s].pos(),now_camp,0)];
			GRID[Tnode[s].x][Tnode[s].y] = now_camp;
			output();
		}
		else {
			int m; int n; cin >> m >> n; Tnode[0] = Root;
			Root = Tnode[Treenode_bulit(m*11+n, now_camp, 0)];
			GRID[m][n] = -now_camp;
			output();
		}
		bs++;
	}
}