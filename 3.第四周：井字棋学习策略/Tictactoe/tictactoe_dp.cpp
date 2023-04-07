#include "problem/tictactoe.hpp"
#include<cstring>
#include<cmath>
#include<queue>
using namespace std;
double discount = 0.9;
int Q_size = 0;
int Q_now = 0;
double POLICY[1 << 18];
bool vis[1 << 18];
class TicTacToePolicy {
public:
    int operator() (TicTacToeState state) {

        // 轮到X时，其选择第一个可行动作
        if (state.active_player() == 0) {

            return state.action_space()[0];

        }
        else {
            double best_value = -1e9; int best_index = 0;
            vector<int>tmp = state.action_space();
            int size = tmp.size();
            for (int i = 0; i < size; i++) {
                TicTacToeState TMP = state.next(tmp[i]);
                double M =POLICY[TMP.state_get()];
                if (M > best_value)best_value = M, best_index = i;
            }
            // 学习得到值函数表之后，把下面这句话替换成为根据值函数表贪心选择动作
            return state.action_space()[best_index];
        }

    }   
}policy;
double DFS(int STATE) {
    Q_now++; if (Q_now > Q_size)Q_size = Q_now;
    if (vis[STATE]) {
        Q_now--; return POLICY[STATE];
    }
    else vis[STATE] = true;
    TicTacToeState study_state;
    study_state.state_set(STATE);
    if (study_state.done()) {
        POLICY[STATE] = study_state.rewards()[1];
        Q_now--;
        return study_state.rewards()[1];
    }
    if (study_state.active_player() == 0) {
        study_state = study_state.next(policy(study_state));
        Q_now--;
        return POLICY[STATE] = DFS(study_state.state_get());
    }
    vector<int>tmp = study_state.action_space();
    int size = tmp.size(); double M = -1e9;
    for (int i = 0; i < size; i++) {
        TicTacToeState TMP = study_state.next(tmp[i]);
        double X = DFS(TMP.state_get());
        if (X > M)M = X;
    }
    POLICY[STATE] = M;
    Q_now--;
    return M;
}

int main() {
    TicTacToeState state;
    int MT; cin >> MT;
    // TODO: 通过与环境多次交互，学习打败X策略的方法
    int MM = CLOCKS_PER_SEC * 15;
    clock_t t0 = clock();
    int expand_times = 0;
    DFS(0);
    cout << "expand_times: " << expand_times << " using time: " << clock() - t0 << endl;
    cout << Q_size << endl;
    cout << expand_times << endl;
    // 测试O是否能够打败X的策略
    while (not state.done()) {
        auto action = policy(state);
        state = state.next(action);
        state.show();
    }
    return 0;
}