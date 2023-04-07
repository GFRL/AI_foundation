/*
#include "problem/tictactoe.hpp"
#include<cstring>
#include<cmath>
#include<queue>
using namespace std;
double discount=0.9;
int Q_size = 0;
class TicTacToePolicy {
public:
    double POLICY[1 << 18];
    double POLICY_NEW[1 << 18];
    int operator() (TicTacToeState state) {

        // 轮到X时，其选择第一个可行动作
        if (state.active_player() == 0) {

            return state.action_space()[0];

        }
        else {
            double best_value=-1e9; int best_index=0;
            vector<int>tmp = state.action_space();
            int size = tmp.size();
            for (int i = 0; i < size; i++) {
                TicTacToeState TMP = state.next(tmp[i]);
                double M = TMP.rewards()[1] + discount * POLICY[TMP.state_get()];
                if (M > best_value)best_value = M, best_index = i;
            }
            // 学习得到值函数表之后，把下面这句话替换成为根据值函数表贪心选择动作
            return state.action_space()[best_index];
        }

    }
    double study() {
        double Max_delta = 0;
        TicTacToeState study_state;
        int s = 1 << 18;
        
        queue<int>STATE; STATE.push(0); Q_size++;
        while (!STATE.empty()) {
            int I = STATE.front(); STATE.pop();
            study_state.state_set(I);
            if (study_state.done())continue;
            else if (study_state.active_player() == 0) {
               TicTacToeState TMP= study_state.next(this->operator()(study_state));
               POLICY_NEW[study_state.state_get()] = TMP.rewards()[1] + discount * POLICY[TMP.state_get()];
               if(!TMP.done())STATE.push(TMP.state_get()), Q_size++;
               continue;
            }
            double best_value = -1e9; int best_index = 0;
            vector<int>tmp = study_state.action_space();
            int size = tmp.size();
            for (int i = 0; i < size; i++) {
                TicTacToeState TMP = study_state.next(tmp[i]);
                double M = TMP.rewards()[1] + discount * POLICY[TMP.state_get()];
                if (M > best_value)best_value = M, best_index = i;
                if (!TMP.done())STATE.push(TMP.state_get()),Q_size++;
            }
            POLICY_NEW[I] = best_value;
            Max_delta = max(Max_delta, abs(POLICY[I] - best_value));
        }
        swap(POLICY, POLICY_NEW);
        return Max_delta;
    }
};


int main() {
    TicTacToeState state;
    TicTacToePolicy policy;
    int MT; cin >> MT;
    // TODO: 通过与环境多次交互，学习打败X策略的方法
    int MM = CLOCKS_PER_SEC * 15;
    clock_t t0 = clock();
    int expand_times = 0;
    memset(policy.POLICY, 0, sizeof(policy.POLICY));
    memset(policy.POLICY_NEW, 0, sizeof(policy.POLICY_NEW));
    while (clock() - t0 < MM) {
        expand_times++;
        double M=policy.study();
        cout << "expand_times: " << expand_times << " using time: " << clock() - t0 << " Max_delta: " << M << endl;
        if (M < 1e-9)break;
    }
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
*/
