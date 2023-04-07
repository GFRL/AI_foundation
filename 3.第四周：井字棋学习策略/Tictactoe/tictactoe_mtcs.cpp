/*
#include "problem/tictactoe.hpp"
#include<cstring>
#include<cmath>
using namespace std;
class Policy {
public:
    double probablity[10];//9�����ܸ���
};
class TicTacToePolicy {
public:
    Policy POLICY[1 << 18];
    int study[1 << 18];//����Ƿ�ѧϰ��
    int previous_choices[9];
    int previous_state[9];
    int round;
    int operator() (TicTacToeState state) {

        // �ֵ�Xʱ����ѡ���һ�����ж���
        if (state.active_player() == 0) {
            
            return state.action_space()[0];

        }
        else {
            int M = state.state_get(); round++; previous_state[round] = M;
            vector<int>tmp = state.action_space();
            int size = tmp.size();
            if (study[M]==0) {     
                for (int i = size - 1; i >= 0; i--) {
                    POLICY[M].probablity[tmp[i]] = 1;
                }
                POLICY[M].probablity[9] = size;
            }
            study[M]++;
            int i = -1; double x = rand()*1.0*POLICY[M].probablity[9] / RAND_MAX;
            double sum = 0;
            while (i < size-1) {
                i++; sum += POLICY[M].probablity[tmp[i]];
                if (x < sum)break;
            }
            // ѧϰ�õ�ֵ������֮�󣬰�������仰�滻��Ϊ����ֵ������̰��ѡ����
            previous_choices[round] = tmp[i];
            return state.action_space()[i];
        }
        
    }
    void value_update(int value) {
        while (round > 0) {
            double p = POLICY[previous_state[round]].probablity[previous_choices[round]]; 
            POLICY[previous_state[round]].probablity[previous_choices[round]] += p * value * pow(2, -1);
            if (POLICY[previous_state[round]].probablity[previous_choices[round]] > 10000)POLICY[previous_state[round]].probablity[previous_choices[round]] = 10000;
            else if (POLICY[previous_state[round]].probablity[previous_choices[round]] < 1e-5)POLICY[previous_state[round]].probablity[previous_choices[round]] = 0;
            POLICY[previous_state[round]].probablity[9] += POLICY[previous_state[round]].probablity[previous_choices[round]]-p;
            previous_choices[round] = previous_state[round] = 0; round--;
        }
    }
};


int main() {
    TicTacToeState state;
    TicTacToePolicy policy;

    // TODO: ͨ���뻷����ν�����ѧϰ���X���Եķ���
    int M = CLOCKS_PER_SEC *1/200;
    clock_t t0 = clock();
    int expand_times = 0;
    memset(policy.POLICY, 0, sizeof(policy.POLICY));
    memset(policy.study, 0, sizeof(policy.study));
    policy.round = 0;
    
    while (clock()-t0<M) {
        expand_times++;
        TicTacToeState Tmp;
        while (not Tmp.done()) {
            auto action = policy(Tmp);
            Tmp = Tmp.next(action);
        }
        policy.value_update(Tmp.rewards()[1]);
    }
    cout << expand_times << endl;
    
    // ����O�Ƿ��ܹ����X�Ĳ���
    while (not state.done()) {
        auto action = policy(state);
        state = state.next(action);
        state.show();
    }
    return 0;
}
*/
