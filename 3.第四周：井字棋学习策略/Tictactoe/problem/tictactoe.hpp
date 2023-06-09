#pragma once

#include <vector>
#include <cassert>
#include <iostream>
#include <cinttypes>

#include "../interface/game_state.hpp"

// 3*3井字棋
class TicTacToeState : public NPlayerGameStateBase<int, 2> {
private:

    // X先手
    static constexpr int X = 0b01, O = 0b10;
    
    uint32_t _state;
    std::vector<int> _action_space;
    int round;

    inline uint32_t take_at(int index) const {
        return (_state >> (index << 1)) & 0b11;
    }

    inline void put_at(int index){
        int piece = active_player() == 0 ? X : O;
        assert(index >= 0 and index < 9 and take_at(index) == 0);
        _state |= (piece << (index << 1));
    }

    inline bool full() const {
        return round == 9;
    }

    bool x_win() const {
        return test_lines_of(X);
    }

    bool o_win() const {
        return test_lines_of(O);
    }

    bool test_lines_of(int piece) const {
        const static uint32_t 
            row_mask =    0b000000000000111111, 
            column_mask = 0b000011000011000011,
            lr_mask =     0b110000001100000011, 
            rl_mask =     0b000011001100110000;
        
        if (round < 5){
            return false;
        }

        const static uint32_t _row_target = (X|(X<<2)|(X<<4)),
            _column_target = (X|(X<<6)|(X<<12)),
            _lr_target = (X|(X<<8)|(X<<16)),
            _rl_target = ((X<<4)|(X<<8)|(X<<12));

        uint32_t row_target = (_row_target << (piece==O)),
            column_target = (_column_target << (piece==O)),
            lr_target = (_lr_target << (piece==O)),
            rl_target = (_rl_target << (piece==O));

        return ((_state & row_mask) == row_target
            or ((_state >> 6) & row_mask) == row_target
            or ((_state >> 12) & row_mask) == row_target
            or (_state & column_mask) == column_target
            or ((_state >> 2) & column_mask) == column_target
            or ((_state >> 4) & column_mask) == column_target
            or (_state & lr_mask) == lr_target
            or (_state & rl_mask) == rl_target
        );
    }

    void update_action_space(){
        _action_space.clear();
        for (int i = 0; i < 9; ++ i){
            if (take_at(i) == 0){
                _action_space.push_back(i);
            }
        }
    }

public:

    TicTacToeState() : _state(0), round(0) {
        update_action_space();
    }

    void state_set(int u) {
        _state = u; round = 0;
        for (int i = 1; i <= 18; i++) {
            if (u & 1)round++; 
           u= u >> 1;
        }
        update_action_space();
    }
    //传入绝对坐标;
    int state_try_get(int index) {
        int piece = active_player() == 0 ? X : O;
        assert(index >= 0 and index < 9 and take_at(index) == 0);
        return _state |= (piece << (index << 1));
    }
    int state_get() {
        return _state;
    }

    inline int n_actions() const override {
        return 9 - round;
    }

    int active_player() const override {
        return round & 1;
    }
    
    std::vector<int> action_space() const override {
        return _action_space;
    }

    std::vector<double> rewards() const override {
        static const std::vector<double> scores_x_win {1, -1},
            scores_o_win {-1, 1},
            scores_tie {0, 0};
        
        return x_win() ? scores_x_win : (o_win() ? scores_o_win : scores_tie);
    }

    std::vector<double> cumulative_rewards() const override {
        return rewards();
    }

    bool done() const override {
        return full() or x_win() or o_win();
    }

    void show() const override {
        static const char pieces[] = "_XO";
        for (int i = 0; i < 3; ++ i){
            for (int j = 0; j < 3; ++ j){
                std::cout << pieces[take_at(i * 3 + j)];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    const TicTacToeState& next(const int& action) const override {
        static TicTacToeState next_state;
        next_state = *this;
        next_state.put_at(action);
        next_state.update_action_space();
        ++ next_state.round;
        return next_state;
    }
    friend struct std::hash<TicTacToeState>;

    friend bool operator== (const TicTacToeState& s1, const TicTacToeState& s2){
        return s1._state == s2._state;
    }
};

template<>
struct std::hash<TicTacToeState>{
    size_t operator()(const TicTacToeState& s) const {
        return s._state;
    }
};
