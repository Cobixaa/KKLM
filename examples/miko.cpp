#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <unordered_map>
#include <optional>
#include "kklm.h"

// Minimal chess representation (8x8) with emoji rendering. Not a full legal move generator.
// Board encoding: 12 planes one-hot [WP,WN,WB,WR,WQ,WK, BP,BN,BB,BR,BQ,BK] of size 64.
// Moves: sample random legal-like pseudo-moves (very limited: single-step pawns, random knight, king).
// This is a toy to showcase self-play + learning plumbing; not a strong or legal engine.

namespace miko {
	enum Piece { WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK, EMPTY };
	static const char *emoji_for(Piece p){
		switch(p){
			case WP: return "♙"; case WN: return "♘"; case WB: return "♗"; case WR: return "♖"; case WQ: return "♕"; case WK: return "♔";
			case BP: return "♟"; case BN: return "♞"; case BB: return "♝"; case BR: return "♜"; case BQ: return "♛"; case BK: return "♚";
			default: return "·";
		}
	}
	struct Board{
		std::array<Piece,64> sq;
		bool white_to_move=true;
		static Board start(){
			Board b{}; b.sq.fill(EMPTY);
			// pawns
			for(int f=0; f<8; ++f){ b.sq[8+f]=WP; b.sq[48+f]=BP; }
			// rooks/knights/bishops
			b.sq[0]=WR; b.sq[7]=WR; b.sq[56]=BR; b.sq[63]=BR;
			b.sq[1]=WN; b.sq[6]=WN; b.sq[57]=BN; b.sq[62]=BN;
			b.sq[2]=WB; b.sq[5]=WB; b.sq[58]=BB; b.sq[61]=BB;
			b.sq[3]=WQ; b.sq[4]=WK; b.sq[59]=BQ; b.sq[60]=BK;
			return b;
		}
		void print() const{
			for(int r=7;r>=0;--r){
				for(int f=0; f<8; ++f){ std::cout<<emoji_for(sq[r*8+f])<<" "; }
				std::cout<<"\n";
			}
			std::cout<<(white_to_move?"White":"Black")<<" to move\n";
		}
		// Encode into 12x64 float vector
		std::vector<float> encode() const{
			std::vector<float> x(12*64,0.f);
			for(int i=0;i<64;++i){ Piece p=sq[i]; if(p!=EMPTY && p<=BK){ x[int(p)*64 + i] = 1.f; } }
			return x;
		}
	};
	struct Move{ int from, to; };
	static std::vector<Move> pseudo_moves(const Board &b){
		std::vector<Move> mv; mv.reserve(48);
		auto push=[&](int f,int t){ if(t>=0 && t<64) mv.push_back({f,t}); };
		for(int i=0;i<64;++i){ Piece p=b.sq[i]; if(p==EMPTY) continue; bool white = (p<=WK);
			if(white!=b.white_to_move) continue;
			int r=i/8, c=i%8;
			auto add_knight=[&](){ int drc[8][2]={{2,1},{2,-1},{-2,1},{-2,-1},{1,2},{1,-2},{-1,2},{-1,-2}}; for(auto &d: drc){ int rr=r+d[0], cc=c+d[1]; if(rr>=0&&rr<8&&cc>=0&&cc<8){ int t=rr*8+cc; if(b.sq[t]==EMPTY || (white!=(b.sq[t]<=WK))) push(i,t);} } };
			switch(p){
				case WP: { int t=i+8; if(t<64 && b.sq[t]==EMPTY) push(i,t); } break;
				case BP: { int t=i-8; if(t>=0 && b.sq[t]==EMPTY) push(i,t); } break;
				case WN: case BN: add_knight(); break;
				case WK: case BK: { for(int dr=-1; dr<=1; ++dr) for(int dc=-1; dc<=1; ++dc){ if(dr==0&&dc==0) continue; int rr=r+dr, cc=c+dc; if(rr>=0&&rr<8&&cc>=0&&cc<8){ int t=rr*8+cc; if(b.sq[t]==EMPTY || (white!=(b.sq[t]<=WK))) push(i,t);} } } break;
				default: break;
			}
		}
		if(mv.empty()) mv.push_back({0,0});
		return mv;
	}
	static void apply_move(Board &b, const Move &m){ auto p=b.sq[m.from]; b.sq[m.to]=p; b.sq[m.from]=EMPTY; b.white_to_move=!b.white_to_move; }
}

// Simple policy network: Linear(12*64 -> 128) -> ReLU -> Linear(128 -> 64*64) producing logits over from-to squares.
struct PolicyNet : kllm::nn::Module {
	kllm::nn::Sequential trunk;
	std::shared_ptr<kllm::nn::Linear> head;
	PolicyNet():trunk({ std::make_shared<kllm::nn::Linear>(12*64,128), }), head(std::make_shared<kllm::nn::Linear>(128, 64*64)){}
	kllm::nn::ValuePtr forward(const kllm::nn::ValuePtr &x) override {
		auto h = trunk.forward(x);
		h = kllm::nn::relu(h);
		return head->forward(h);
	}
	std::vector<kllm::nn::ValuePtr> parameters() override {
		auto ps = trunk.parameters();
		auto hp = head->parameters();
		ps.insert(ps.end(), hp.begin(), hp.end());
		return ps;
	}
};

static int argmax(const std::vector<float> &v){ int a=0; float best=v[0]; for(int i=1;i<(int)v.size();++i){ if(v[i]>best){ best=v[i]; a=i; } } return a; }

int main(){
	using namespace kllm::nn;
	std::mt19937 rng(123);
	miko::Board b = miko::Board::start();
	PolicyNet net;
	auto params = collect_parameters(net);
	Adam opt(params, 1e-3f);

	// Self-play episodes
	for(int episode=0; episode<5; ++episode){
		b = miko::Board::start();
		for(int t=0; t<40; ++t){
			std::vector<float> x = b.encode();
			auto inp = tensor(x, {1, 12*64}, false);
			auto logits = net.forward(inp);
			// Sample a random pseudo-legal move and set it as target one-hot
			auto mv = miko::pseudo_moves(b);
			int chosen = mv[rng()%mv.size()].from*64 + mv[rng()%mv.size()].to;
			std::vector<int> lab = { chosen };
			auto loss = cross_entropy_logits(logits, lab);
			opt.zero_grad(); loss->backward(); opt.step();
			// Play the move greedily from logits for visualization
			int pred = argmax(logits->values);
			miko::Move step{ pred/64, pred%64 };
			miko::apply_move(b, step);
		}
		std::cout<<"Episode "<<episode<<" state:\n"; b.print();
	}

	// Demonstrate save/load of model parameters
	auto st = kllm::nn::save_module(net, "miko.ckpt");
	if(!st.ok()) std::cout<<"save failed: "<<st.message<<"\n";
	st = kllm::nn::load_module(net, "miko.ckpt");
	if(!st.ok()) std::cout<<"load failed: "<<st.message<<"\n";
	std::cout<<"Saved and loaded miko policy.\n";
	return 0;
}