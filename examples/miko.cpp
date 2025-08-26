#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <unordered_map>
#include <optional>
#include <array>
#include <cmath>
#include <limits>
#include <deque>
#include <numeric>
#include <algorithm>
#include "kklm.h"

// AlphaZero-style toy chess learner on a minimal pseudo-legal move set.
// Board: 12 planes [WP,WN,WB,WR,WQ,WK, BP,BN,BB,BR,BQ,BK] + side-to-move plane.
// Moves: limited pseudo moves (single-step pawns; knights; kings with captures). Not a full chess engine.
// MCTS with PUCT, Dirichlet noise at root, policy+value network, replay buffer training.

namespace miko {
	enum Piece { WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK, EMPTY };
	static const char *emoji_for(Piece p){
		switch(p){
			case WP: return "♙"; case WN: return "♘"; case WB: return "♗"; case WR: return "♖"; case WQ: return "♕"; case WK: return "♔";
			case BP: return "♟"; case BN: return "♞"; case BB: return "♝"; case BR: return "♜"; case BQ: return "♛"; case BK: return "♚";
			default: return "·";
		}
	}
	struct Move{ int from, to; };
	inline int move_index(const Move &m){ return m.from*64 + m.to; }
	struct Board{
		std::array<Piece,64> sq{};
		bool white_to_move=true;
		static Board start(){
			Board b{}; b.sq.fill(EMPTY);
			for(int f=0; f<8; ++f){ b.sq[8+f]=WP; b.sq[48+f]=BP; }
			b.sq[0]=WR; b.sq[7]=WR; b.sq[56]=BR; b.sq[63]=BR;
			b.sq[1]=WN; b.sq[6]=WN; b.sq[57]=BN; b.sq[62]=BN;
			b.sq[2]=WB; b.sq[5]=WB; b.sq[58]=BB; b.sq[61]=BB;
			b.sq[3]=WQ; b.sq[4]=WK; b.sq[59]=BQ; b.sq[60]=BK;
			return b;
		}
		void print() const{
			for(int r=7;r>=0;--r){ for(int f=0; f<8; ++f){ std::cout<<emoji_for(sq[r*8+f])<<" "; } std::cout<<"\n"; }
			std::cout<<(white_to_move?"White":"Black")<<" to move\n";
		}
		std::vector<float> encode() const{
			std::vector<float> x(13*64,0.f);
			for(int i=0;i<64;++i){ Piece p=sq[i]; if(p!=EMPTY && p<=BK){ x[int(p)*64 + i] = 1.f; } }
			for(int i=0;i<64;++i){ x[12*64 + i] = white_to_move ? 1.f : 0.f; }
			return x;
		}
	};

	static std::vector<Move> pseudo_moves(const Board &b){
		std::vector<Move> mv; mv.reserve(64);
		auto push=[&](int f,int t){ if(t>=0 && t<64){ int ff=f%8, tt=t%8; if(std::abs(ff-tt)<=2) mv.push_back({f,t}); } };
		for(int i=0;i<64;++i){ Piece p=b.sq[i]; if(p==EMPTY) continue; bool white = (p<=WK); if(white!=b.white_to_move) continue; int r=i/8, c=i%8;
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
	static bool is_terminal(const Board &b, int &winner){ bool wk=false,bk=false; for(int i=0;i<64;++i){ if(b.sq[i]==WK) wk=true; if(b.sq[i]==BK) bk=true; } if(!wk && !bk){ winner=0; return true; } if(!wk){ winner=-1; return true; } if(!bk){ winner=+1; return true; } winner=0; return false; }
}

// Policy+Value network (MLP trunk): 13*64 -> 256 -> 256 -> heads
struct PolicyValueNet : kllm::nn::Module {
	kllm::nn::Sequential trunk;
	std::shared_ptr<kllm::nn::Linear> head_policy;
	std::shared_ptr<kllm::nn::Linear> head_value;
	PolicyValueNet()
		: trunk({ std::make_shared<kllm::nn::Linear>(13*64, 256), std::make_shared<kllm::nn::Linear>(256, 256) })
		, head_policy(std::make_shared<kllm::nn::Linear>(256, 64*64))
		, head_value(std::make_shared<kllm::nn::Linear>(256, 1)) {}
	kllm::nn::ValuePtr forward(const kllm::nn::ValuePtr &x) override {
		auto h = trunk.forward(x);
		h = kllm::nn::relu(h);
		return head_policy->forward(h);
	}
	void forward_both(const kllm::nn::ValuePtr &x, kllm::nn::ValuePtr &policy_logits, kllm::nn::ValuePtr &value_out){
		auto h = trunk.forward(x);
		h = kllm::nn::relu(h);
		policy_logits = head_policy->forward(h);
		auto v = head_value->forward(h);
		value_out = v;
	}
	std::vector<kllm::nn::ValuePtr> parameters() override {
		auto ps = trunk.parameters();
		auto hp = head_policy->parameters(); ps.insert(ps.end(), hp.begin(), hp.end());
		auto hv = head_value->parameters(); ps.insert(ps.end(), hv.begin(), hv.end());
		return ps;
	}
};

// Soft-target cross-entropy over logits with target probabilities per row
static kllm::nn::ValuePtr cross_entropy_logits_soft(const kllm::nn::ValuePtr &logits, const std::vector<float> &target_probs_flat, std::size_t classes){
	using namespace kllm::nn;
	if(!logits || logits->shape.size()!=2) return nullptr;
	size_t B = logits->shape[0], C = logits->shape[1]; if(C!=classes) return nullptr;
	if(target_probs_flat.size() != B*C) return nullptr;
	auto out = Value::create({1}, logits->requires_grad);
	std::vector<float> sm(B*C);
	double loss = 0.0;
	for(size_t i=0;i<B;++i){
		float maxv = logits->values[i*C]; for(size_t c=1;c<C;++c) maxv = std::max(maxv, logits->values[i*C+c]);
		float sum=0.f; for(size_t c=0;c<C;++c){ float e = std::exp(logits->values[i*C+c]-maxv); sm[i*C+c]=e; sum+=e; }
		for(size_t c=0;c<C;++c) sm[i*C+c] /= (sum==0.f?1.f:sum);
		for(size_t c=0;c<C;++c){ float pi = target_probs_flat[i*C+c]; if(pi>0.f){ float p = sm[i*C+c]; loss += - double(pi) * std::log(p<=1e-12f?1e-12f:p); }
		}
	}
	out->values[0] = float(loss / double(B));
	out->parents = { logits };
	Value *Lp = logits.get(); Value *op = out.get();
	out->backward_fn = [op, Lp, sm, B, C, target_probs_flat](){ if(!Lp->requires_grad) return; float g = op->grad[0]/float(B); for(size_t i=0;i<B;++i){ for(size_t c=0;c<C;++c){ float grad = sm[i*C+c] - target_probs_flat[i*C+c]; Lp->grad[i*C+c] += g * grad; } } };
	return out;
}

struct MCTSConfig {
	int simulations = 128;
	float c_puct = 1.25f;
	float dirichlet_alpha = 0.3f;
	float dirichlet_eps = 0.25f;
	unsigned seed = 1234;
};

struct Node {
	miko::Board state;
	bool is_expanded = false;
	bool terminal = false;
	int terminal_winner = 0; // +1 white, -1 black, 0 none
	std::vector<int> actions;           // legal action indices (from*64+to)
	std::vector<float> priors;          // P
	std::vector<int> visit_counts;      // N
	std::vector<float> value_sum;       // W
	std::vector<float> mean_value;      // Q
	std::vector<std::unique_ptr<Node>> children; // for each action
	float leaf_value = 0.0f;            // V from network
};

struct MCTS {
	PolicyValueNet &net;
	MCTSConfig cfg;
	std::mt19937 rng;
	MCTS(PolicyValueNet &n, const MCTSConfig &c):net(n),cfg(c),rng(c.seed){}

	void add_dirichlet_noise(std::vector<float> &p){
		if(p.empty()) return;
		std::gamma_distribution<float> gamma(cfg.dirichlet_alpha, 1.0f);
		std::vector<float> noise(p.size());
		float sum=0.f; for(float &v:noise){ v = gamma(rng); sum += v; }
		if(sum<=0.f) return;
		for(size_t i=0;i<p.size();++i){ float n = noise[i]/sum; p[i] = (1.0f - cfg.dirichlet_eps)*p[i] + cfg.dirichlet_eps*n; }
	}

	void expand(Node &node){
		if(node.is_expanded) return;
		int w; node.terminal = miko::is_terminal(node.state, w); node.terminal_winner = w;
		if(node.terminal){ node.is_expanded = true; node.leaf_value = (w==0?0.0f:(node.state.white_to_move? (w>0?+1.f:-1.f) : (w>0?-1.f:+1.f))); return; }
		auto legal = miko::pseudo_moves(node.state);
		std::vector<float> x = node.state.encode();
		auto inp = kllm::nn::tensor(x, {1, 13*64}, false);
		kllm::nn::ValuePtr policy_logits, value_out;
		net.forward_both(inp, policy_logits, value_out);
		// Extract policy over legal moves, mask and normalize
		std::vector<float> p_legal; p_legal.reserve(legal.size());
		std::vector<int> acts; acts.reserve(legal.size());
		for(const auto &m : legal){ int idx = miko::move_index(m); acts.push_back(idx); p_legal.push_back(policy_logits->values[idx]); }
		// softmax over legal
		float maxv = p_legal.empty()?0.f:p_legal[0]; for(float v: p_legal) maxv = std::max(maxv, v);
		float sum=0.f; for(float &v: p_legal){ v = std::exp(v - maxv); sum += v; }
		if(sum<=0.f){ for(float &v: p_legal) v = 1.0f / float(p_legal.size()); }
		else { for(float &v: p_legal) v /= sum; }
		// Root noise will be added by caller
		node.actions = std::move(acts);
		node.priors = std::move(p_legal);
		node.visit_counts.assign(node.actions.size(), 0);
		node.value_sum.assign(node.actions.size(), 0.0f);
		node.mean_value.assign(node.actions.size(), 0.0f);
		node.children.resize(node.actions.size());
		node.leaf_value = value_out->values[0];
		node.is_expanded = true;
	}

	float simulate(Node &root){
		std::vector<Node*> path; path.reserve(128);
		Node *node = &root; path.push_back(node);
		// Selection
		while(node->is_expanded && !node->terminal){
			int total_N = 0; for(int n : node->visit_counts) total_N += n; float sqrt_sum = std::sqrt(float(total_N) + 1e-6f);
			int best_i = 0; float best_score = -std::numeric_limits<float>::infinity();
			for(size_t i=0;i<node->actions.size();++i){ float Q = node->mean_value[i]; float U = cfg.c_puct * node->priors[i] * sqrt_sum / (1.0f + float(node->visit_counts[i])); float s = Q + U; if(s > best_score){ best_score = s; best_i = int(i);} }
			// Move to child
			if(!node->children[best_i]){
				node->children[best_i] = std::make_unique<Node>();
				node->children[best_i]->state = node->state;
				miko::Move m{ node->actions[best_i]/64, node->actions[best_i]%64 };
				miko::apply_move(node->children[best_i]->state, m);
			}
			node = node->children[best_i].get();
			path.push_back(node);
			if(!node->is_expanded) break;
		}
		// Expand leaf
		expand(*node);
		float v = node->leaf_value; // Value for side-to-move at leaf
		// Backup (alternate sign going up the path except for leaf node perspective)
		for(int i=int(path.size())-2; i>=0; --i){ Node *parent = path[size_t(i)]; Node *child = path[size_t(i+1)];
			// Identify which action index led to child
			int aidx = -1; for(size_t k=0;k<parent->children.size();++k){ if(parent->children[k].get()==child){ aidx = int(k); break; } }
			if(aidx<0) continue;
			parent->visit_counts[size_t(aidx)] += 1;
			parent->value_sum[size_t(aidx)] += v;
			parent->mean_value[size_t(aidx)] = parent->value_sum[size_t(aidx)] / float(parent->visit_counts[size_t(aidx)]);
			v = -v; // next parent is opponent
		}
		return v;
	}

	std::vector<float> run(const miko::Board &start, bool add_noise){
		Node root{}; root.state = start;
		expand(root);
		if(add_noise){ add_dirichlet_noise(root.priors); }
		for(int s=0; s<cfg.simulations; ++s){ simulate(root); }
		// Produce visit distribution over all 4096 moves
		std::vector<float> pi(64*64, 0.0f);
		float sum = 0.f; for(size_t i=0;i<root.actions.size();++i){ float v = float(root.visit_counts[i]); pi[root.actions[i]] = v; sum += v; }
		if(sum>0.f){ for(float &v : pi) v /= sum; }
		return pi;
	}
};

struct ReplaySample { std::vector<float> state; std::vector<float> pi; float z; };

int main(){
	using namespace kllm::nn;
	// KLLM tuning (CPU)
	kllm::set_num_threads(std::min<unsigned>(8u, std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 8u));
	kllm::set_parallel_threshold(1<<14);
	kllm::set_large_slab_bytes(1<<20);
	kllm::global_config().enable_matmul_blocked = true;
	kllm::global_config().matmul_block_m = 64;
	kllm::global_config().matmul_block_n = 128;
	kllm::global_config().matmul_block_k = 128;
	kllm::global_config().release_tls_fused_buffers = true;

	std::mt19937 rng(123);
	PolicyValueNet net; auto params = collect_parameters(net);
	Adam opt(params, 1e-3f);

	MCTSConfig mcfg; mcfg.simulations = 128; mcfg.c_puct = 1.25f; mcfg.dirichlet_alpha = 0.3f; mcfg.dirichlet_eps = 0.25f; mcfg.seed = 777;
	MCTS mcts(net, mcfg);

	// Replay buffer
	std::deque<ReplaySample> buffer; const size_t buffer_cap = 5000; const size_t batch_size = 32;

	auto train_step = [&](size_t steps){
		if(buffer.size() < batch_size) return;
		std::uniform_int_distribution<size_t> uid(0, buffer.size()-1);
		for(size_t step=0; step<steps; ++step){
			std::vector<float> X; X.reserve(batch_size * 13 * 64);
			std::vector<float> PI; PI.reserve(batch_size * 64 * 64);
			std::vector<float> Z; Z.reserve(batch_size);
			for(size_t i=0;i<batch_size;++i){ const auto &s = buffer[uid(rng)]; X.insert(X.end(), s.state.begin(), s.state.end()); PI.insert(PI.end(), s.pi.begin(), s.pi.end()); Z.push_back(s.z); }
			auto x = tensor(X, {batch_size, 13*64}, false);
			ValuePtr logits, value; net.forward_both(x, logits, value);
			auto loss_pi = cross_entropy_logits_soft(logits, PI, 64*64);
			auto zt = tensor(Z, {batch_size, 1}, false);
			auto loss_v = mse_loss(value, zt);
			auto loss = add(loss_pi, loss_v);
			opt.zero_grad(); loss->backward(); opt.step();
		}
	};

	int episodes = 6; int max_moves = 60; // keep runtime reasonable
	for(int ep=0; ep<episodes; ++ep){
		miko::Board b = miko::Board::start();
		std::vector<std::vector<float>> states; std::vector<std::vector<float>> dists; states.reserve(max_moves); dists.reserve(max_moves);
		int winner = 0;
		for(int t=0; t<max_moves; ++t){
			// MCTS to get visit distribution
			auto pi = mcts.run(b, /*add_noise=*/(t==0));
			states.push_back(b.encode()); dists.push_back(pi);
			// Sample move proportional to pi^tau (tau=1 here); fallback to argmax
			std::vector<miko::Move> legal = miko::pseudo_moves(b);
			std::vector<float> probs; probs.reserve(legal.size()); float sum=0.f;
			for(const auto &m: legal){ float v = pi[miko::move_index(m)]; probs.push_back(v); sum += v; }
			if(sum<=0.f){ for(float &v: probs) v = 1.0f / float(probs.size()); }
			else { for(float &v: probs) v /= sum; }
			std::discrete_distribution<int> dd(probs.begin(), probs.end());
			int choice = dd(rng);
			miko::apply_move(b, legal[size_t(choice)]);
							if(miko::is_terminal(b, winner)){ break; }
		}
		// Assign outcomes to samples from side-to-move perspective at each state
		float z_final = (winner==0?0.f:(winner>0?+1.f:-1.f));
		bool side = true; // start white
		for(size_t i=0;i<states.size(); ++i){ float z = side ? z_final : -z_final; buffer.push_back({states[i], dists[i], z}); if(buffer.size()>buffer_cap) buffer.pop_front(); side = !side; }
		train_step(64);
		std::cout << "Episode "<<ep<<" done, winner="<<(winner>0?"White":(winner<0?"Black":"Draw"))<<", buffer="<<buffer.size()<<"\n";
	}

	// Save model
	auto st = kllm::nn::save_module(net, "miko.ckpt"); if(!st.ok()) std::cout<<"save failed: "<<st.message<<"\n";
	// Quick sanity: run a short visual game
	miko::Board show = miko::Board::start();
	for(int t=0; t<16; ++t){ auto pi = mcts.run(show, false); int best=0; for(int i=1;i<64*64;++i) if(pi[i]>pi[best]) best=i; miko::Move m{best/64, best%64}; miko::apply_move(show, m); }
	std::cout<<"Final state after greedy moves:\n"; show.print();
	std::cout<<"Saved and trained miko policy-value.\n";
	return 0;
}