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
#include <sstream>
#include <chrono>
#include <cctype>
#include "kklm.h"
#include "chess.h"

// AlphaZero-style legal chess learner using bitboards and 8x8x73 policy mapping.

struct PolicyValueNet : kllm::nn::Module {
	kllm::nn::Sequential trunk;
	std::shared_ptr<kllm::nn::Linear> head_policy;
	std::shared_ptr<kllm::nn::Linear> head_value;
	PolicyValueNet()
		: trunk({ std::make_shared<kllm::nn::Linear>(13*64, 256), std::make_shared<kllm::nn::Linear>(256, 256) })
		, head_policy(std::make_shared<kllm::nn::Linear>(256, 8*8*73))
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
		value_out = head_value->forward(h);
	}
	std::vector<kllm::nn::ValuePtr> parameters() override {
		auto ps = trunk.parameters();
		auto hp = head_policy->parameters(); ps.insert(ps.end(), hp.begin(), hp.end());
		auto hv = head_value->parameters(); ps.insert(ps.end(), hv.begin(), hv.end());
		return ps;
	}
};

static kllm::nn::ValuePtr cross_entropy_logits_soft(const kllm::nn::ValuePtr &logits, const std::vector<float> &target_probs_flat, std::size_t classes){
	using namespace kllm::nn;
	if(!logits || logits->shape.size()!=2) return nullptr;
	size_t B = logits->shape[0], C = logits->shape[1]; if(C!=classes) return nullptr;
	if(target_probs_flat.size() != B*C) return nullptr;
	auto out = Value::create({1}, logits->requires_grad);
	std::vector<float> sm(B*C);
	double loss = 0.0;
	for(size_t i=0;i<B;++i){ float maxv = logits->values[i*C]; for(size_t c=1;c<C;++c) maxv = std::max(maxv, logits->values[i*C+c]); float sum=0.f; for(size_t c=0;c<C;++c){ float e = std::exp(logits->values[i*C+c]-maxv); sm[i*C+c]=e; sum+=e; } for(size_t c=0;c<C;++c) sm[i*C+c] /= (sum==0.f?1.f:sum); for(size_t c=0;c<C;++c){ float pi = target_probs_flat[i*C+c]; if(pi>0.f){ float p = sm[i*C+c]; loss += - double(pi) * std::log(p<=1e-12f?1e-12f:p); } } }
	out->values[0] = float(loss / double(B)); out->parents = { logits };
	kllm::nn::Value *Lp = logits.get(); kllm::nn::Value *op = out.get();
	out->backward_fn = [op, Lp, sm, B, C, target_probs_flat](){ if(!Lp->requires_grad) return; float g = op->grad[0]/float(B); for(size_t i=0;i<B;++i){ for(size_t c=0;c<C;++c){ float grad = sm[i*C+c] - target_probs_flat[i*C+c]; Lp->grad[i*C+c] += g * grad; } } };
	return out;
}

struct MCTSConfig { int simulations=256; float c_puct=1.25f; float dirichlet_alpha=0.3f; float dirichlet_eps=0.25f; unsigned seed=1234; };

struct Node {
	azchess::Position state;
	bool is_expanded=false; bool terminal=false; int terminal_winner=0;
	std::vector<int> actions; std::vector<float> priors; std::vector<int> visit_counts; std::vector<float> value_sum; std::vector<float> mean_value; std::vector<std::unique_ptr<Node>> children; float leaf_value=0.0f;
};

struct MCTS {
	PolicyValueNet &net; MCTSConfig cfg; std::mt19937 rng;
	MCTS(PolicyValueNet &n, const MCTSConfig &c):net(n),cfg(c),rng(c.seed){}
	void add_dirichlet_noise(std::vector<float>&p){ if(p.empty()) return; std::gamma_distribution<float> g(cfg.dirichlet_alpha,1.0f); std::vector<float> n(p.size()); float s=0.f; for(float &v:n){ v=g(rng); s+=v; } if(s<=0.f) return; for(size_t i=0;i<p.size();++i){ n[i]/=s; p[i]=(1.0f-cfg.dirichlet_eps)*p[i]+cfg.dirichlet_eps*n[i]; }}
	void expand(Node &node){ if(node.is_expanded) return; int w = azchess::winner_on_terminal(node.state); node.terminal = (w!=0) || azchess::no_legal_moves(node.state); node.terminal_winner = w; if(node.terminal){ node.is_expanded=true; node.leaf_value = (w==0?0.0f:(node.state.white_to_move? (w>0?+1.f:-1.f) : (w>0?-1.f:+1.f))); return; }
		auto legal = azchess::legal_moves(node.state);
		std::vector<float> x = azchess::encode_features(node.state);
		auto inp = kllm::nn::tensor(x, {1, 13*64}, false);
		kllm::nn::ValuePtr policy_logits, value_out; net.forward_both(inp, policy_logits, value_out);
		std::vector<float> probs(8*8*73, 0.0f);
		for(const auto &m : legal){ int j = azchess::az73_index_for(m, node.state.white_to_move); probs[j] = policy_logits->values[j]; }
		// softmax over legal only
		float maxv = -1e30f; for(const auto &m: legal){ int j=azchess::az73_index_for(m,node.state.white_to_move); maxv = std::max(maxv, probs[j]); }
		float sum=0.f; for(const auto &m: legal){ int j=azchess::az73_index_for(m,node.state.white_to_move); float e = std::exp(probs[j]-maxv); probs[j]=e; sum+=e; }
		std::vector<int> acts; acts.reserve(legal.size()); std::vector<float> pri; pri.reserve(legal.size());
		for(const auto &m: legal){ int j=azchess::az73_index_for(m,node.state.white_to_move); float p = (sum>0.f? probs[j]/sum : 1.0f/float(legal.size())); acts.push_back(j); pri.push_back(p); }
		node.actions = std::move(acts); node.priors = std::move(pri); node.visit_counts.assign(node.actions.size(), 0); node.value_sum.assign(node.actions.size(), 0.0f); node.mean_value.assign(node.actions.size(), 0.0f); node.children.resize(node.actions.size()); node.leaf_value = value_out->values[0]; node.is_expanded=true; }
	float simulate(Node &root){ std::vector<Node*> path; path.reserve(256); Node *node=&root; path.push_back(node); while(node->is_expanded && !node->terminal){ int total=0; for(int n: node->visit_counts) total+=n; float srt = std::sqrt(float(total)+1e-6f); int best=0; float bests=-1e30f; for(size_t i=0;i<node->actions.size();++i){ float Q=node->mean_value[i]; float U = cfg.c_puct * node->priors[i] * srt / (1.0f + float(node->visit_counts[i])); float sc=Q+U; if(sc>bests){ bests=sc; best=int(i);} } if(!node->children[best]){ node->children[best]=std::make_unique<Node>(); node->children[best]->state = node->state; // decode move index back to Move
			int idx = node->actions[best]; auto legal = azchess::legal_moves(node->state); bool applied=false; for(const auto &m: legal){ if(azchess::az73_index_for(m, node->state.white_to_move)==idx){ azchess::apply_move(node->children[best]->state, m); applied=true; break; } } if(!applied){ // fallback to first legal
				if(!legal.empty()) azchess::apply_move(node->children[best]->state, legal[0]); }
		}
		node = node->children[best].get(); path.push_back(node); if(!node->is_expanded) break; }
		expand(*node); float v=node->leaf_value; for(int i=int(path.size())-2; i>=0; --i){ Node *parent=path[size_t(i)], *child=path[size_t(i+1)]; int aidx=-1; for(size_t k=0;k<parent->children.size();++k){ if(parent->children[k].get()==child){ aidx=int(k); break; } } if(aidx<0) continue; parent->visit_counts[size_t(aidx)]+=1; parent->value_sum[size_t(aidx)]+=v; parent->mean_value[size_t(aidx)] = parent->value_sum[size_t(aidx)]/float(parent->visit_counts[size_t(aidx)]); v=-v; } return v; }
	std::vector<float> run(const azchess::Position &start, bool add_noise){ Node root{}; root.state = start; expand(root); if(add_noise){ add_dirichlet_noise(root.priors); } for(int s=0;s<cfg.simulations;++s){ simulate(root);} std::vector<float> pi(8*8*73, 0.0f); float sum=0.f; for(size_t i=0;i<root.actions.size();++i){ float v=float(root.visit_counts[i]); pi[root.actions[i]]=v; sum+=v; } if(sum>0.f){ for(float &v: pi) v/=sum; } return pi; }
};

struct ReplaySample { std::vector<float> state; std::vector<float> pi; float z; };

// --- Helpers: board print and UCI ---
static std::string sq_to_coord(int s){ int f = s%8, r = s/8; std::string c; c.push_back(char('a'+f)); c.push_back(char('1'+r)); return c; }
static char piece_char(int color, int pc){ const char *W = "PNBRQK"; const char *B = "pnbrqk"; return color==azchess::WHITE ? W[pc] : B[pc]; }
static void print_position(const azchess::Position &p){ for(int r=7;r>=0;--r){ for(int f=0;f<8;++f){ int s=r*8+f; char ch='.'; for(int c=0;c<2;++c){ for(int pc=0;pc<6;++pc){ if(p.pieces[c][pc] & azchess::bit(s)){ ch = piece_char(c, pc); } } } std::cout<<ch<<' '; } std::cout<<"  "<<r+1<<"\n"; } std::cout<<"a b c d e f g h\n"; std::cout<<(p.white_to_move?"White":"Black")<<" to move\n"; }
static std::string move_to_uci(const azchess::Move &m){ std::string u = sq_to_coord(m.from) + sq_to_coord(m.to); if(m.promote>=0){ char pr='q'; if(m.promote==azchess::KNIGHT) pr='n'; else if(m.promote==azchess::BISHOP) pr='b'; else if(m.promote==azchess::ROOK) pr='r'; else if(m.promote==azchess::QUEEN) pr='q'; u.push_back(pr); } return u; }
static bool uci_to_move(const azchess::Position &p, const std::string &uci, azchess::Move &out){ if(uci.size()<4) return false; auto coord_to_sq=[&](char f, char r){ if(f<'a'||f>'h'||r<'1'||r>'8') return -1; return (r-'1')*8 + (f-'a'); }; int from = coord_to_sq(uci[0], uci[1]); int to = coord_to_sq(uci[2], uci[3]); if(from<0||to<0) return false; int promo=-1; if(uci.size()>=5){ char c=std::tolower(uci[4]); if(c=='n') promo=azchess::KNIGHT; else if(c=='b') promo=azchess::BISHOP; else if(c=='r') promo=azchess::ROOK; else if(c=='q') promo=azchess::QUEEN; }
	auto legal = azchess::legal_moves(p); for(const auto &m: legal){ if(m.from==from && m.to==to){ if(m.promote>=0){ if(promo==m.promote) { out=m; return true; } else continue; } out=m; return true; } } return false; }

// UCI loop
static void uci_loop(PolicyValueNet &net){ std::ios::sync_with_stdio(false); std::cin.tie(nullptr); MCTSConfig cfg; cfg.simulations=256; MCTS mcts(net, cfg); azchess::Position pos = azchess::startpos(); std::string line; while(std::getline(std::cin, line)){ std::istringstream iss(line); std::string cmd; if(!(iss>>cmd)) continue; if(cmd=="uci"){ std::cout<<"id name Miko-AlphaZero\n"; std::cout<<"id author miko\n"; std::cout<<"uciok\n"; }
		else if(cmd=="isready"){ std::cout<<"readyok\n"; }
		else if(cmd=="ucinewgame"){ pos = azchess::startpos(); }
		else if(cmd=="position"){ std::string type; iss>>type; if(type=="startpos"){ pos = azchess::startpos(); std::string token; if(iss>>token){ if(token=="moves"){ std::string mv; while(iss>>mv){ azchess::Move m{}; if(uci_to_move(pos, mv, m)) azchess::apply_move(pos, m); } } } } }
		else if(cmd=="go"){ int movetime_ms=0, depth=0; std::string t; while(iss>>t){ if(t=="movetime"){ iss>>movetime_ms; } else if(t=="depth"){ iss>>depth; } }
			// Simple: ignore depth/time and use fixed simulations
			auto pi = mcts.run(pos, false); auto legal = azchess::legal_moves(pos); if(legal.empty()){ std::cout<<"bestmove 0000\n"; continue; } int best=0; float bestp=-1.0f; for(size_t i=0;i<legal.size();++i){ int j=azchess::az73_index_for(legal[i], pos.white_to_move); float p = pi[j]; if(p>bestp){ bestp=p; best=int(i);} } std::string u = move_to_uci(legal[size_t(best)]); std::cout<<"bestmove "<<u<<"\n"; }
		else if(cmd=="quit"||cmd=="stop"){ break; }
	}
}

int main(){ using namespace kllm::nn; kllm::set_num_threads(std::min<unsigned>(8u, std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 8u)); kllm::set_parallel_threshold(1<<14); kllm::set_large_slab_bytes(1<<20); kllm::global_config().enable_matmul_blocked = true; kllm::global_config().matmul_block_m = 64; kllm::global_config().matmul_block_n = 128; kllm::global_config().matmul_block_k = 128; kllm::global_config().release_tls_fused_buffers = true;
	std::mt19937 rng(123); PolicyValueNet net; auto params = collect_parameters(net); Adam opt(params, 1e-3f); MCTSConfig mcfg; mcfg.simulations=256; mcfg.seed=777; MCTS mcts(net, mcfg);
	std::deque<ReplaySample> buffer; const size_t buffer_cap=20000; const size_t batch_size=64;
	auto train_step = [&](size_t steps){ if(buffer.size()<batch_size) return; std::uniform_int_distribution<size_t> uid(0, buffer.size()-1); for(size_t step=0; step<steps; ++step){ std::vector<float> X; X.reserve(batch_size*13*64); std::vector<float> PI; PI.reserve(batch_size*8*8*73); std::vector<float> Z; Z.reserve(batch_size); for(size_t i=0;i<batch_size;++i){ const auto &s=buffer[uid(rng)]; X.insert(X.end(), s.state.begin(), s.state.end()); PI.insert(PI.end(), s.pi.begin(), s.pi.end()); Z.push_back(s.z);} auto x=tensor(X,{batch_size,13*64},false); ValuePtr logits,value; net.forward_both(x,logits,value); auto loss_pi = cross_entropy_logits_soft(logits, PI, 8*8*73); auto zt = tensor(Z,{batch_size,1},false); auto loss_v = mse_loss(value, zt); auto loss = add(loss_pi, loss_v); opt.zero_grad(); loss->backward(); opt.step(); } };
	// Menu loop
	for(;;){ std::cout<<"\n--- Miko Menu ---\n"; std::cout<<"1) Train self-play\n2) Play vs engine (UCI moves)\n3) Perft\n4) UCI mode\n5) Save model\n6) Load model\n7) Exit\n> "; std::cout.flush(); int choice=0; if(!(std::cin>>choice)){ break; } if(choice==1){ int episodes=2, max_moves=80; std::cout<<"episodes? "; std::cin>>episodes; std::cout<<"max moves? "; std::cin>>max_moves; for(int ep=0; ep<episodes; ++ep){ azchess::Position s = azchess::startpos(); std::vector<std::vector<float>> states; std::vector<std::vector<float>> dists; int winner=0; for(int t=0;t<max_moves;++t){ auto pi = mcts.run(s, /*add_noise=*/(t==0)); states.push_back(azchess::encode_features(s)); dists.push_back(pi); auto legal = azchess::legal_moves(s); if(legal.empty()){ break; } std::vector<float> probs; probs.reserve(legal.size()); float sum=0.f; for(const auto &m: legal){ float v = pi[azchess::az73_index_for(m, s.white_to_move)]; probs.push_back(v); sum+=v; } if(sum<=0.f){ for(float &v: probs) v = 1.0f/float(probs.size()); } else { for(float &v: probs) v/=sum; } std::discrete_distribution<int> dd(probs.begin(), probs.end()); int choice_mv=dd(rng); azchess::apply_move(s, legal[size_t(choice_mv)]); int w = azchess::winner_on_terminal(s); if(w!=0){ winner=w; break; } }
			float z_final = (winner==0?0.f:(winner>0?+1.f:-1.f)); bool side=true; for(size_t i=0;i<states.size(); ++i){ float z = side ? z_final : -z_final; buffer.push_back({states[i], dists[i], z}); if(buffer.size()>buffer_cap) buffer.pop_front(); side = !side; } train_step(64); std::cout<<"Episode "<<ep<<" winner="<<(winner>0?"White":(winner<0?"Black":"Draw"))<<" buffer="<<buffer.size()<<"\n"; }
	}
	else if(choice==2){ azchess::Position s = azchess::startpos(); std::cout<<"Play as (w/b)? "; char sidec='w'; std::cin>>sidec; bool human_is_white = (sidec!='b'); std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); for(;;){ print_position(s); int w = azchess::winner_on_terminal(s); if(w!=0){ std::cout<<"Result: "<<(w>0?"1-0":(w<0?"0-1":"1/2-1/2"))<<"\n"; break; } if((s.white_to_move && human_is_white) || (!s.white_to_move && !human_is_white)){ std::string mv; std::cout<<"Your move (uci): "; std::getline(std::cin, mv); azchess::Move m{}; if(uci_to_move(s, mv, m)){ azchess::apply_move(s, m); } else { std::cout<<"Invalid move.\n"; } } else { auto pi = mcts.run(s, false); auto legal = azchess::legal_moves(s); if(legal.empty()){ std::cout<<"No moves.\n"; break; } int best=0; float bestp=-1.0f; for(size_t i=0;i<legal.size();++i){ int j=azchess::az73_index_for(legal[i], s.white_to_move); float p = pi[j]; if(p>bestp){ bestp=p; best=int(i);} } auto bm = legal[size_t(best)]; std::cout<<"Engine: "<<move_to_uci(bm)<<"\n"; azchess::apply_move(s, bm); } }
	}
	else if(choice==3){ azchess::Position p = azchess::startpos(); int depth=3; std::cout<<"Depth? "; std::cin>>depth; auto start=std::chrono::steady_clock::now(); auto nodes = azchess::perft(p, depth); auto end=std::chrono::steady_clock::now(); double ms = std::chrono::duration<double,std::milli>(end-start).count(); std::cout<<"perft("<<depth<<")="<<nodes<<" in "<<ms<<" ms\n"; }
	else if(choice==4){ std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); std::cout<<"Entering UCI mode...\n"; uci_loop(net); std::cout<<"UCI mode exited.\n"; }
	else if(choice==5){ auto st = kllm::nn::save_module(net, "miko.ckpt"); if(!st.ok()) std::cout<<"save failed: "<<st.message<<"\n"; else std::cout<<"Saved.\n"; }
	else if(choice==6){ auto st = kllm::nn::load_module(net, "miko.ckpt"); if(!st.ok()) std::cout<<"load failed: "<<st.message<<"\n"; else std::cout<<"Loaded.\n"; }
	else if(choice==7){ break; }
	else { std::cout<<"Unknown option.\n"; }
	}
	return 0; }