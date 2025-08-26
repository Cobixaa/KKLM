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

int main(){ using namespace kllm::nn; kllm::set_num_threads(std::min<unsigned>(8u, std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 8u)); kllm::set_parallel_threshold(1<<14); kllm::set_large_slab_bytes(1<<20); kllm::global_config().enable_matmul_blocked = true; kllm::global_config().matmul_block_m = 64; kllm::global_config().matmul_block_n = 128; kllm::global_config().matmul_block_k = 128; kllm::global_config().release_tls_fused_buffers = true;
	std::mt19937 rng(123); PolicyValueNet net; auto params = collect_parameters(net); Adam opt(params, 1e-3f); MCTSConfig mcfg; mcfg.simulations=256; mcfg.seed=777; MCTS mcts(net, mcfg);
	std::deque<ReplaySample> buffer; const size_t buffer_cap=10000; const size_t batch_size=64;
	auto train_step = [&](size_t steps){ if(buffer.size()<batch_size) return; std::uniform_int_distribution<size_t> uid(0, buffer.size()-1); for(size_t step=0; step<steps; ++step){ std::vector<float> X; X.reserve(batch_size*13*64); std::vector<float> PI; PI.reserve(batch_size*8*8*73); std::vector<float> Z; Z.reserve(batch_size); for(size_t i=0;i<batch_size;++i){ const auto &s=buffer[uid(rng)]; X.insert(X.end(), s.state.begin(), s.state.end()); PI.insert(PI.end(), s.pi.begin(), s.pi.end()); Z.push_back(s.z);} auto x=tensor(X,{batch_size,13*64},false); ValuePtr logits,value; net.forward_both(x,logits,value); auto loss_pi = cross_entropy_logits_soft(logits, PI, 8*8*73); auto zt = tensor(Z,{batch_size,1},false); auto loss_v = mse_loss(value, zt); auto loss = add(loss_pi, loss_v); opt.zero_grad(); loss->backward(); opt.step(); } };
	// perft sanity
	azchess::Position p = azchess::startpos(); unsigned long long p2 = azchess::perft(p, 2); std::cout<<"perft(2)="<<p2<<"\n";
	int episodes=4, max_moves=80; for(int ep=0; ep<episodes; ++ep){ azchess::Position s = azchess::startpos(); std::vector<std::vector<float>> states; std::vector<std::vector<float>> dists; int winner=0; for(int t=0;t<max_moves;++t){ auto pi = mcts.run(s, /*add_noise=*/(t==0)); states.push_back(azchess::encode_features(s)); dists.push_back(pi); auto legal = azchess::legal_moves(s); if(legal.empty()){ break; } std::vector<float> probs; probs.reserve(legal.size()); float sum=0.f; for(const auto &m: legal){ float v = pi[azchess::az73_index_for(m, s.white_to_move)]; probs.push_back(v); sum+=v; } if(sum<=0.f){ for(float &v: probs) v = 1.0f/float(probs.size()); } else { for(float &v: probs) v/=sum; } std::discrete_distribution<int> dd(probs.begin(), probs.end()); int choice=dd(rng); azchess::apply_move(s, legal[size_t(choice)]); int w = azchess::winner_on_terminal(s); if(w!=0){ winner=w; break; } }
		float z_final = (winner==0?0.f:(winner>0?+1.f:-1.f)); bool side=true; for(size_t i=0;i<states.size(); ++i){ float z = side ? z_final : -z_final; buffer.push_back({states[i], dists[i], z}); if(buffer.size()>buffer_cap) buffer.pop_front(); side = !side; } train_step(64); std::cout<<"Episode "<<ep<<" winner="<<(winner>0?"White":(winner<0?"Black":"Draw"))<<" buffer="<<buffer.size()<<"\n"; }
	auto st = kllm::nn::save_module(net, "miko.ckpt"); if(!st.ok()) std::cout<<"save failed: "<<st.message<<"\n"; std::cout<<"Saved and trained legal AlphaZero Miko.\n"; return 0; }