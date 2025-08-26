#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <cstring>
#include <cassert>
#include <algorithm>

namespace azchess {
	using U64 = unsigned long long;
	constexpr U64 ONE = 1ULL;

	enum Color { WHITE = 0, BLACK = 1 };
	enum Piece { PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5 };

	inline int sq(int r, int f){ return r*8 + f; }
	inline int rank_of(int s){ return s/8; }
	inline int file_of(int s){ return s%8; }
	inline U64 bit(int s){ return ONE << static_cast<unsigned>(s); }

	static U64 KnightAtt[64];
	static U64 KingAtt[64];
	static U64 PawnAtt[2][64];
	static U64 FileAMask = 0x0101010101010101ULL;
	static U64 FileHMask = 0x8080808080808080ULL;
	static bool TablesInited = false;

	inline void init_tables(){
		if(TablesInited) return;
		for(int s=0;s<64;++s){
			// Knight
			U64 a = 0;
			int r = rank_of(s), f = file_of(s);
			a |= (r+2<8 && f+1<8) ? bit((r+2)*8 + (f+1)) : 0;
			a |= (r+2<8 && f-1>=0) ? bit((r+2)*8 + (f-1)) : 0;
			a |= (r-2>=0 && f+1<8) ? bit((r-2)*8 + (f+1)) : 0;
			a |= (r-2>=0 && f-1>=0) ? bit((r-2)*8 + (f-1)) : 0;
			a |= (r+1<8 && f+2<8) ? bit((r+1)*8 + (f+2)) : 0;
			a |= (r+1<8 && f-2>=0) ? bit((r+1)*8 + (f-2)) : 0;
			a |= (r-1>=0 && f+2<8) ? bit((r-1)*8 + (f+2)) : 0;
			a |= (r-1>=0 && f-2>=0) ? bit((r-1)*8 + (f-2)) : 0;
			KnightAtt[s] = a;
			// King
			U64 k=0;
			for(int dr=-1; dr<=1; ++dr){ for(int df=-1; df<=1; ++df){ if(!dr && !df) continue; int rr=r+dr, ff=f+df; if(rr>=0&&rr<8&&ff>=0&&ff<8) k |= bit(rr*8+ff); }}
			KingAtt[s]=k;
			// Pawn attacks
			U64 w=0,bm=0;
			if(r+1<8){ if(f-1>=0) w |= bit((r+1)*8 + (f-1)); if(f+1<8) w |= bit((r+1)*8 + (f+1)); }
			if(r-1>=0){ if(f-1>=0) bm |= bit((r-1)*8 + (f-1)); if(f+1<8) bm |= bit((r-1)*8 + (f+1)); }
			PawnAtt[WHITE][s]=w; PawnAtt[BLACK][s]=bm;
		}
		TablesInited=true;
	}

	inline U64 north_fill(U64 b){ b |= (b<<8); b |= (b<<16); b |= (b<<32); return b; }
	inline U64 south_fill(U64 b){ b |= (b>>8); b |= (b>>16); b |= (b>>32); return b; }

	inline int lsb(U64 b){ assert(b); return __builtin_ctzll(b); }
	inline int msb(U64 b){ assert(b); return 63 - __builtin_clzll(b); }

	inline U64 rook_attacks(int s, U64 occ){
		U64 att=0; int r=rank_of(s), f=file_of(s);
		for(int rr=r+1; rr<8; ++rr){ att|=bit(rr*8+f); if(occ & bit(rr*8+f)) break; }
		for(int rr=r-1; rr>=0; --rr){ att|=bit(rr*8+f); if(occ & bit(rr*8+f)) break; }
		for(int ff=f+1; ff<8; ++ff){ att|=bit(r*8+ff); if(occ & bit(r*8+ff)) break; }
		for(int ff=f-1; ff>=0; --ff){ att|=bit(r*8+ff); if(occ & bit(r*8+ff)) break; }
		return att;
	}
	inline U64 bishop_attacks(int s, U64 occ){
		U64 att=0; int r=rank_of(s), f=file_of(s);
		for(int rr=r+1,ff=f+1; rr<8&&ff<8; ++rr,++ff){ att|=bit(rr*8+ff); if(occ & bit(rr*8+ff)) break; }
		for(int rr=r+1,ff=f-1; rr<8&&ff>=0; ++rr,--ff){ att|=bit(rr*8+ff); if(occ & bit(rr*8+ff)) break; }
		for(int rr=r-1,ff=f+1; rr>=0&&ff<8; --rr,++ff){ att|=bit(rr*8+ff); if(occ & bit(rr*8+ff)) break; }
		for(int rr=r-1,ff=f-1; rr>=0&&ff>=0; --rr,--ff){ att|=bit(rr*8+ff); if(occ & bit(rr*8+ff)) break; }
		return att;
	}

	struct Move {
		int from=0, to=0;
		int piece=PAWN; // moving piece type
		int capture=-1; // captured piece type or -1
		int promote=-1; // promotion target piece type (KNIGHT/BISHOP/ROOK/QUEEN) or -1
		bool enpassant=false;
		bool castle=false; // true for king move that castles
	};

	struct Position {
		U64 pieces[2][6]{};
		U64 occ[2]{};
		U64 occ_all=0;
		int castling=0; // 1:WK,2:WQ,4:BK,8:BQ
		int ep_square=-1;
		bool white_to_move=true;
		int halfmove=0, fullmove=1;
	};

	inline void update_occ(Position &p){ p.occ[WHITE]=0; p.occ[BLACK]=0; for(int pc=0;pc<6;++pc){ p.occ[WHITE]|=p.pieces[WHITE][pc]; p.occ[BLACK]|=p.pieces[BLACK][pc]; } p.occ_all = p.occ[WHITE]|p.occ[BLACK]; }

	inline Position startpos(){ init_tables(); Position p{}; p.white_to_move=true; p.castling=1|2|4|8; p.ep_square=-1; p.halfmove=0; p.fullmove=1;
		// Pawns
		for(int f=0; f<8; ++f){ p.pieces[WHITE][PAWN] |= bit(sq(1,f)); p.pieces[BLACK][PAWN] |= bit(sq(6,f)); }
		// Rooks
		p.pieces[WHITE][ROOK] |= bit(sq(0,0))|bit(sq(0,7)); p.pieces[BLACK][ROOK] |= bit(sq(7,0))|bit(sq(7,7));
		// Knights
		p.pieces[WHITE][KNIGHT] |= bit(sq(0,1))|bit(sq(0,6)); p.pieces[BLACK][KNIGHT] |= bit(sq(7,1))|bit(sq(7,6));
		// Bishops
		p.pieces[WHITE][BISHOP] |= bit(sq(0,2))|bit(sq(0,5)); p.pieces[BLACK][BISHOP] |= bit(sq(7,2))|bit(sq(7,5));
		// Queens
		p.pieces[WHITE][QUEEN] |= bit(sq(0,3)); p.pieces[BLACK][QUEEN] |= bit(sq(7,3));
		// Kings
		p.pieces[WHITE][KING] |= bit(sq(0,4)); p.pieces[BLACK][KING] |= bit(sq(7,4));
		update_occ(p); return p; }

	inline U64 attacks_for(const Position &p, int side){
		U64 occ = p.occ_all; U64 att = 0;
		U64 bb;
		// Pawns
		bb = p.pieces[side][PAWN];
		while(bb){ int s = lsb(bb); bb &= bb-1; att |= PawnAtt[side][s]; }
		// Knights
		bb = p.pieces[side][KNIGHT]; while(bb){ int s=lsb(bb); bb&=bb-1; att |= KnightAtt[s]; }
		// Bishops/Queens
		bb = p.pieces[side][BISHOP]; while(bb){ int s=lsb(bb); bb&=bb-1; att |= bishop_attacks(s, occ); }
		bb = p.pieces[side][ROOK]; while(bb){ int s=lsb(bb); bb&=bb-1; att |= rook_attacks(s, occ); }
		bb = p.pieces[side][QUEEN]; while(bb){ int s=lsb(bb); bb&=bb-1; att |= bishop_attacks(s, occ) | rook_attacks(s, occ); }
		// King
		bb = p.pieces[side][KING]; while(bb){ int s=lsb(bb); bb&=bb-1; att |= KingAtt[s]; }
		return att;
	}

	inline bool in_check(const Position &p, int side){ U64 kbb = p.pieces[side][KING]; if(!kbb) return false; int ks = lsb(kbb); U64 opp_att = attacks_for(p, side^1); return (opp_att & bit(ks)) != 0; }

	inline void apply_move(Position &p, const Move &m){
		int side = p.white_to_move ? WHITE : BLACK;
		int opside = side^1;
		U64 fromb = bit(m.from), tob = bit(m.to);
		// Remove moving piece from origin
		p.pieces[side][m.piece] &= ~fromb;
		// Handle captures
		if(m.enpassant){ int cap_sq = p.white_to_move ? (m.to-8) : (m.to+8); p.pieces[opside][PAWN] &= ~bit(cap_sq); }
		else if(m.capture>=0){ p.pieces[opside][m.capture] &= ~tob; }
		// Promotions
		if(m.promote>=0){ p.pieces[side][m.promote] |= tob; }
		else { p.pieces[side][m.piece] |= tob; }
		// Castling move (king move with rook hop)
		if(m.castle){ if(side==WHITE){ if(m.to==sq(0,6)){ // O-O
			p.pieces[WHITE][ROOK] &= ~bit(sq(0,7)); p.pieces[WHITE][ROOK] |= bit(sq(0,5)); }
			else if(m.to==sq(0,2)){ p.pieces[WHITE][ROOK] &= ~bit(sq(0,0)); p.pieces[WHITE][ROOK] |= bit(sq(0,3)); }
		} else { if(m.to==sq(7,6)){ p.pieces[BLACK][ROOK] &= ~bit(sq(7,7)); p.pieces[BLACK][ROOK] |= bit(sq(7,5)); }
			else if(m.to==sq(7,2)){ p.pieces[BLACK][ROOK] &= ~bit(sq(7,0)); p.pieces[BLACK][ROOK] |= bit(sq(7,3)); } }
		}
		// Update castling rights
		if(m.piece==KING){ if(side==WHITE){ p.castling &= ~(1|2); } else { p.castling &= ~(4|8); } }
		if(m.piece==ROOK){ if(side==WHITE){ if(m.from==sq(0,0)) p.castling &= ~2; if(m.from==sq(0,7)) p.castling &= ~1; } else { if(m.from==sq(7,0)) p.castling &= ~8; if(m.from==sq(7,7)) p.castling &= ~4; } }
		if(m.capture>=0 && m.capture==ROOK){ if(opside==WHITE){ if(m.to==sq(0,0)) p.castling &= ~2; if(m.to==sq(0,7)) p.castling &= ~1; } else { if(m.to==sq(7,0)) p.castling &= ~8; if(m.to==sq(7,7)) p.castling &= ~4; } }
		// En passant target
		p.ep_square = -1;
		if(m.piece==PAWN){ int dr = rank_of(m.to)-rank_of(m.from); if(dr==2 || dr==-2){ p.ep_square = (m.from + m.to)/2; } }
		// Halfmove clock
		if(m.piece==PAWN || m.capture>=0) p.halfmove=0; else p.halfmove++;
		if(!p.white_to_move) p.fullmove++;
		p.white_to_move = !p.white_to_move; update_occ(p);
	}

	inline void undo_not_supported(){}

	inline void gen_pseudo(const Position &p, std::vector<Move> &out){
		int side = p.white_to_move?WHITE:BLACK; int opside=side^1;
		U64 friend_occ = p.occ[side]; U64 enemy_occ = p.occ[opside];
		// Pawns
		U64 pawns = p.pieces[side][PAWN];
		if(side==WHITE){
			U64 single = (pawns<<8) & ~p.occ_all;
			U64 dbl = ((single & (0x000000000000FF00ULL<<8))<<8) & ~p.occ_all; // from rank2 to rank4
			U64 promo_rank = 0xFF00000000000000ULL; // rank 8
			U64 m=single; while(m){ int to=lsb(m); m&=m-1; int from=to-8; bool promo = (bit(to)&promo_rank)!=0; if(promo){ for(int pr=KNIGHT; pr<=QUEEN; ++pr){ out.push_back({from,to,PAWN,-1,pr,false,false}); } } else { out.push_back({from,to,PAWN,-1,-1,false,false}); } }
			m=dbl; while(m){ int to=lsb(m); m&=m-1; int from=to-16; out.push_back({from,to,PAWN,-1,-1,false,false}); }
			U64 capL = ((pawns & ~FileAMask)<<7) & enemy_occ; U64 capR = ((pawns & ~FileHMask)<<9) & enemy_occ;
			m=capL; while(m){ int to=lsb(m); m&=m-1; int from=to-7; bool promo=(bit(to)&promo_rank)!=0; int cap_piece=-1; for(int pc=0;pc<6;++pc) if(p.pieces[opside][pc]&bit(to)){ cap_piece=pc; break; }
				if(promo){ for(int pr=KNIGHT; pr<=QUEEN; ++pr){ out.push_back({from,to,PAWN,cap_piece,pr,false,false}); } } else { out.push_back({from,to,PAWN,cap_piece,-1,false,false}); } }
			m=capR; while(m){ int to=lsb(m); m&=m-1; int from=to-9; bool promo=(bit(to)&promo_rank)!=0; int cap_piece=-1; for(int pc=0;pc<6;++pc) if(p.pieces[opside][pc]&bit(to)){ cap_piece=pc; break; }
				if(promo){ for(int pr=KNIGHT; pr<=QUEEN; ++pr){ out.push_back({from,to,PAWN,cap_piece,pr,false,false}); } }
				else { out.push_back({from,to,PAWN,cap_piece,-1,false,false}); }
			}
			// en passant
			if(p.ep_square>=0){ int eps=p.ep_square; U64 epsb=bit(eps); U64 pl = ((pawns & ~FileAMask)<<7)&epsb; U64 pr = ((pawns & ~FileHMask)<<9)&epsb; if(pl){ int from=eps-7; out.push_back({from,eps,PAWN,PAWN,true,false}); } if(pr){ int from=eps-9; out.push_back({from,eps,PAWN,PAWN,true,false}); }
			}
		} else {
			U64 single = (pawns>>8) & ~p.occ_all;
			U64 dbl = ((single & (0x00FF000000000000ULL>>8))>>8) & ~p.occ_all; // from rank7 to rank5
			U64 promo_rank = 0x00000000000000FFULL; // rank 1
			U64 m=single; while(m){ int to=lsb(m); m&=m-1; int from=to+8; bool promo=(bit(to)&promo_rank)!=0; if(promo){ for(int pr=KNIGHT; pr<=QUEEN; ++pr){ out.push_back({from,to,PAWN,-1,pr,false,false}); } } else { out.push_back({from,to,PAWN,-1,-1,false,false}); } }
			m=dbl; while(m){ int to=lsb(m); m&=m-1; int from=to+16; out.push_back({from,to,PAWN,-1,-1,false,false}); }
			U64 capL = ((pawns & ~FileHMask)>>7) & enemy_occ; U64 capR = ((pawns & ~FileAMask)>>9) & enemy_occ;
			m=capL; while(m){ int to=lsb(m); m&=m-1; int from=to+7; bool promo=(bit(to)&promo_rank)!=0; int cap_piece=-1; for(int pc=0;pc<6;++pc) if(p.pieces[opside][pc]&bit(to)){ cap_piece=pc; break; }
				if(promo){ for(int pr=KNIGHT; pr<=QUEEN; ++pr){ out.push_back({from,to,PAWN,cap_piece,pr,false,false}); } } else { out.push_back({from,to,PAWN,cap_piece,-1,false,false}); } }
			m=capR; while(m){ int to=lsb(m); m&=m-1; int from=to+9; bool promo=(bit(to)&promo_rank)!=0; int cap_piece=-1; for(int pc=0;pc<6;++pc) if(p.pieces[opside][pc]&bit(to)){ cap_piece=pc; break; }
				if(promo){ for(int pr=KNIGHT; pr<=QUEEN; ++pr){ out.push_back({from,to,PAWN,cap_piece,pr,false,false}); } }
				else { out.push_back({from,to,PAWN,cap_piece,-1,false,false}); }
			}
			if(p.ep_square>=0){ int eps=p.ep_square; U64 epsb=bit(eps); U64 pl = ((pawns & ~FileHMask)>>7)&epsb; U64 pr = ((pawns & ~FileAMask)>>9)&epsb; if(pl){ int from=eps+7; out.push_back({from,eps,PAWN,PAWN,true,false}); } if(pr){ int from=eps+9; out.push_back({from,eps,PAWN,PAWN,true,false}); }
			}
		}
		// Knights
		U64 bb = p.pieces[side][KNIGHT]; while(bb){ int s=lsb(bb); bb&=bb-1; U64 att = KnightAtt[s] & ~friend_occ; U64 m=att; while(m){ int to=lsb(m); m&=m-1; int cap=-1; if(enemy_occ & bit(to)){ for(int pc=0;pc<6;++pc) if(p.pieces[opside][pc]&bit(to)){ cap=pc; break; } } out.push_back({s,to,KNIGHT,cap,-1,false,false}); }}
		// Bishops
		bb = p.pieces[side][BISHOP]; while(bb){ int s=lsb(bb); bb&=bb-1; U64 att = bishop_attacks(s, p.occ_all) & ~friend_occ; U64 m=att; while(m){ int to=lsb(m); m&=m-1; int cap=-1; if(enemy_occ & bit(to)){ for(int pc=0;pc<6;++pc) if(p.pieces[opside][pc]&bit(to)){ cap=pc; break; } } out.push_back({s,to,BISHOP,cap,-1,false,false}); }}
		// Rooks
		bb = p.pieces[side][ROOK]; while(bb){ int s=lsb(bb); bb&=bb-1; U64 att = rook_attacks(s, p.occ_all) & ~friend_occ; U64 m=att; while(m){ int to=lsb(m); m&=m-1; int cap=-1; if(enemy_occ & bit(to)){ for(int pc=0;pc<6;++pc) if(p.pieces[opside][pc]&bit(to)){ cap=pc; break; } } out.push_back({s,to,ROOK,cap,-1,false,false}); }}
		// Queens
		bb = p.pieces[side][QUEEN]; while(bb){ int s=lsb(bb); bb&=bb-1; U64 att = (rook_attacks(s, p.occ_all)|bishop_attacks(s, p.occ_all)) & ~friend_occ; U64 m=att; while(m){ int to=lsb(m); m&=m-1; int cap=-1; if(enemy_occ & bit(to)){ for(int pc=0;pc<6;++pc) if(p.pieces[opside][pc]&bit(to)){ cap=pc; break; } } out.push_back({s,to,QUEEN,cap,-1,false,false}); }}
		// King
		bb = p.pieces[side][KING]; while(bb){ int s=lsb(bb); bb&=bb-1; U64 att = KingAtt[s] & ~friend_occ; U64 m=att; while(m){ int to=lsb(m); m&=m-1; int cap=-1; if(enemy_occ & bit(to)){ for(int pc=0;pc<6;++pc) if(p.pieces[opside][pc]&bit(to)){ cap=pc; break; } } out.push_back({s,to,KING,cap,-1,false,false}); }
			// Castling
			if(side==WHITE){
				if((p.castling&1) && !(p.occ_all & (bit(sq(0,5))|bit(sq(0,6))))){ // O-O empty
					// Squares under attack e1,f1,g1
					Position t=p; if(!in_check(t,WHITE)){ // e1 not in check
						// simulate f1
						Move mv{s, sq(0,5), KING, -1, -1, false, false}; apply_move(t, mv);
						if(!in_check(t,WHITE)){
							// simulate g1
							mv = Move{sq(0,5), sq(0,6), KING, -1, -1, false, false}; apply_move(t, mv);
							if(!in_check(t,WHITE)) out.push_back({s, sq(0,6), KING, -1, -1, false, true});
						}
					}
				}
				if((p.castling&2) && !(p.occ_all & (bit(sq(0,1))|bit(sq(0,2))|bit(sq(0,3))))){
					Position t=p; if(!in_check(t,WHITE)){
						Move mv{s, sq(0,3), KING, -1, -1, false, false}; apply_move(t,mv);
						if(!in_check(t,WHITE)){
							mv = Move{sq(0,3), sq(0,2), KING, -1, -1, false, false}; apply_move(t,mv);
							if(!in_check(t,WHITE)) out.push_back({s, sq(0,2), KING, -1, -1, false, true});
						}
					}
				}
			} else {
				if((p.castling&4) && !(p.occ_all & (bit(sq(7,5))|bit(sq(7,6))))){
					Position t=p; if(!in_check(t,BLACK)){
						Move mv{s, sq(7,5), KING, -1, -1, false, false}; apply_move(t,mv); if(!in_check(t,BLACK)){ mv = Move{sq(7,5), sq(7,6), KING, -1, -1, false, false}; apply_move(t,mv); if(!in_check(t,BLACK)) out.push_back({s, sq(7,6), KING, -1, -1, false, true}); }
					}
				}
				if((p.castling&8) && !(p.occ_all & (bit(sq(7,1))|bit(sq(7,2))|bit(sq(7,3))))){
					Position t=p; if(!in_check(t,BLACK)){
						Move mv{s, sq(7,3), KING, -1, -1, false, false}; apply_move(t,mv); if(!in_check(t,BLACK)){ mv = Move{sq(7,3), sq(7,2), KING, -1, -1, false, false}; apply_move(t,mv); if(!in_check(t,BLACK)) out.push_back({s, sq(7,2), KING, -1, -1, false, true}); }
					}
				}
			}
		}
	}

	inline std::vector<Move> legal_moves(const Position &p){ std::vector<Move> mv; gen_pseudo(p, mv); std::vector<Move> out; out.reserve(mv.size()); for(const auto &m : mv){ Position t=p; apply_move(t, m); if(!in_check(t, p.white_to_move?WHITE:BLACK)) out.push_back(m); } return out; }

	inline bool no_legal_moves(const Position &p){ auto mv = legal_moves(p); return mv.empty(); }

	inline int winner_on_terminal(const Position &p){ // +1 white, -1 black, 0 draw
		if(!no_legal_moves(p)) return 0; // not terminal by moves
		if(in_check(p, p.white_to_move?WHITE:BLACK)){ return p.white_to_move? -1 : +1; } // side to move checkmated
		return 0; // stalemate draw
	}

	// AlphaZero 8x8x73 mapping
	inline int az73_plane_for(const Move &m, bool white_to_move){
		int fr = rank_of(m.from), ff = file_of(m.from);
		int tr = rank_of(m.to), tf = file_of(m.to);
		int dr = tr - fr, df = tf - ff;
		int dr_rel = white_to_move ? dr : -dr;
		int df_rel = white_to_move ? df : -df;
		// Knight moves
		if(m.piece==KNIGHT){
			static const int kdr[8] = {2,2,1,1,-1,-1,-2,-2};
			static const int kdf[8] = {1,-1,2,-2,2,-2,1,-1};
			for(int i=0;i<8;++i){ if(dr_rel==kdr[i] && df_rel==kdf[i]) return 56 + i; }
		}
		// Underpromotions (KNIGHT,BISHOP,ROOK) in dirs: forward(0), diag-right(1), diag-left(2)
		if(m.piece==PAWN && m.promote>=0 && m.promote!=QUEEN){ int dir3 = 0; if(df_rel==0) dir3=0; else if(df_rel>0) dir3=1; else dir3=2; int promIndex = (m.promote==KNIGHT?0:(m.promote==BISHOP?1:2)); return 56 + 8 + dir3*3 + promIndex; }
		// Sliding / pawn non-underpromotions treated as queenlike
		int dir=-1; int dist=0; int adr = dr_rel==0?0:(dr_rel>0?1:-1); int adf = df_rel==0?0:(df_rel>0?1:-1);
		if(adr==1 && adf==0) dir=0; // N
		else if(adr==-1 && adf==0) dir=1; // S
		else if(adr==0 && adf==1) dir=2; // E
		else if(adr==0 && adf==-1) dir=3; // W
		else if(adr==1 && adf==1) dir=4; // NE
		else if(adr==1 && adf==-1) dir=5; // NW
		else if(adr==-1 && adf==1) dir=6; // SE
		else if(adr==-1 && adf==-1) dir=7; // SW
		dist = std::max(std::abs(dr_rel), std::abs(df_rel)); if(dist<1) dist=1; if(dist>7) dist=7;
		return dir<0 ? 0 : (dir*7 + (dist-1));
	}

	inline int az73_index_for(const Move &m, bool white_to_move){ int plane = az73_plane_for(m, white_to_move); return m.from*73 + plane; }

	inline std::vector<float> encode_features(const Position &p){ std::vector<float> x(13*64, 0.0f); for(int s=0;s<64;++s){ for(int c=0;c<2;++c){ for(int pc=0;pc<6;++pc){ if(p.pieces[c][pc] & bit(s)){ int plane = (c==WHITE?0:6) + pc; x[plane*64 + s] = 1.0f; } } } }
		for(int s=0;s<64;++s) x[12*64 + s] = p.white_to_move ? 1.0f : 0.0f; return x; }

	inline unsigned long long perft(Position &p, int depth){ if(depth==0) return 1ULL; auto mv = legal_moves(p); if(depth==1) return (unsigned long long)mv.size(); unsigned long long nodes=0; for(const auto &m: mv){ Position t=p; apply_move(t,m); nodes += perft(t, depth-1); } return nodes; }
}