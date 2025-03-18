package iialib.games.algs;

import iialib.games.model.IBoard;
import iialib.games.model.IMove;
import iialib.games.model.IRole;
import iialib.games.model.Player;

public class AIPlayer<Move extends IMove, Role extends IRole, Board extends IBoard<Move, Role, Board>> extends Player<Role> {

    private GameAlgorithm<Move, Role, Board> ai;

    public AIPlayer(Role role) {
        super(role);
    }

    public AIPlayer(Role role, GameAlgorithm<Move, Role, Board> alg) {
        super(role);
        this.ai = alg;
    }

    public void setAi(GameAlgorithm<Move, Role, Board> ai) {
        this.ai = ai;
    }

    public Move bestMove(Board board) {
        return (ai.bestMove(board, this.getRole()));
    }

    public Board playMove(Board board, Move move) {
        return (board.play(move, this.getRole()));
    }

}
