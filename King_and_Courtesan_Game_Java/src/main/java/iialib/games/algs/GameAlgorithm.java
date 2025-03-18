package iialib.games.algs;

import iialib.games.model.IBoard;
import iialib.games.model.IMove;
import iialib.games.model.IRole;

public interface GameAlgorithm<Move extends IMove, Role extends IRole, Board extends IBoard<Move, Role, Board>> {

    Move bestMove(Board board, Role playerRole);

}
